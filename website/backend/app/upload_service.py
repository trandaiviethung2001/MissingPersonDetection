from __future__ import annotations

import asyncio
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .settings import AppSettings


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalar / array / nested containers to plain Python types.

    FastAPI/pydantic's JSON serializer can't handle numpy.int64 / float32 etc.,
    and the detection pipeline often returns bbox tuples with numpy scalars
    (ByteTrack's xyxy -> .astype(int) yields numpy ints).
    """
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# How many finished jobs to keep around for status polling. Older jobs are
# evicted in FIFO order so the dict can't grow unbounded over a long session.
MAX_JOB_HISTORY = 25


class UploadService:
    """Headless video-upload detection pipeline with job tracking.

    The frontend now does:
        1. POST /api/upload         -> creates a job, returns its id immediately
        2. GET  /api/upload/{id}    -> polls progress until state == "done"

    so that a loading bar can show real per-frame progress while the
    CPU-bound detection runs in the background. CPU work is still
    serialized via an asyncio lock — only one job actually executes at a
    time even if several are queued.
    """

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._cpu_lock = asyncio.Lock()
        self._jobs_lock = threading.Lock()
        self._jobs: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def create_job(
        self,
        input_path: Path,
        output_path: Path,
        original_filename: str,
        threshold: float | None,
        frame_skip: int | None,
    ) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "state": "queued",          # queued | uploading_done | processing | done | error
            "progress": 0.0,            # 0..1, frames_done / frames_total
            "frames_done": 0,
            "frames_total": 0,
            "error": None,
            "result": None,
            "started_at": time.time(),
            "finished_at": None,
            "original_filename": original_filename,
            "_input_path": str(input_path),
            "_output_path": str(output_path),
            "_threshold": threshold,
            "_frame_skip": frame_skip,
        }
        with self._jobs_lock:
            self._jobs[job_id] = job
            self._evict_old()
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            # Strip private fields before exposing
            return {k: v for k, v in job.items() if not k.startswith("_")}

    def _update_job(self, job_id: str, **changes: Any) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.update(changes)

    def _evict_old(self) -> None:
        # Caller already holds _jobs_lock
        while len(self._jobs) > MAX_JOB_HISTORY:
            self._jobs.popitem(last=False)

    async def run_job(self, job_id: str) -> None:
        """Run the detection pipeline for a queued job. Updates progress as it goes."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            input_path = job["_input_path"]
            output_path = job["_output_path"]
            threshold = job["_threshold"]
            frame_skip = job["_frame_skip"]

        # Serialize CPU-bound work — only one detection runs at a time even if
        # several jobs are queued. Other jobs sit in "queued" state until the
        # lock is free.
        async with self._cpu_lock:
            self._update_job(job_id, state="processing")
            loop = asyncio.get_running_loop()

            def on_progress(done: int, total: int) -> None:
                # Called from the worker thread — schedule a thread-safe update
                self._update_job(
                    job_id,
                    frames_done=int(done),
                    frames_total=int(total),
                    progress=float(done) / max(1.0, float(total)),
                )

            try:
                detections = await loop.run_in_executor(
                    None,
                    _run_detection_sync,
                    input_path,
                    output_path,
                    threshold if threshold is not None else self.settings.threshold,
                    frame_skip if frame_skip is not None else self.settings.frame_skip,
                    on_progress,
                )
                summary = self.summarize(detections)
                self._update_job(
                    job_id,
                    state="done",
                    progress=1.0,
                    finished_at=time.time(),
                    result={
                        "video_url": f"/api/uploads/{Path(output_path).name}",
                        "detections": detections,
                        "summary": summary,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - report any failure to the user
                self._update_job(
                    job_id,
                    state="error",
                    error=str(exc),
                    finished_at=time.time(),
                )

    # ------------------------------------------------------------------
    # Backwards-compatible synchronous helper (used by upload tests)
    # ------------------------------------------------------------------

    async def process(
        self,
        input_path: Path,
        output_path: Path,
        threshold: float | None = None,
        frame_skip: int | None = None,
    ) -> list[dict[str, Any]]:
        async with self._cpu_lock:
            return await asyncio.to_thread(
                _run_detection_sync,
                str(input_path),
                str(output_path),
                threshold if threshold is not None else self.settings.threshold,
                frame_skip if frame_skip is not None else self.settings.frame_skip,
                None,
            )

    @staticmethod
    def summarize(detections: list[dict[str, Any]]) -> dict[str, Any]:
        """Compact summary object for the frontend."""
        if not detections:
            return {
                "count": 0,
                "people": [],
                "text": "No missing persons detected.",
            }

        by_name: dict[str, list[dict[str, Any]]] = {}
        for d in detections:
            by_name.setdefault(d["person"], []).append(d)

        people = []
        for name, dets in by_name.items():
            times = [d["timestamp"] for d in dets]
            sims = [d["similarity"] for d in dets]
            people.append(
                {
                    "name": name,
                    "person_id": dets[0].get("person_id"),
                    "hits": len(dets),
                    "time_start": min(times),
                    "time_end": max(times),
                    "best_similarity": max(sims),
                    "avg_similarity": sum(sims) / len(sims),
                }
            )
        people.sort(key=lambda p: -p["best_similarity"])

        lines = [f"Found {len(detections)} hit(s) for {len(by_name)} person(s):"]
        for p in people:
            lines.append(
                f"  {p['name']}: {p['hits']} hit(s) "
                f"between {p['time_start']:.1f}s and {p['time_end']:.1f}s "
                f"(best {p['best_similarity']:.1%}, avg {p['avg_similarity']:.1%})"
            )
        return {
            "count": len(detections),
            "people": people,
            "text": "\n".join(lines),
        }


def _run_detection_sync(
    video_path: str,
    output_path: str,
    threshold: float,
    frame_skip: int,
    progress_callback: Callable[[int, int], None] | None,
) -> list[dict[str, Any]]:
    """Thread-pool worker — lazy imports so startup stays snappy."""
    from detect_missing_person import main as run_detection

    result = run_detection(
        video_path=video_path,
        output_path=output_path,
        threshold=float(threshold),
        frame_skip=int(frame_skip),
        no_display=True,
        progress_callback=progress_callback,
    )
    # Pipeline returns bboxes as numpy-int tuples; make the log JSON-safe
    # before it leaves the worker thread.
    return _to_jsonable(result or [])
