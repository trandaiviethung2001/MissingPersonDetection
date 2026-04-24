from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

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


class UploadService:
    """Headless video-upload detection pipeline.

    Runs the same detection + tracking code the live dashboard uses, but on a
    file instead of a camera stream. Calls are serialized via an asyncio lock
    because the underlying pipeline is CPU-bound and sharing the interpreter
    across parallel uploads would just thrash.
    """

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._lock = asyncio.Lock()

    async def process(
        self,
        input_path: Path,
        output_path: Path,
        threshold: float | None = None,
        frame_skip: int | None = None,
    ) -> list[dict[str, Any]]:
        async with self._lock:
            return await asyncio.to_thread(
                _run_detection_sync,
                str(input_path),
                str(output_path),
                threshold if threshold is not None else self.settings.threshold,
                frame_skip if frame_skip is not None else self.settings.frame_skip,
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
) -> list[dict[str, Any]]:
    """Thread-pool worker — lazy imports so startup stays snappy."""
    from detect_missing_person import main as run_detection

    result = run_detection(
        video_path=video_path,
        output_path=output_path,
        threshold=float(threshold),
        frame_skip=int(frame_skip),
        no_display=True,
    )
    # Pipeline returns bboxes as numpy-int tuples; make the log JSON-safe
    # before it leaves the worker thread.
    return _to_jsonable(result or [])
