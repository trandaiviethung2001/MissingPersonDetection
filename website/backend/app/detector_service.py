from __future__ import annotations

import base64
from concurrent.futures import Future, ThreadPoolExecutor
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2

from .settings import AppSettings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detect_missing_person import (  # noqa: E402
    MissingPersonPipeline,
    draw_info_overlay,
    draw_predicted_track,
    draw_tracked_person,
)
from utils import TrackState  # noqa: E402


@dataclass(slots=True)
class RuntimeStats:
    frame_count: int = 0
    processed_count: int = 0
    started_at: float = 0.0
    last_telemetry_at: float = 0.0
    persons_count: int = 0
    faces_count: int = 0
    match_count: int = 0
    active_tracks: int = 0


class DetectorRuntime:
    """Long-lived camera preview + detection runtime for the dashboard."""

    def __init__(self, settings: AppSettings, broadcaster: Callable[[dict[str, Any]], None]) -> None:
        self.settings = settings
        self._broadcast = broadcaster
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._capture: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None
        self._pipeline: MissingPersonPipeline | None = None

        self._stats = RuntimeStats()
        self._mission_active = False
        self._manual_status = "IDLE"
        self._recording_enabled = False
        self._latest_frame = None
        self._latest_tracked: list[dict[str, Any]] = []
        self._latest_tracked_updated_at = 0.0
        self._track_hold_seconds = 1.5
        self._current_detection: dict[str, Any] | None = None
        self._last_detection_sent: dict[str, Any] | None = None
        self._camera_opened = False
        self._last_error: str | None = None
        self._detection_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detector-runtime")
        self._pending_detection: Future[dict[str, Any]] | None = None

    def ensure_started(self) -> None:
        with self._lock:
            self._ensure_thread()

    def start(self) -> None:
        with self._lock:
            self._ensure_thread()
            self._mission_active = True
            self._manual_status = "SCANNING"
        self._send_status("SCANNING")

    def pause(self) -> None:
        with self._lock:
            self._mission_active = False
            self._manual_status = "PAUSED"
        self._send_status("PAUSED")

    def return_to_home(self) -> None:
        with self._lock:
            self._mission_active = False
            self._manual_status = "RTL"
        self._send_status("RTL")

    def emergency_stop(self) -> None:
        with self._lock:
            self._mission_active = False
            self._manual_status = "EMERGENCY"
        self._send_status("EMERGENCY")

    def set_recording(self, enabled: bool) -> None:
        with self._lock:
            self._recording_enabled = enabled
            if not enabled:
                self._close_writer()

        self._broadcast(
            {"type": "ack", "data": {"command": "record", "success": True, "enabled": enabled}}
        )

    def snapshot(self) -> None:
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()

        success = False
        file_path = None
        if frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.settings.snapshot_dir / f"snapshot_{timestamp}.jpg"
            success = bool(cv2.imwrite(str(file_path), frame))

        self._broadcast(
            {
                "type": "ack",
                "data": {
                    "command": "snapshot",
                    "success": success,
                    "path": str(file_path) if file_path else None,
                },
            }
        )

    def force_lock(self, target_id: str | None) -> None:
        target = (target_id or "").strip().upper()
        locked = False

        with self._lock:
            pipeline = self._pipeline
            if pipeline and pipeline.person_tracker:
                for tracked in pipeline.person_tracker.tracked_persons.values():
                    if f"T{tracked.track_id}" != target:
                        continue
                    tracked.state = TrackState.LOCKED
                    tracked.frames_since_verify = 0
                    tracked.consecutive_verify_fails = 0
                    self._mission_active = True
                    self._manual_status = "LOCKED"
                    locked = True
                    break

        self._broadcast(
            {
                "type": "ack",
                "data": {"command": "lock_target", "success": locked, "targetId": target_id},
            }
        )
        if locked:
            self._send_status("LOCKED")

    def get_runtime_info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "camera_source": self.settings.camera_source,
                "camera_width": self.settings.camera_width,
                "camera_height": self.settings.camera_height,
                "frame_skip": self.settings.frame_skip,
                "threshold": self.settings.threshold,
                "db_path": str(self.settings.db_path),
                "recording_enabled": self._recording_enabled,
                "mission_active": self._mission_active,
                "status": self._effective_status(),
                "camera_opened": self._camera_opened,
                "last_error": self._last_error,
            }

    def current_status_message(self) -> dict[str, Any]:
        return {"type": "status", "data": {"system": self._effective_status()}}

    def current_detection_message(self) -> dict[str, Any]:
        with self._lock:
            detection = self._current_detection or self._empty_detection()
        return {"type": "detection", "data": detection}

    def current_telemetry_message(self) -> dict[str, Any]:
        return {"type": "telemetry", "data": self._build_telemetry()}

    def shutdown(self) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=5)

        pending = self._pending_detection
        if pending is not None:
            pending.cancel()
        self._detection_executor.shutdown(wait=False, cancel_futures=True)

        with self._lock:
            self._release_capture()
            self._close_writer()
            self._thread = None

    def reload_pipeline(self) -> None:
        fresh_pipeline = MissingPersonPipeline(
            threshold=self.settings.threshold,
            frame_skip=self.settings.frame_skip,
            db_path=str(self.settings.db_path),
        )
        with self._lock:
            self._pipeline = fresh_pipeline
            self._last_error = None

    def _ensure_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="detector-runtime", daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        try:
            self._initialize_pipeline()
            self._open_capture()
            self._stats.started_at = time.time()
            last_stream_sent = 0.0

            while not self._stop_event.is_set():
                frame = self._read_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                now = time.perf_counter()

                with self._lock:
                    self._latest_frame = frame.copy()
                    mission_active = self._mission_active
                    recording_enabled = self._recording_enabled
                self._resolve_pending_detection(allow_apply=mission_active)

                if mission_active:
                    self._schedule_detection(frame)
                else:
                    with self._lock:
                        self._latest_tracked = []
                        self._latest_tracked_updated_at = 0.0
                    self._send_detection_if_changed(self._empty_detection())

                stream_frame = None
                if now - last_stream_sent >= max(0.01, 1.0 / self.settings.stream_fps):
                    stream_frame = self._frame_for_stream(frame)
                    self._broadcast_frame(stream_frame)
                    last_stream_sent = now

                self._maybe_send_telemetry(time.time())

                if recording_enabled:
                    if stream_frame is None:
                        stream_frame = self._frame_for_stream(frame)
                    self._write_recording(stream_frame)
        except Exception as exc:
            with self._lock:
                self._mission_active = False
                self._manual_status = "EMERGENCY"
                self._last_error = str(exc)
            self._broadcast(
                {
                    "type": "ack",
                    "data": {"command": "runtime_error", "success": False, "detail": str(exc)},
                }
            )
            self._broadcast({"type": "status", "data": {"system": "EMERGENCY"}})
        finally:
            self._resolve_pending_detection(allow_apply=False)
            with self._lock:
                self._release_capture()
                self._close_writer()

    def _initialize_pipeline(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                return
            self._pipeline = MissingPersonPipeline(
                threshold=self.settings.threshold,
                frame_skip=self.settings.frame_skip,
                db_path=str(self.settings.db_path),
            )

    def _open_capture(self) -> None:
        with self._lock:
            if self._capture is not None and self._capture.isOpened():
                return
            self._capture = cv2.VideoCapture(self.settings.camera_source)
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera_width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera_height)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self._capture.isOpened():
                self._camera_opened = False
                raise RuntimeError(f"Cannot open camera/video source: {self.settings.camera_source}")
            self._camera_opened = True
            self._last_error = None

    def _read_frame(self):
        with self._lock:
            capture = self._capture

        if capture is None:
            return None

        ok, frame = capture.read()
        if ok:
            return self._normalize_frame(frame)

        with self._lock:
            self._release_capture()
            try:
                self._open_capture()
            except RuntimeError as exc:
                self._last_error = str(exc)
                return None
        return None

    def _normalize_frame(self, frame):
        target_size = (self.settings.camera_width, self.settings.camera_height)
        current_size = (frame.shape[1], frame.shape[0])
        if current_size == target_size:
            return frame
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    def _process_detection_frame(self, frame, pipeline: MissingPersonPipeline) -> dict[str, Any]:
        result = pipeline.process_frame(frame)

        strongest = self._pick_strongest_track(result["tracking"]["tracked"])
        detection = self._build_detection_payload(strongest, frame.shape)
        active_tracks = len(pipeline.person_tracker._active_tracks()) if pipeline.person_tracker else 0
        return {
            "frame": result["frame"],
            "processed": result["processed"],
            "tracked": result["tracking"]["tracked"],
            "detection": detection,
            "status": "LOCKED" if strongest and strongest["state"] == TrackState.LOCKED else "SCANNING",
            "frame_count": pipeline.frame_count,
            "processed_count": pipeline.processed_count,
            "active_tracks": active_tracks,
            "persons_count": len(result["persons"]),
            "faces_count": len(result["faces"]),
            "match_count": result["match_count"],
        }

    def _pick_strongest_track(self, tracked: list[dict[str, Any]]) -> dict[str, Any] | None:
        candidates = [item for item in tracked if item["state"] in {TrackState.WATCHING, TrackState.LOCKED}]
        if not candidates:
            return None

        def score(item: dict[str, Any]) -> tuple[int, float]:
            is_locked = 1 if item["state"] == TrackState.LOCKED else 0
            similarity = float(item.get("best_similarity") or item.get("similarity") or 0.0)
            return (is_locked, similarity)

        return max(candidates, key=score)

    def _build_detection_payload(self, track: dict[str, Any] | None, shape: tuple[int, ...]) -> dict[str, Any]:
        if track is None:
            return self._empty_detection()

        height, width = shape[:2]
        x1, y1, x2, y2 = track["bbox"]
        similarity = float(track.get("best_similarity") or track.get("similarity") or 0.0)
        state = track["state"].value if hasattr(track["state"], "value") else str(track["state"])
        return {
            "targetId": f"T{track['track_id']}",
            "personName": track.get("person_name") or "Unknown",
            "confidence": round(similarity * 100),
            "bbox": {
                "x": max(0.0, min(1.0, x1 / width)),
                "y": max(0.0, min(1.0, y1 / height)),
                "w": max(0.0, min(1.0, (x2 - x1) / width)),
                "h": max(0.0, min(1.0, (y2 - y1) / height)),
            },
            "status": state,
        }

    @staticmethod
    def _empty_detection() -> dict[str, Any]:
        return {
            "targetId": "—",
            "personName": None,
            "confidence": 0,
            "bbox": None,
            "status": "IDLE",
        }

    def _send_detection_if_changed(self, payload: dict[str, Any]) -> None:
        changed = payload != self._last_detection_sent
        with self._lock:
            self._current_detection = payload
            if changed:
                self._last_detection_sent = payload.copy()
        if changed:
            self._broadcast({"type": "detection", "data": payload})

    def _send_status(self, status: str) -> None:
        with self._lock:
            if not self._mission_active and status in {"SCANNING", "LOCKED"}:
                return
            self._manual_status = status
        self._broadcast({"type": "status", "data": {"system": status}})

    def _schedule_detection(self, frame) -> None:
        with self._lock:
            if self._pipeline is None:
                return
            pending = self._pending_detection
            if pending is not None and not pending.done():
                return
            pipeline = self._pipeline

        self._pending_detection = self._detection_executor.submit(
            self._process_detection_frame,
            frame.copy(),
            pipeline,
        )

    def _resolve_pending_detection(self, allow_apply: bool) -> None:
        pending = self._pending_detection
        if pending is None or not pending.done():
            return

        self._pending_detection = None
        result = pending.result()
        if not allow_apply:
            return

        now = time.time()
        self._stats.frame_count = result["frame_count"]
        self._stats.processed_count = result["processed_count"]
        self._stats.active_tracks = result["active_tracks"]

        if not result["processed"]:
            return

        with self._lock:
            if result["tracked"]:
                self._latest_tracked = list(result["tracked"])
                self._latest_tracked_updated_at = now
            elif now - self._latest_tracked_updated_at > self._track_hold_seconds:
                self._latest_tracked = []
            holding_previous_track = not result["tracked"] and bool(self._latest_tracked)
        self._stats.persons_count = result["persons_count"]
        self._stats.faces_count = result["faces_count"]
        self._stats.match_count = result["match_count"]
        if holding_previous_track:
            return
        self._send_detection_if_changed(result["detection"])
        self._send_status(result["status"])

    def _frame_for_stream(self, fallback_frame):
        with self._lock:
            mission_active = self._mission_active
            tracked = list(self._latest_tracked)
            stats = {
                "frame_count": self._stats.frame_count,
                "processed_count": self._stats.processed_count,
                "persons_count": self._stats.persons_count,
                "faces_count": self._stats.faces_count,
                "match_count": self._stats.match_count,
                "active_tracks": self._stats.active_tracks,
            }
            pipeline = self._pipeline

        if not mission_active:
            return fallback_frame

        frame = fallback_frame.copy()
        if pipeline is not None and pipeline.person_tracker is not None:
            predicted_ids = set()
            for pred in pipeline.person_tracker.peek_predicted_boxes():
                predicted_ids.add(int(pred["track_id"]))
                draw_predicted_track(frame, pred)

            for track in tracked:
                if track["state"] == TrackState.LOCKED and int(track["track_id"]) in predicted_ids:
                    continue
                draw_tracked_person(frame, track)

        draw_info_overlay(
            frame,
            stats["frame_count"],
            pipeline.current_fps() if pipeline is not None else 0.0,
            stats["persons_count"],
            stats["faces_count"],
            stats["match_count"],
            stats["active_tracks"],
        )
        return frame

    def _effective_status(self) -> str:
        with self._lock:
            if self._mission_active and self._current_detection:
                return "LOCKED" if self._current_detection.get("status") == "LOCKED" else "SCANNING"
            return self._manual_status

    def _broadcast_frame(self, frame) -> None:
        encoded = self._encode_frame(frame)
        if encoded is None:
            return
        self._broadcast({"type": "frame", "data": encoded})

    def _encode_frame(self, frame) -> str | None:
        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.jpeg_quality],
        )
        if not ok:
            return None
        return base64.b64encode(buffer.tobytes()).decode("ascii")

    def _maybe_send_telemetry(self, now: float) -> None:
        if now - self._stats.last_telemetry_at < 1.0:
            return
        self._stats.last_telemetry_at = now
        self._broadcast({"type": "telemetry", "data": self._build_telemetry()})

    def _build_telemetry(self) -> dict[str, Any]:
        elapsed = max(0.001, time.time() - self._stats.started_at) if self._stats.started_at else 0.001
        processed_fps = self._stats.processed_count / elapsed if elapsed > 0 else 0.0
        mission_active = self._mission_active
        return {
            "altitude": 12 if mission_active else 0,
            "speed": round(min(40.0, processed_fps * 2.0), 1),
            "battery": 100,
            "signal": 100,
            "processingFps": round(processed_fps, 2),
            "cameraOpened": self._camera_opened,
        }

    def _write_recording(self, frame) -> None:
        with self._lock:
            if self._writer is None:
                self._writer = self._create_writer(frame)
            writer = self._writer
        if writer is not None:
            writer.write(frame)

    def _create_writer(self, frame) -> cv2.VideoWriter:
        height, width = frame.shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.settings.recording_dir / f"recording_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(path), fourcc, max(1.0, self.settings.stream_fps), (width, height))

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._camera_opened = False

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
