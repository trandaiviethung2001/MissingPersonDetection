from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_camera_source(value: str) -> int | str:
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


@dataclass(slots=True)
class AppSettings:
    project_root: Path
    website_root: Path
    app_root: Path
    frontend_dir: Path
    snapshot_dir: Path
    recording_dir: Path
    training_dir: Path
    upload_dir: Path
    lock_snapshot_dir: Path
    missing_persons_db_dir: Path
    camera_source: int | str
    camera_width: int
    camera_height: int
    frame_skip: int
    threshold: float
    jpeg_quality: int
    stream_fps: float
    db_path: Path

    @classmethod
    def load(cls) -> "AppSettings":
        app_root = Path(__file__).resolve().parents[1]
        website_root = Path(__file__).resolve().parents[2]
        project_root = Path(__file__).resolve().parents[3]
        frontend_dir = website_root / "frontend"
        snapshot_dir = website_root / "data" / "snapshots"
        recording_dir = website_root / "data" / "recordings"
        training_dir = website_root / "data" / "training_runs"
        upload_dir = website_root / "data" / "uploads"
        lock_snapshot_dir = website_root / "data" / "lock_snapshots"
        missing_persons_db_dir = project_root / "missing_persons_db"

        snapshot_dir.mkdir(parents=True, exist_ok=True)
        recording_dir.mkdir(parents=True, exist_ok=True)
        training_dir.mkdir(parents=True, exist_ok=True)
        upload_dir.mkdir(parents=True, exist_ok=True)
        lock_snapshot_dir.mkdir(parents=True, exist_ok=True)

        db_default = missing_persons_db_dir / "embeddings.pkl"

        return cls(
            project_root=project_root,
            website_root=website_root,
            app_root=app_root,
            frontend_dir=frontend_dir,
            snapshot_dir=snapshot_dir,
            recording_dir=recording_dir,
            training_dir=training_dir,
            upload_dir=upload_dir,
            lock_snapshot_dir=lock_snapshot_dir,
            missing_persons_db_dir=missing_persons_db_dir,
            camera_source=_parse_camera_source(os.getenv("LOST_PERSON_CAMERA_SOURCE", "0")),
            camera_width=int(os.getenv("LOST_PERSON_CAMERA_WIDTH", "1280")),
            camera_height=int(os.getenv("LOST_PERSON_CAMERA_HEIGHT", "720")),
            frame_skip=int(os.getenv("LOST_PERSON_FRAME_SKIP", "5")),
            threshold=float(os.getenv("LOST_PERSON_THRESHOLD", "0.4")),
            jpeg_quality=int(os.getenv("LOST_PERSON_JPEG_QUALITY", "55")),
            stream_fps=float(os.getenv("LOST_PERSON_STREAM_FPS", "24")),
            db_path=Path(os.getenv("LOST_PERSON_DB_PATH", str(db_default))),
        )
