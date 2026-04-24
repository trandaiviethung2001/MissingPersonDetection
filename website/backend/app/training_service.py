from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile

from detect_missing_person import resolve_latest_embeddings_path
from utils.face_recognizer import FaceRecognizer

from .settings import AppSettings


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class TrainingService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def list_people(self) -> list[dict]:
        people = []
        for person_dir in sorted(self.settings.missing_persons_db_dir.glob("person_*")):
            if not person_dir.is_dir():
                continue
            name_file = person_dir / "name.txt"
            name = name_file.read_text(encoding="utf-8").strip() if name_file.exists() else person_dir.name
            images = sorted(
                file.name for file in person_dir.iterdir()
                if file.is_file() and file.suffix.lower() in ALLOWED_EXTENSIONS
            )
            people.append(
                {
                    "person_id": person_dir.name,
                    "name": name,
                    "image_count": len(images),
                    "images": images,
                }
            )
        return people

    async def add_person(self, name: str, files: list[UploadFile]) -> dict:
        clean_name = (name or "").strip()
        if not clean_name:
            raise ValueError("Name is required.")
        if not files:
            raise ValueError("At least one image is required.")

        person_dir = self._next_person_dir()
        person_dir.mkdir(parents=True, exist_ok=False)
        (person_dir / "name.txt").write_text(clean_name, encoding="utf-8")

        saved_files = []
        for index, file in enumerate(files, start=1):
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                suffix = ".jpg"
            destination = person_dir / f"photo{index}{suffix}"
            with destination.open("wb") as output:
                shutil.copyfileobj(file.file, output)
            saved_files.append(destination.name)
            await file.close()

        return {
            "person_id": person_dir.name,
            "name": clean_name,
            "saved_files": saved_files,
        }

    def train_embeddings(self) -> dict:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.settings.training_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        versioned_output = run_dir / "embeddings.pkl"
        root_output = self.settings.missing_persons_db_dir / "embeddings.pkl"

        database = FaceRecognizer.build_database(
            db_dir=str(self.settings.missing_persons_db_dir),
            output_path=str(versioned_output),
        )
        shutil.copy2(versioned_output, root_output)
        latest = resolve_latest_embeddings_path(root_output, self.settings.missing_persons_db_dir)

        return {
            "people_count": len(database),
            "embedding_count": sum(len(person["embeddings"]) for person in database),
            "versioned_output": str(versioned_output),
            "latest_output": str(latest),
        }

    def _next_person_dir(self) -> Path:
        existing = []
        for person_dir in self.settings.missing_persons_db_dir.glob("person_*"):
            if not person_dir.is_dir():
                continue
            try:
                existing.append(int(person_dir.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        next_id = max(existing, default=0) + 1
        return self.settings.missing_persons_db_dir / f"person_{next_id:03d}"
