from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .detector_service import DetectorRuntime
from .settings import AppSettings
from .training_service import TrainingService
from .upload_service import UploadService


ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    async def broadcast_json(self, payload: dict[str, Any]) -> None:
        stale = []
        for websocket in list(self._connections):
            try:
                await websocket.send_json(payload)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(websocket)

    def broadcast_from_thread(self, payload: dict[str, Any]) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.broadcast_json(payload), self._loop)


settings = AppSettings.load()
manager = ConnectionManager()
runtime = DetectorRuntime(settings=settings, broadcaster=manager.broadcast_from_thread)
training_service = TrainingService(settings=settings)
upload_service = UploadService(settings=settings)

app = FastAPI(title="Drone Lost Person Finder Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=settings.frontend_dir / "static"), name="static")


@app.on_event("startup")
async def on_startup() -> None:
    manager.attach_loop(asyncio.get_running_loop())
    runtime.ensure_started()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    runtime.shutdown()


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "service": "drone-lost-person-finder", "runtime": runtime.get_runtime_info()}


@app.get("/api/config")
async def config() -> dict[str, Any]:
    return runtime.get_runtime_info()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(settings.frontend_dir / "index.html")


@app.get("/training")
async def training_page() -> FileResponse:
    return FileResponse(settings.frontend_dir / "training.html")


@app.get("/api/training/people")
async def training_people() -> dict[str, Any]:
    return {"people": training_service.list_people()}


@app.post("/api/training/person")
async def create_person(
    name: str = Form(...),
    images: list[UploadFile] = File(...),
) -> dict[str, Any]:
    try:
        result = await training_service.add_person(name=name, files=images)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "person": result}


@app.post("/api/training/run")
async def run_training() -> dict[str, Any]:
    try:
        result = training_service.train_embeddings()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    runtime.reload_pipeline()
    return {"ok": True, "training": result}


@app.get("/upload")
async def upload_page() -> FileResponse:
    return FileResponse(settings.frontend_dir / "upload.html")


@app.post("/api/upload")
async def api_upload(
    video: UploadFile = File(...),
    threshold: float | None = Form(default=None),
    frame_skip: int | None = Form(default=None),
) -> dict[str, Any]:
    """Accept the upload, kick off detection in the background, return a job id.

    The frontend then polls ``GET /api/upload/{job_id}`` to drive its
    progress bar and pick up the final result.
    """
    filename = video.filename or "uploaded.mp4"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext or '(none)'}."
            f" Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTS))}",
        )

    upload_id = uuid.uuid4().hex[:10]
    input_path = settings.upload_dir / f"{upload_id}_input{ext}"
    output_path = settings.upload_dir / f"{upload_id}_annotated.mp4"

    # Stream the upload to disk (handles large files without loading into RAM)
    with open(input_path, "wb") as f:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    job_id = upload_service.create_job(
        input_path=input_path,
        output_path=output_path,
        original_filename=filename,
        threshold=threshold,
        frame_skip=frame_skip,
    )
    # Detection is CPU-heavy; run it as a background asyncio task so the
    # POST returns immediately and the frontend can start polling status.
    asyncio.create_task(upload_service.run_job(job_id))

    return {
        "ok": True,
        "job_id": job_id,
        "upload_id": upload_id,
        "original_filename": filename,
        "status_url": f"/api/upload/{job_id}",
    }


@app.get("/api/upload/{job_id}")
async def api_upload_status(job_id: str) -> dict[str, Any]:
    job = upload_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/uploads/{filename}")
async def api_upload_file(filename: str) -> FileResponse:
    # Guard against path traversal — must be a bare filename
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = settings.upload_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.get("/api/training/person/{person_id}/image/{filename}")
async def api_training_person_image(person_id: str, filename: str) -> FileResponse:
    for segment in (person_id, filename):
        if "/" in segment or "\\" in segment or ".." in segment:
            raise HTTPException(status_code=400, detail="Invalid path")
    if not person_id.startswith("person_"):
        raise HTTPException(status_code=400, detail="Invalid person id")
    path = settings.missing_persons_db_dir / person_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@app.get("/api/lock_snapshots/{filename}")
async def api_lock_snapshot(filename: str) -> FileResponse:
    # Same path-traversal guard as uploads
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = settings.lock_snapshot_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="image/jpeg")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    await websocket.send_json({"type": "config", "data": runtime.get_runtime_info()})
    await websocket.send_json(runtime.current_status_message())
    await websocket.send_json(runtime.current_detection_message())
    await websocket.send_json(runtime.current_telemetry_message())

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "ack", "data": {"command": "invalid_json", "success": False}})
                continue

            command = (message.get("command") or "").strip().lower()
            if command == "start":
                runtime.start()
            elif command == "pause":
                runtime.pause()
            elif command == "snapshot":
                runtime.snapshot()
            elif command == "record":
                runtime.set_recording(bool(message.get("enabled")))
            elif command == "rtl":
                runtime.return_to_home()
            elif command == "emergency_stop":
                runtime.emergency_stop()
            elif command == "lock_target":
                runtime.force_lock(message.get("targetId"))
            else:
                await websocket.send_json({"type": "ack", "data": {"command": command or "unknown", "success": False}})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
