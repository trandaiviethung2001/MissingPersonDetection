from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .detector_service import DetectorRuntime
from .settings import AppSettings
from .training_service import TrainingService


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
