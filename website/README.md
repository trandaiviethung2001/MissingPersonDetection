# Website

Dashboard web app + FastAPI backend for real-time missing person detection.

## Structure

```text
website/
  backend/
    app/
      main.py
      detector_service.py
      settings.py
      training_service.py
  frontend/
    index.html
    training.html
    static/
      app.js
      training.js
      style.css
  data/
    recordings/
    snapshots/
    training_runs/
```

## Run

```powershell
cd c:\Users\aa\Desktop\my_project\MissingPersonDetection
python -m pip install -r requirements.txt
python -m uvicorn website.backend.app.main:app --host 0.0.0.0 --port 8765 --reload
```

Open `http://localhost:8765`

Open `http://localhost:8765/training` for quick face training.

## Environment variables

- `LOST_PERSON_CAMERA_SOURCE=0`
- `LOST_PERSON_DB_PATH=..\..\missing_persons_db\embeddings.pkl`
- `LOST_PERSON_FRAME_SKIP=5`
- `LOST_PERSON_THRESHOLD=0.4`
- `LOST_PERSON_STREAM_FPS=10`
