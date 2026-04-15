FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY detect_missing_person.py .
COPY utils/ utils/
COPY yolov8n.pt .

RUN mkdir -p missing_persons_db models output

# Pre-download InsightFace model
RUN python -c "from insightface.app import FaceAnalysis; \
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
    app.prepare(ctx_id=0, det_size=(640, 640)); \
    print('InsightFace model downloaded')"

VOLUME ["/app/missing_persons_db", "/app/output"]

ENTRYPOINT ["python", "detect_missing_person.py"]
CMD ["--help"]
