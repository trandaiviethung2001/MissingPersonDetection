import torch
from ultralytics import YOLO


class PersonDetector:
    """
    TensorRT-safe YOLO wrapper (drop-in replacement).
    Works with .engine, .onnx, and .pt without crashing.
    """

    def __init__(
        self,
        model_path="yolov8n.engine",
        confidence_threshold=0.5,
        image_size=640,
    ):
        self.model = YOLO(model_path)

        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.person_class_id = 0

        # DO NOT call fuse() for engine/onnx
        # DO NOT call .to(device)

        print(f"[PersonDetector-TensorRT] Loaded model: {model_path}")

    def detect(self, frame):
        """
        Fast inference (TensorRT path)
        """

        results = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            classes=[self.person_class_id],  # BIG speed win
            verbose=False,
            device=0,  # safe for TensorRT runtime
            max_det=20,  # reduces postprocess cost
        )[0]

        detections = []

        if results.boxes is None:
            return detections

        for i in range(len(results.boxes)):
            conf = float(results.boxes.conf[i])
            x1, y1, x2, y2 = results.boxes.xyxy[i].cpu().numpy().astype(int)

            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": conf
            })

        return detections

    def crop_persons(self, frame, detections):
        h, w = frame.shape[:2]
        crops = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                crops.append({
                    "crop": crop,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": det["confidence"]
                })

        return crops