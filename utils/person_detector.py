"""
Person detection using YOLOv8.
Detects all persons in a video frame to narrow down face search regions.
"""
import numpy as np
from ultralytics import YOLO


class PersonDetector:
    """Detect persons in images/frames using YOLOv8."""

    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5, image_size=416):
        """
        Args:
            model_path: Path to YOLO model (.pt). Auto-downloads if missing.
            confidence_threshold: Minimum confidence threshold.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.person_class_id = 0  # COCO class 0 = "person"

    def detect(self, frame):
        """
        Detect all persons in a frame.

        Args:
            frame: BGR image (numpy array) from OpenCV.

        Returns:
            List[dict] with keys:
                - "bbox": (x1, y1, x2, y2) bounding box coordinates
                - "confidence": float confidence score
        """
        results = self.model(frame, imgsz=self.image_size, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id == self.person_class_id and conf >= self.confidence_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": conf
                })

        return detections

    def crop_persons(self, frame, detections):
        """
        Crop each detected person from the original frame.

        Args:
            frame: Original BGR image.
            detections: Results from detect().

        Returns:
            List[dict] with keys:
                - "crop": cropped numpy array
                - "bbox": (x1, y1, x2, y2) in original frame
                - "confidence": float
        """
        h, w = frame.shape[:2]
        crops = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append({
                    "crop": crop,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": det["confidence"]
                })

        return crops
