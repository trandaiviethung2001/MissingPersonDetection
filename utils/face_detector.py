"""
Face detection in cropped person regions using InsightFace.
"""
import numpy as np
import cv2


class FaceDetector:
    """Detect faces in cropped person images using InsightFace."""

    def __init__(self, app, det_thresh=0.5):
        """
        Args:
            app: Initialized InsightFace FaceAnalysis app.
            det_thresh: Confidence threshold for face detection.
        """
        self.app = app
        self.det_thresh = det_thresh

    def detect_faces(self, image):
        """
        Detect all faces in an image.

        Args:
            image: BGR image (numpy array).

        Returns:
            List of insightface Face objects.
        """
        faces = self.app.get(image)
        faces = [f for f in faces if f.det_score >= self.det_thresh]
        return faces

    def detect_faces_in_crops(self, person_crops):
        """
        Detect faces in person crops and map coordinates back to the original frame.

        Args:
            person_crops: List[dict] from PersonDetector.crop_persons().

        Returns:
            List[dict] with keys:
                - "face_bbox": (x1, y1, x2, y2) in original frame
                - "face_bbox_local": (x1, y1, x2, y2) in crop
                - "person_bbox": (x1, y1, x2, y2) person bounding box
                - "embedding": 512-d numpy array (ArcFace)
                - "det_score": face detection confidence
        """
        results = []

        for person in person_crops:
            crop = person["crop"]
            px1, py1, px2, py2 = person["bbox"]

            faces = self.detect_faces(crop)

            for face in faces:
                fx1, fy1, fx2, fy2 = face.bbox.astype(int)

                # Map crop coords to original frame
                gx1 = fx1 + px1
                gy1 = fy1 + py1
                gx2 = fx2 + px1
                gy2 = fy2 + py1

                embedding = face.embedding if hasattr(face, 'embedding') and face.embedding is not None else None

                results.append({
                    "face_bbox": (gx1, gy1, gx2, gy2),
                    "face_bbox_local": (fx1, fy1, fx2, fy2),
                    "person_bbox": person["bbox"],
                    "embedding": embedding,
                    "det_score": float(face.det_score)
                })

        return results
