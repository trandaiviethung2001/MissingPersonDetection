"""
Face detection in cropped person regions using InsightFace.
"""
import numpy as np


class FaceDetector:
    """Detect faces in cropped person images using InsightFace."""

    def __init__(self, app, det_thresh=0.5, max_faces=1):
        """
        Args:
            app: Initialized InsightFace FaceAnalysis app.
            det_thresh: Confidence threshold for face detection.
            max_faces: Maximum faces per crop (important for speed + stability)
        """
        self.app = app
        self.det_thresh = det_thresh
        self.max_faces = max_faces

    def detect_faces(self, image):
        """
        Detect all faces in an image crop.
        """
        faces = self.app.get(image, max_num=self.max_faces)

        # filter weak detections
        faces = [f for f in faces if f.det_score >= self.det_thresh]

        return faces

    def detect_faces_in_crops(self, person_crops):
        """
        Detect faces in person crops and map coordinates back to original frame.
        """
        results = []

        for person in person_crops:
            crop = person["crop"]
            px1, py1, px2, py2 = person["bbox"]

            faces = self.detect_faces(crop)

            # skip if no face detected
            if len(faces) == 0:
                continue

            for face in faces:
                fx1, fy1, fx2, fy2 = face.bbox.astype(int)

                # map to original frame
                gx1 = fx1 + px1
                gy1 = fy1 + py1
                gx2 = fx2 + px1
                gy2 = fy2 + py1

                embedding = getattr(face, "embedding", None)

                results.append({
                    "face_bbox": (gx1, gy1, gx2, gy2),
                    "face_bbox_local": (fx1, fy1, fx2, fy2),
                    "person_bbox": person["bbox"],
                    "embedding": embedding,
                    "det_score": float(face.det_score)
                })

        return results