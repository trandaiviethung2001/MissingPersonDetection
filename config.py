"""
Configuration for Missing Person Detection System.
Uses InsightFace (ArcFace) + YOLOv8.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MISSING_PERSONS_DB_DIR = os.path.join(BASE_DIR, "missing_persons_db")
EMBEDDINGS_FILE = os.path.join(MISSING_PERSONS_DB_DIR, "embeddings.pkl")
YOLO_MODEL_PATH = "yolov8n.pt"  # Auto-downloads if not found
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_PERSON_CLASS_ID = 0  # COCO class 0 = "person"
INSIGHTFACE_MODEL = "buffalo_l"
FACE_DETECTION_THRESHOLD = 0.5
RECOGNITION_THRESHOLD = 0.4
FRAME_SKIP = 5  # Process every N frames
DISPLAY_OUTPUT = True  # Show output video window
ALERT_COLOR = (0, 0, 255)  # Red - missing person box
PERSON_COLOR = (255, 200, 0)  # Blue - normal person box
FACE_COLOR = (0, 255, 0)  # Green - face box
FONT_SCALE = 0.7
FONT_THICKNESS = 2
