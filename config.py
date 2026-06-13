"""
Configuration for Missing Person Detection System.
Uses InsightFace (ArcFace) + YOLOv8.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MISSING_PERSONS_DB_DIR = os.path.join(BASE_DIR, "missing_persons_db")
EMBEDDINGS_FILE = os.path.join(MISSING_PERSONS_DB_DIR, "embeddings.pkl")
YOLO_MODEL_PATH = "yolov8n.engine"  # Auto-downloads if not found
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "416"))
YOLO_PERSON_CLASS_ID = 0  # COCO class 0 = "person"
INSIGHTFACE_MODEL = "buffalo_l"
INSIGHTFACE_DET_SIZE = int(os.getenv("INSIGHTFACE_DET_SIZE", "160"))
FACE_DETECTION_THRESHOLD = 0.5
RECOGNITION_THRESHOLD = 0.4
FRAME_SKIP = 3  # Process every N frames
DISPLAY_OUTPUT = True  # Show output video window
ALERT_COLOR = (0, 0, 255)  # Red - missing person box
PERSON_COLOR = (255, 200, 0)  # Blue - normal person box
FACE_COLOR = (0, 255, 0)  # Green - face box
FONT_SCALE = 0.7
FONT_THICKNESS = 2
FACE_SKIP = 2   # run face every 2 frames

# --- Webcam capture ---
# Match the proven ffplay settings:
# ffplay -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 /dev/video0
CAMERA_WIDTH = int(os.getenv("LOST_PERSON_CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("LOST_PERSON_CAMERA_HEIGHT", "720"))
CAMERA_FPS = float(os.getenv("LOST_PERSON_CAMERA_FPS", "30"))
CAMERA_FOURCC = os.getenv("LOST_PERSON_CAMERA_FOURCC", "MJPG")
CAMERA_BUFFER_SIZE = int(os.getenv("LOST_PERSON_CAMERA_BUFFER_SIZE", "1"))

# --- Tracking (ByteTrack + state machine) ---
TRACKING_ENABLED = True
TRACKING_WATCH_THRESHOLD = 0.35       # Similarity to enter WATCHING
TRACKING_WATCH_FRAMES = 2             # Consecutive frames above threshold → WATCHING
TRACKING_TRIGGER_SCORE = 0.40         # Weighted score to trigger LOCKED
TRACKING_TRIGGER_HIGH = 0.50          # At least 1 frame must exceed this
TRACKING_MAX_WATCHING_FRAMES = 10     # Max frames in WATCHING before reset
BYTETRACK_LOST_BUFFER = 30            # Frames to keep lost tracks
BYTETRACK_MATCH_THRESHOLD = 0.8       # IoU threshold for ByteTrack association

# --- LOCKED state: soft lock with periodic re-verification ---
LOCKED_REVERIFY_INTERVAL = 60         # Re-verify face every N processed frames
LOCKED_REVERIFY_OK = 0.35             # similarity >= this → re-verify passed
LOCKED_REVERIFY_DROP = 0.25           # similarity < this → re-verify failed
LOCKED_REVERIFY_MAX_FAILS = 3         # Consecutive re-verify fails → drop track
LOCKED_BBOX_LOST_TIMEOUT = 2.0        # Seconds without bbox → drop track
LOCKED_PREDICT_DRAW_TIMEOUT = 0.5     # Seconds to keep drawing phantom predictions
LOCKED_OUT_OF_FRAME_MARGIN = 0.6      # Drop track when >this fraction of bbox left the frame

# --- Colors ---
WATCHING_COLOR = (0, 165, 255)        # Orange – WATCHING state box
LOCKED_COLOR = (0, 0, 255)            # Red – LOCKED state box

# --- Tactical map overlay ---
TACTICAL_MAP_ENABLED = True           # Draw top-down map inset for LOCKED persons
