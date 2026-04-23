"""
Missing Person Detection from screen capture.

This script reuses the same detection pipeline as detect_missing_person.py,
but takes frames from the desktop instead of a physical webcam.

Edit the config variables at the bottom of this file, then run:
    python detect_missing_person_screen.py
"""
import ctypes
import ctypes.wintypes
import sys
import time

import cv2
import numpy as np
from PIL import ImageGrab
from insightface.app import FaceAnalysis

import config
from detect_missing_person import (
    _bbox_overlap,
    draw_face_box,
    draw_info_overlay,
    draw_person_box,
    draw_predicted_track,
    draw_tracked_person,
    print_summary,
)
from utils import FaceDetector, FaceRecognizer, PersonDetector, PersonTracker, TrackState


def list_visible_windows():
    """Return a list of visible top-level windows on Windows."""
    if sys.platform != "win32":
        return []

    user32 = ctypes.windll.user32
    windows = []

    enum_windows_proc = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p
    )

    def callback(hwnd, lparam):
        if not user32.IsWindowVisible(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True

        title_buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title_buffer, length + 1)
        title = title_buffer.value.strip()
        if not title:
            return True

        rect = ctypes.wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True

        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width <= 0 or height <= 0:
            return True

        windows.append({
            "hwnd": hwnd,
            "title": title,
            "rect": (rect.left, rect.top, rect.right, rect.bottom),
        })
        return True

    user32.EnumWindows(enum_windows_proc(callback), 0)
    return windows


def get_window_rect(hwnd):
    """Get the current bounding rectangle for a window handle."""
    if sys.platform != "win32":
        raise RuntimeError("Window handle capture is only supported on Windows.")

    user32 = ctypes.windll.user32
    rect = ctypes.wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None

    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None

    return (rect.left, rect.top, rect.right, rect.bottom)


def find_window_region(window_title_keyword):
    """Find a visible window by title keyword and return its region."""
    if not window_title_keyword:
        return None, None

    if sys.platform != "win32":
        raise RuntimeError("Window selection by app title is only supported on Windows.")

    keyword = window_title_keyword.casefold()
    matches = [
        window for window in list_visible_windows()
        if keyword in window["title"].casefold()
    ]

    if not matches:
        raise RuntimeError(
            f"Cannot find any visible window matching keyword: {window_title_keyword!r}"
        )

    best_match = max(matches, key=lambda window: len(window["title"]))
    return best_match["hwnd"], best_match["rect"], best_match["title"]


def print_visible_windows():
    """Print visible window titles for easier app selection."""
    windows = list_visible_windows()
    if not windows:
        print("No visible windows found, or this platform is not Windows.")
        return

    print("\nVisible windows:")
    for index, window in enumerate(windows, start=1):
        print(f"  {index:>2}. {window['title']} | region={window['rect']}")
    print()


def choose_window_interactively():
    """Show visible windows and let the user choose one by number."""
    windows = list_visible_windows()
    if not windows:
        raise RuntimeError("No visible windows found, or this platform is not Windows.")

    print("\nVisible windows:")
    for index, window in enumerate(windows, start=1):
        print(f"  {index:>2}. {window['title']} | region={window['rect']}")

    while True:
        raw = input("\nChoose window number to capture (Enter to cancel): ").strip()
        if raw == "":
            return None, None, None
        if not raw.isdigit():
            print("Please enter a valid number.")
            continue

        selected_index = int(raw)
        if not (1 <= selected_index <= len(windows)):
            print(f"Please choose a number from 1 to {len(windows)}.")
            continue

        selected = windows[selected_index - 1]
        return selected["hwnd"], selected["rect"], selected["title"]


def parse_region(region_values):
    """Parse screen region as (left, top, right, bottom)."""
    if region_values is None:
        return None

    if len(region_values) != 4:
        raise ValueError("--region requires exactly 4 integers: left top right bottom")

    left, top, right, bottom = region_values
    if right <= left or bottom <= top:
        raise ValueError("Screen region must satisfy right > left and bottom > top")

    return (left, top, right, bottom)


def grab_screen_frame(region=None):
    """Capture a frame from the desktop and return it as a BGR numpy array."""
    screenshot = ImageGrab.grab(bbox=region, include_layered_windows=True)
    frame_rgb = np.array(screenshot)
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def is_mostly_black(frame, threshold=8, black_ratio=0.95):
    """Heuristic to detect black captures from unsupported windows."""
    if frame.size == 0:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray <= threshold)) >= black_ratio


def create_screen_writer(output_path, frame_size, fps):
    """Create VideoWriter for captured screen frames."""
    width, height = frame_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def main(output_path=None, db_path=None, threshold=None, frame_skip=None,
         no_display=False, region=None, capture_fps=10,
         window_title_keyword=None, print_windows=False,
         choose_window=False):
    """Main pipeline for missing person detection from screen capture."""
    embeddings_path = db_path or config.EMBEDDINGS_FILE
    thresh = threshold or config.RECOGNITION_THRESHOLD
    skip = frame_skip or config.FRAME_SKIP
    display = (not no_display) and config.DISPLAY_OUTPUT
    frame_interval = 1.0 / capture_fps if capture_fps > 0 else 0.0

    if print_windows:
        print_visible_windows()

    selected_window_hwnd = None
    selected_window_title = None
    if choose_window:
        selected_window_hwnd, region, selected_window_title = choose_window_interactively()
        if selected_window_hwnd is None:
            print("Window selection canceled. Exiting.")
            return []
    elif window_title_keyword:
        selected_window_hwnd, region, selected_window_title = find_window_region(window_title_keyword)

    print("=" * 60)
    print("  MISSING PERSON DETECTION SYSTEM - SCREEN CAPTURE")
    print("=" * 60)
    print("\n  Source:    Desktop screen [LIVE]")
    print(f"  Database:  {embeddings_path}")
    print(f"  Threshold: {thresh}")
    print(f"  Skip:      every {skip} frames")
    print(f"  Capture FPS target: {capture_fps}")
    if selected_window_title:
        print(f"  Window:    {selected_window_title}")
        print(f"  Region:    {region}")
    elif region:
        print(f"  Region:    {region}")
    else:
        print("  Region:    full screen")
    print()

    print("[1/4] Initializing Person Detector (YOLO)...")
    person_detector = PersonDetector(
        model_path=config.YOLO_MODEL_PATH,
        confidence_threshold=config.YOLO_CONFIDENCE_THRESHOLD
    )

    print("[2/4] Initializing InsightFace (ArcFace)...")
    face_app = FaceAnalysis(
        name=config.INSIGHTFACE_MODEL,
        providers=["CPUExecutionProvider"]
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    face_detector = FaceDetector(
        app=face_app,
        det_thresh=config.FACE_DETECTION_THRESHOLD
    )

    print("[3/4] Initializing Face Recognizer...")
    face_recognizer = FaceRecognizer(
        embeddings_path=embeddings_path,
        threshold=thresh
    )

    person_tracker = None
    if config.TRACKING_ENABLED:
        print("[3.5/4] Initializing Person Tracker (ByteTrack)...")
        person_tracker = PersonTracker(
            lost_track_buffer=config.BYTETRACK_LOST_BUFFER,
            minimum_matching_threshold=config.BYTETRACK_MATCH_THRESHOLD,
            frame_rate=max(int(capture_fps), 1),
            watch_threshold=config.TRACKING_WATCH_THRESHOLD,
            watch_frames=config.TRACKING_WATCH_FRAMES,
            trigger_score=config.TRACKING_TRIGGER_SCORE,
            trigger_high=config.TRACKING_TRIGGER_HIGH,
            max_watching_frames=config.TRACKING_MAX_WATCHING_FRAMES,
            reverify_interval=config.LOCKED_REVERIFY_INTERVAL,
            reverify_ok=config.LOCKED_REVERIFY_OK,
            reverify_drop=config.LOCKED_REVERIFY_DROP,
            reverify_max_fails=config.LOCKED_REVERIFY_MAX_FAILS,
            bbox_lost_timeout=config.LOCKED_BBOX_LOST_TIMEOUT,
        )

    print("[4/4] Starting screen capture...")
    first_frame = grab_screen_frame(region)
    height, width = first_frame.shape[:2]
    print(f"\n  Resolution: {width}x{height}")
    print("  Mode: REAL-TIME")
    print("\nProcessing... (press 'q' to stop)\n")

    writer = None
    if output_path:
        writer = create_screen_writer(output_path, (width, height), capture_fps)
        print(f"  Saving output to: {output_path}")

    frame_count = 0
    processed_count = 0
    detections_log = []
    start_time = time.time()
    last_capture_time = 0.0
    black_frame_warning_shown = False

    while True:
        now = time.time()
        if frame_interval > 0 and (now - last_capture_time) < frame_interval:
            time.sleep(min(0.005, frame_interval - (now - last_capture_time)))
            continue

        if selected_window_hwnd is not None:
            current_region = get_window_rect(selected_window_hwnd)
            if current_region is None:
                print("\nSelected window is no longer available. Stopping.")
                break
            region = current_region

        frame = grab_screen_frame(region)
        last_capture_time = time.time()
        frame_count += 1

        if selected_window_hwnd is not None and not black_frame_warning_shown and is_mostly_black(frame):
            print("\nWARNING: Captured window appears black.")
            print("This often happens with Chrome/Edge when hardware acceleration or DRM video is enabled.")
            print("Try disabling browser hardware acceleration, avoid protected video,")
            print("or capture the full screen / use a different app window.")
            black_frame_warning_shown = True

        if frame_count % skip != 0:
            if person_tracker:
                for pred in person_tracker.get_predicted_boxes():
                    draw_predicted_track(frame, pred)
            if writer:
                writer.write(frame)
            if display:
                cv2.imshow("Missing Person Detection - Screen", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nStopped by user.")
                    break
            continue

        processed_count += 1
        persons = person_detector.detect(frame)
        crops = person_detector.crop_persons(frame, persons)
        face_results = face_detector.detect_faces_in_crops(crops)

        match_count = 0
        if person_tracker:
            tracking = person_tracker.update(
                frame, persons, face_results, face_recognizer
            )

            for track in tracking["tracked"]:
                draw_tracked_person(frame, track)
                if track["state"] == TrackState.LOCKED:
                    match_count += 1

            tracked_person_bboxes = {tuple(t["bbox"]) for t in tracking["tracked"]}
            for face_info in face_results:
                pbbox = tuple(face_info["person_bbox"])
                already_drawn = any(
                    _bbox_overlap(pbbox, tb) > 0.3 for tb in tracked_person_bboxes
                )
                if not already_drawn:
                    draw_face_box(frame, face_info["face_bbox"], (200, 200, 200))

            for alert in tracking["alerts"]:
                timestamp = time.time() - start_time
                detections_log.append({
                    "frame": frame_count,
                    "timestamp": timestamp,
                    "person": alert["person_name"],
                    "person_id": alert["person_id"],
                    "similarity": alert["similarity"],
                    "bbox": alert["bbox"]
                })
                print(f"  !!! LOCKED: {alert['person_name']} "
                      f"at frame {frame_count} ({timestamp:.1f}s) "
                      f"- best_similarity={alert['similarity']:.3f} "
                      f"[Track T{alert['track_id']}]")
        else:
            for person in persons:
                draw_person_box(frame, person["bbox"], config.PERSON_COLOR)

            for face_info in face_results:
                embedding = face_info.get("embedding")
                if embedding is None:
                    draw_face_box(frame, face_info["face_bbox"], (200, 200, 200))
                    continue

                name, similarity, person_id = face_recognizer.match(embedding)
                if name is not None:
                    match_count += 1
                    timestamp = time.time() - start_time
                    draw_person_box(
                        frame,
                        face_info["person_bbox"],
                        config.ALERT_COLOR,
                        f"MISSING: {name} ({similarity * 100:.0f}%)"
                    )
                    draw_face_box(frame, face_info["face_bbox"], config.FACE_COLOR)
                    detections_log.append({
                        "frame": frame_count,
                        "timestamp": timestamp,
                        "person": name,
                        "person_id": person_id,
                        "similarity": similarity,
                        "bbox": face_info["person_bbox"]
                    })
                    print(f"  !!! DETECTED: {name} at frame {frame_count} "
                          f"({timestamp:.1f}s) - similarity={similarity:.3f}")
                else:
                    draw_face_box(frame, face_info["face_bbox"], (200, 200, 200))

        elapsed = time.time() - start_time
        current_fps = processed_count / elapsed if elapsed > 0 else 0
        active_tracks = len(person_tracker._active_tracks()) if person_tracker else 0
        draw_info_overlay(
            frame,
            frame_count,
            current_fps,
            len(persons),
            len(face_results),
            match_count,
            active_tracks if person_tracker else None
        )

        if writer:
            writer.write(frame)

        if display:
            cv2.imshow("Missing Person Detection - Screen", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped by user.")
                break

    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    elapsed_total = time.time() - start_time
    print(f"\nCompleted in {elapsed_total:.1f}s")
    print(f"Processed {processed_count} frames (skipped {frame_count - processed_count})")
    print_summary(detections_log)
    return detections_log


if __name__ == "__main__":
    OUTPUT_PATH = None
    DB_PATH = "missing_persons_db/embeddings.pkl"
    THRESHOLD = None
    FRAME_SKIP = None
    NO_DISPLAY = False
    CAPTURE_FPS = 10
    SCREEN_REGION = None
    WINDOW_TITLE_KEYWORD = None
    PRINT_VISIBLE_WINDOWS = False
    CHOOSE_WINDOW = True
    # Example region: SCREEN_REGION = (0, 0, 1280, 720)
    # Example window: WINDOW_TITLE_KEYWORD = "YouTube"
    # Set PRINT_VISIBLE_WINDOWS = True to print all current window titles first.
    # Set CHOOSE_WINDOW = True to choose from a numbered list of open windows.

    if CAPTURE_FPS <= 0:
        raise ValueError("CAPTURE_FPS must be > 0")

    main(
        output_path=OUTPUT_PATH,
        db_path=DB_PATH,
        threshold=THRESHOLD,
        frame_skip=FRAME_SKIP,
        no_display=NO_DISPLAY,
        region=parse_region(SCREEN_REGION),
        capture_fps=CAPTURE_FPS,
        window_title_keyword=WINDOW_TITLE_KEYWORD,
        print_windows=PRINT_VISIBLE_WINDOWS,
        choose_window=CHOOSE_WINDOW
    )
