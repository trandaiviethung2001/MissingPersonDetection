"""
Missing Person Detection System - Main Pipeline.

Pipeline:
1. Read input video (crowd footage)
2. YOLO detects all persons in each frame
3. InsightFace detects faces in each person region
4. ArcFace embedding matched against missing person database
5. Alert when a missing person is detected

Usage:
    python detect_missing_person.py --video path/to/video.mp4
    python detect_missing_person.py --video path/to/video.mp4 --output output/result.mp4
"""
import argparse
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis

import config
from utils import PersonDetector, FaceDetector, FaceRecognizer


def setup_video_writer(cap, output_path):
    """Create VideoWriter matching source FPS and resolution."""
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def draw_person_box(frame, bbox, color, label=None):
    """Draw bounding box for a person."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       config.FONT_SCALE, config.FONT_THICKNESS)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                    (255, 255, 255), config.FONT_THICKNESS)


def draw_face_box(frame, face_bbox, color):
    """Draw bounding box for a face."""
    x1, y1, x2, y2 = face_bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def draw_alert(frame, face_info, name, similarity):
    """Draw alert when a missing person is detected."""
    confidence = similarity * 100
    label = f"MISSING: {name} ({confidence:.0f}%)"

    draw_person_box(frame, face_info["person_bbox"], config.ALERT_COLOR, label)
    draw_face_box(frame, face_info["face_bbox"], config.FACE_COLOR)

    x1, y1, x2, y2 = face_info["face_bbox"]
    cv2.putText(frame, "! ALERT !", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def draw_info_overlay(frame, frame_count, fps, total_persons, total_faces, matches):
    """Draw info overlay on frame corner."""
    info_lines = [
        f"Frame: {frame_count}",
        f"FPS: {fps:.1f}",
        f"Persons: {total_persons}",
        f"Faces: {total_faces}",
        f"Matches: {matches}",
    ]
    y = 30
    for line in info_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)
        y += 25


def print_summary(detections_log):
    """Print detection summary."""
    if not detections_log:
        print("\n" + "=" * 60)
        print("RESULT: No missing persons detected in video.")
        print("=" * 60)
        return

    print("\n" + "=" * 60)
    print(f"RESULT: {len(detections_log)} missing person appearances detected")
    print("=" * 60)

    # Group by name
    by_name = {}
    for det in detections_log:
        name = det["person"]
        if name not in by_name:
            by_name[name] = []
        by_name[name].append(det)

    for name, dets in by_name.items():
        print(f"\n  [{name}]")
        print(f"  Detections: {len(dets)}")
        timestamps = [d["timestamp"] for d in dets]
        print(f"  Time range: {min(timestamps):.1f}s - {max(timestamps):.1f}s")
        avg_sim = np.mean([d["similarity"] for d in dets])
        print(f"  Avg similarity: {avg_sim:.1%}")

        for det in dets[:5]:
            t = det["timestamp"]
            s = det["similarity"]
            print(f"    - Frame {det['frame']:>5d} | {t:>6.1f}s | similarity={s:.3f}")
        if len(dets) > 5:
            print(f"    ... and {len(dets) - 5} more")

    print("\n" + "=" * 60)


def main(video_path, output_path=None, db_path=None, threshold=None,
         frame_skip=None, no_display=False):
    """Main pipeline for missing person detection in video."""
    embeddings_path = db_path or config.EMBEDDINGS_FILE
    thresh = threshold or config.RECOGNITION_THRESHOLD
    skip = frame_skip or config.FRAME_SKIP
    display = (not no_display) and config.DISPLAY_OUTPUT

    print("=" * 60)
    print("  MISSING PERSON DETECTION SYSTEM")
    print("=" * 60)
    print(f"\n  Video:     {video_path}")
    print(f"  Database:  {embeddings_path}")
    print(f"  Threshold: {thresh}")
    print(f"  Skip:      every {skip} frames")
    print()

    # === 1. Initialize modules ===
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

    # === 2. Open video ===
    print("[4/4] Opening video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n  Resolution: {width}x{height}")
    print(f"  FPS: {video_fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames / video_fps:.1f}s")
    print(f"  Frames to process: ~{total_frames // skip}")
    print("\nProcessing... (press 'q' to stop)\n")

    writer = None
    if output_path:
        writer = setup_video_writer(cap, output_path)
        print(f"  Saving output to: {output_path}")

    # === 3. Frame processing loop ===
    frame_count = 0
    processed_count = 0
    detections_log = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames
        if frame_count % skip != 0:
            if writer:
                writer.write(frame)
            continue

        processed_count += 1

        # Step 4: Detect persons with YOLO
        persons = person_detector.detect(frame)

        for person in persons:
            draw_person_box(frame, person["bbox"], config.PERSON_COLOR)

        # Step 5: Detect faces in each person region
        crops = person_detector.crop_persons(frame, persons)
        face_results = face_detector.detect_faces_in_crops(crops)

        # Step 6: Match faces against database
        match_count = 0
        for face_info in face_results:
            embedding = face_info.get("embedding")
            if embedding is None:
                draw_face_box(frame, face_info["face_bbox"], (200, 200, 200))
                continue

            name, similarity, person_id = face_recognizer.match(embedding)

            if name is not None:
                match_count += 1
                draw_alert(frame, face_info, name, similarity)
                detections_log.append({
                    "frame": frame_count,
                    "timestamp": frame_count / video_fps,
                    "person": name,
                    "person_id": person_id,
                    "similarity": similarity,
                    "bbox": face_info["person_bbox"]
                })
                print(f"  !!! DETECTED: {name} at frame {frame_count} "
                      f"({frame_count / video_fps:.1f}s) - similarity={similarity:.3f}")
            else:
                draw_face_box(frame, face_info["face_bbox"], (200, 200, 200))

        # Step 7: Draw info overlay
        elapsed = time.time() - start_time
        current_fps = processed_count / elapsed if elapsed > 0 else 0
        draw_info_overlay(frame, frame_count, current_fps,
                          len(persons), len(face_results), match_count)

        # Step 8: Display / write video
        if display:
            cv2.imshow("Missing Person Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped by user.")
                break

        if writer:
            writer.write(frame)

        # Progress
        if processed_count % 20 == 0:
            progress = frame_count / total_frames * 100 if total_frames > 0 else 0
            print(f"  Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} | "
                  f"FPS: {current_fps:.1f} | Persons: {len(persons)} | "
                  f"Faces: {len(face_results)}")

    # === 9. Cleanup and report ===
    cap.release()
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
    parser = argparse.ArgumentParser(
        description="Missing person detection in crowd video"
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to input video"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save output video (optional)"
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to embeddings.pkl (overrides config)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Cosine similarity threshold (overrides config)"
    )
    parser.add_argument(
        "--skip", type=int, default=None,
        help="Process every N frames (overrides config)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable video display window"
    )

    args = parser.parse_args()

    main(
        video_path=args.video,
        output_path=args.output,
        db_path=args.db,
        threshold=args.threshold,
        frame_skip=args.skip,
        no_display=args.no_display
    )
