"""
Missing Person Detection System - Main Pipeline.

Pipeline:
1. Read input video (crowd footage) or live webcam/stream
2. YOLO detects all persons in each frame
3. InsightFace detects faces in each person region
4. ArcFace embedding matched against missing person database
5. Alert when a missing person is detected

Usage:
    python detect_missing_person.py --video path/to/video.mp4
    python detect_missing_person.py --video path/to/video.mp4 --output output/result.mp4
    python detect_missing_person.py --webcam
    python detect_missing_person.py --webcam --camera-id 1
"""
import argparse
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis

import config
from utils import PersonDetector, FaceDetector, FaceRecognizer, PersonTracker, TrackState


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
    """Draw alert when a missing person is detected (legacy, non-tracking mode)."""
    confidence = similarity * 100
    label = f"MISSING: {name} ({confidence:.0f}%)"

    draw_person_box(frame, face_info["person_bbox"], config.ALERT_COLOR, label)
    draw_face_box(frame, face_info["face_bbox"], config.FACE_COLOR)

    x1, y1, x2, y2 = face_info["face_bbox"]
    cv2.putText(frame, "! ALERT !", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def draw_tracked_person(frame, track_info):
    """Draw bounding box based on tracking state."""
    bbox = track_info["bbox"]
    state = track_info["state"]
    name = track_info.get("person_name")
    tid = track_info.get("track_id", "")
    best_sim = track_info.get("best_similarity", 0)

    if state == TrackState.LOCKED:
        confidence = best_sim * 100
        label = f"LOCKED: {name} ({confidence:.0f}%) [T{tid}]"
        draw_person_box(frame, bbox, config.LOCKED_COLOR, label)

        # Alert flash
        x1, y1, x2, y2 = bbox
        cv2.putText(frame, "! ALERT !", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        face_info = track_info.get("face_info")
        if face_info:
            draw_face_box(frame, face_info["face_bbox"], config.FACE_COLOR)

    elif state == TrackState.WATCHING:
        sim = track_info.get("similarity", 0)
        label = f"WATCHING: {name or '?'} [{sim:.0%}] [T{tid}]"
        draw_person_box(frame, bbox, config.WATCHING_COLOR, label)

    else:
        draw_person_box(frame, bbox, config.PERSON_COLOR)


def draw_predicted_track(frame, pred):
    """Draw predicted bbox for a TRACKING person on skipped frames."""
    bbox = pred["bbox"]
    name = pred.get("person_name", "?")
    tid = pred.get("track_id", "")
    best_sim = pred.get("best_similarity", 0)
    confidence = best_sim * 100
    label = f"LOCKED: {name} ({confidence:.0f}%) [T{tid}]"

    x1, y1, x2, y2 = bbox
    # Dashed-style: draw thinner box
    cv2.rectangle(frame, (x1, y1), (x2, y2), config.LOCKED_COLOR, 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                   config.FONT_SCALE, config.FONT_THICKNESS)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1),
                  config.LOCKED_COLOR, -1)
    cv2.putText(frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                (255, 255, 255), config.FONT_THICKNESS)


def draw_info_overlay(frame, frame_count, fps, total_persons, total_faces,
                      matches, active_tracks=None):
    """Draw info overlay on frame corner."""
    info_lines = [
        f"Frame: {frame_count}",
        f"FPS: {fps:.1f}",
        f"Persons: {total_persons}",
        f"Faces: {total_faces}",
        f"Matches: {matches}",
    ]
    if active_tracks is not None:
        info_lines.append(f"Tracking: {active_tracks}")
    y = 30
    for line in info_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)
        y += 25


def _bbox_overlap(box_a, box_b):
    """IoU between two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


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


def parse_video_source(source):
    """Parse video source - file path, device index, or stream URL."""
    if source is None:
        return None
    try:
        return int(source)
    except (ValueError, TypeError):
        return source


def main(video_path, output_path=None, threshold=None,
         frame_skip=None, no_display=False):
    """Main pipeline for missing person detection in video or webcam."""
    embeddings_path = config.EMBEDDINGS_FILE
    thresh = threshold or config.RECOGNITION_THRESHOLD
    skip = frame_skip or config.FRAME_SKIP
    display = (not no_display) and config.DISPLAY_OUTPUT

    is_live = isinstance(video_path, int) or (
        isinstance(video_path, str) and video_path.startswith(("rtsp://", "http://", "https://"))
    )

    print("=" * 60)
    print("  MISSING PERSON DETECTION SYSTEM")
    print("=" * 60)
    if is_live:
        source_label = f"Webcam (device {video_path})" if isinstance(video_path, int) else video_path
        print(f"\n  Source:    {source_label} [LIVE]")
    else:
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

    person_tracker = None
    if config.TRACKING_ENABLED:
        print("[3.5/4] Initializing Person Tracker (ByteTrack)...")
        person_tracker = PersonTracker(
            lost_track_buffer=config.BYTETRACK_LOST_BUFFER,
            minimum_matching_threshold=config.BYTETRACK_MATCH_THRESHOLD,
            frame_rate=30,
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

    # === 2. Open video / webcam ===
    print("[4/4] Opening video source...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video source: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"\n  Resolution: {width}x{height}")
    print(f"  FPS: {video_fps}")

    if is_live:
        print(f"  Mode: REAL-TIME")
        total_frames = 0
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
            if is_live:
                continue
            break

        frame_count += 1

        # Skip frames – but still draw active tracking predictions
        if frame_count % skip != 0:
            if person_tracker:
                for pred in person_tracker.get_predicted_boxes():
                    draw_predicted_track(frame, pred)
            if writer:
                writer.write(frame)
            if display:
                cv2.imshow("Missing Person Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nStopped by user.")
                    break
            continue

        processed_count += 1

        # Step 4: Detect persons with YOLO
        persons = person_detector.detect(frame)

        # Step 5: Detect faces in each person region
        crops = person_detector.crop_persons(frame, persons)
        face_results = face_detector.detect_faces_in_crops(crops)

        # Step 6: Tracking mode – ByteTrack + state machine
        match_count = 0
        if person_tracker:
            tracking = person_tracker.update(
                frame, persons, face_results, face_recognizer
            )

            # Draw tracked persons by state
            for track in tracking["tracked"]:
                draw_tracked_person(frame, track)
                if track["state"] == TrackState.LOCKED:
                    match_count += 1

            # Draw untracked faces (those not associated with any track)
            tracked_person_bboxes = {tuple(t["bbox"]) for t in tracking["tracked"]}
            for face_info in face_results:
                pbbox = tuple(face_info["person_bbox"])
                already_drawn = any(
                    _bbox_overlap(pbbox, tb) > 0.3 for tb in tracked_person_bboxes
                )
                if not already_drawn:
                    draw_face_box(frame, face_info["face_bbox"], (200, 200, 200))

            # Log new alerts
            for alert in tracking["alerts"]:
                timestamp = frame_count / video_fps
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

        # Step 6 (legacy): Direct matching without tracking
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
                    draw_alert(frame, face_info, name, similarity)
                    timestamp = frame_count / video_fps
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

        # Step 7: Draw info overlay
        elapsed = time.time() - start_time
        current_fps = processed_count / elapsed if elapsed > 0 else 0
        active_tracks = len(person_tracker._active_tracks()) if person_tracker else 0
        draw_info_overlay(frame, frame_count, current_fps,
                          len(persons), len(face_results), match_count,
                          active_tracks if person_tracker else None)

        # Step 8: Display / write video
        if display:
            cv2.imshow("Missing Person Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped by user.")
                break

        if writer:
            writer.write(frame)

        # Progress
        if not is_live and processed_count % 20 == 0:
            progress = frame_count / total_frames * 100 if total_frames > 0 else 0
            print(f"  Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} | "
                  f"FPS: {current_fps:.1f} | Persons: {len(persons)} | "
                  f"Faces: {len(face_results)} | Active tracks: {active_tracks}")

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
        description="Missing person detection in crowd video or live webcam"
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--video",
        help="Path to input video file or stream URL (rtsp://, http://)"
    )
    source_group.add_argument(
        "--webcam", action="store_true",
        help="Use webcam for real-time detection"
    )
    parser.add_argument(
        "--camera-id", type=int, default=0,
        help="Camera device index (default: 0, used with --webcam)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save output video (optional)"
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

    if args.webcam:
        video_source = args.camera_id
    else:
        video_source = parse_video_source(args.video)

    main(
        video_path=video_source,
        output_path=args.output,
        threshold=args.threshold,
        frame_skip=args.skip,
        no_display=args.no_display
    )
