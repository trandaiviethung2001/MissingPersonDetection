"""
Person tracking with ByteTrack + confidence accumulation state machine.

State machine:
    IDLE → (similarity > 0.35 for 2 consecutive frames) → WATCHING
    WATCHING → (weighted score > 0.40 + at least 1 frame > 0.50) → LOCKED
    WATCHING → (score drops / timeout) → IDLE
    LOCKED:
        - Tracker is the primary driver, label is frozen
        - Every 30 frames: re-verify face if visible
            - similarity > 0.35 → OK, continue
            - similarity < 0.25 for 3 consecutive re-verifies → DROP
        - Bbox lost (out of frame / long occlusion) for 2 seconds → DROP

Post-LOCKED tracking:
    - ByteTrack maintains identity via bbox IoU + Kalman filter
    - Body appearance (HSV histogram) for re-identification
    - Movement velocity prediction for occlusion handling
    - Person stays tracked even when face turns away or is temporarily blocked
"""
import time
from enum import Enum

import cv2
import numpy as np
import supervision as sv


class TrackState(Enum):
    IDLE = "IDLE"
    WATCHING = "WATCHING"
    LOCKED = "LOCKED"


class TrackedPerson:
    """State machine + appearance model for a single tracked person."""

    def __init__(self, track_id,
                 watch_threshold=0.35, watch_frames=2,
                 trigger_score=0.40, trigger_high=0.50,
                 max_watching_frames=10,
                 reverify_interval=30,
                 reverify_ok=0.35, reverify_drop=0.25,
                 reverify_max_fails=3):
        self.track_id = track_id
        self.state = TrackState.IDLE

        # --- IDLE → WATCHING config ---
        self.watch_threshold = watch_threshold
        self.watch_frames = watch_frames

        # --- WATCHING → LOCKED config ---
        self.trigger_score = trigger_score
        self.trigger_high = trigger_high
        self.max_watching_frames = max_watching_frames

        # --- LOCKED re-verify config ---
        self.reverify_interval = reverify_interval
        self.reverify_ok = reverify_ok
        self.reverify_drop = reverify_drop
        self.reverify_max_fails = reverify_max_fails

        # --- State counters ---
        self.consecutive_above = 0
        self.watching_scores = []
        self.has_high_frame = False

        # LOCKED re-verify counters
        self.frames_since_verify = 0
        self.consecutive_verify_fails = 0

        # --- Identity (frozen once LOCKED) ---
        self.person_name = None
        self.person_id = None
        self.best_similarity = 0.0

        # --- Appearance model (body) ---
        self.color_histogram = None
        self.last_bbox = None
        self.velocity = (0.0, 0.0)
        self.bbox_history = []

        # --- Bbox-lost tracking (wall-clock) ---
        self.last_bbox_seen_time = None   # set by PersonTracker
        self.frames_since_face = 0
        self.total_frames = 0

    # ------------------------------------------------------------------
    # Appearance model
    # ------------------------------------------------------------------

    def update_appearance(self, frame, bbox):
        """Update body appearance model with current crop."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        if self.color_histogram is None:
            self.color_histogram = hist.copy()
        else:
            self.color_histogram = 0.7 * self.color_histogram + 0.3 * hist

        if self.last_bbox is not None:
            prev_cx = (self.last_bbox[0] + self.last_bbox[2]) / 2
            prev_cy = (self.last_bbox[1] + self.last_bbox[3]) / 2
            curr_cx = (x1 + x2) / 2
            curr_cy = (y1 + y2) / 2
            self.velocity = (curr_cx - prev_cx, curr_cy - prev_cy)

        self.last_bbox = (x1, y1, x2, y2)
        self.bbox_history.append((x1, y1, x2, y2))
        if len(self.bbox_history) > 30:
            self.bbox_history.pop(0)

    def appearance_similarity(self, frame, bbox):
        """Compare body appearance of a candidate bbox with stored model."""
        if self.color_histogram is None:
            return 0.0
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return cv2.compareHist(self.color_histogram, hist, cv2.HISTCMP_CORREL)

    def predict_next_bbox(self):
        """Predict next bounding box using constant-velocity model."""
        if self.last_bbox is None:
            return None
        x1, y1, x2, y2 = self.last_bbox
        dx, dy = self.velocity
        return (int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy))

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def update_state(self, similarity, face_detected):
        """
        Feed a new observation into the state machine.

        Args:
            similarity: cosine similarity to the best-matching DB person.
                        0.0 when no face was detected this frame.
            face_detected: True if a face embedding was available.

        Returns:
            str or None – state transition label if a transition happened:
                "WATCHING"   – entered WATCHING
                "LOCKED"     – entered LOCKED (alert!)
                "IDLE_RESET" – dropped back to IDLE from WATCHING
                "IDLE_DROP"  – dropped back to IDLE from LOCKED (re-verify failed)
        """
        self.total_frames += 1

        if self.state == TrackState.IDLE:
            if not face_detected:
                return None
            return self._handle_idle(similarity)

        if self.state == TrackState.WATCHING:
            if not face_detected:
                return None
            return self._handle_watching(similarity)

        # LOCKED – tracker is primary, only re-verify periodically
        return self._handle_locked(similarity, face_detected)

    def _handle_idle(self, similarity):
        if similarity > self.watch_threshold:
            self.consecutive_above += 1
            if self.consecutive_above >= self.watch_frames:
                self.state = TrackState.WATCHING
                self.watching_scores = [similarity]
                self.has_high_frame = similarity > self.trigger_high
                return "WATCHING"
        else:
            self.consecutive_above = 0
        return None

    def _handle_watching(self, similarity):
        self.watching_scores.append(similarity)
        if similarity > self.trigger_high:
            self.has_high_frame = True

        if len(self.watching_scores) >= 2:
            weights = np.linspace(0.5, 1.0, len(self.watching_scores))
            weighted = float(np.average(self.watching_scores, weights=weights))
            if weighted > self.trigger_score and self.has_high_frame:
                self.state = TrackState.LOCKED
                self.frames_since_verify = 0
                self.consecutive_verify_fails = 0
                return "LOCKED"

        if len(self.watching_scores) >= self.max_watching_frames:
            self._reset_to_idle()
            return "IDLE_RESET"

        if similarity < 0.25:
            recent = self.watching_scores[-3:]
            if len(recent) >= 3 and all(s < 0.30 for s in recent):
                self._reset_to_idle()
                return "IDLE_RESET"

        return None

    def _handle_locked(self, similarity, face_detected):
        """
        LOCKED state: tracker is primary driver.
        Only re-verify face periodically, not every frame.
        """
        self.frames_since_verify += 1

        if not face_detected:
            self.frames_since_face += 1
            return None

        self.frames_since_face = 0

        # Not yet time to re-verify — ignore face similarity
        if self.frames_since_verify < self.reverify_interval:
            return None

        # --- Time to re-verify ---
        self.frames_since_verify = 0

        if similarity >= self.reverify_ok:
            # Re-verify passed
            self.consecutive_verify_fails = 0
            return None

        if similarity < self.reverify_drop:
            # Re-verify failed
            self.consecutive_verify_fails += 1
            if self.consecutive_verify_fails >= self.reverify_max_fails:
                self._reset_to_idle()
                return "IDLE_DROP"
            return None

        # Ambiguous range (between drop and ok) — don't count as fail, just reset
        return None

    def _reset_to_idle(self):
        self.state = TrackState.IDLE
        self.consecutive_above = 0
        self.watching_scores = []
        self.has_high_frame = False
        self.frames_since_verify = 0
        self.consecutive_verify_fails = 0


# ======================================================================
# Main tracker
# ======================================================================

class PersonTracker:
    """
    ByteTrack-based person tracker with confidence-accumulation state machine.

    Combines:
    - ByteTrack for frame-to-frame identity via IoU + Kalman filter
    - Per-track state machine (IDLE → WATCHING → LOCKED)
    - Soft lock with periodic face re-verification
    - Body appearance (HSV histogram) for re-ID after occlusion
    - Velocity prediction for smooth tracking through occlusion
    - 2-second bbox-lost timeout for LOCKED tracks
    """

    def __init__(self,
                 track_activation_threshold=0.25,
                 lost_track_buffer=30,
                 minimum_matching_threshold=0.8,
                 frame_rate=30,
                 watch_threshold=0.35,
                 watch_frames=2,
                 trigger_score=0.40,
                 trigger_high=0.50,
                 max_watching_frames=10,
                 reverify_interval=30,
                 reverify_ok=0.35,
                 reverify_drop=0.25,
                 reverify_max_fails=3,
                 bbox_lost_timeout=2.0):
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        self._sm_cfg = dict(
            watch_threshold=watch_threshold,
            watch_frames=watch_frames,
            trigger_score=trigger_score,
            trigger_high=trigger_high,
            max_watching_frames=max_watching_frames,
            reverify_interval=reverify_interval,
            reverify_ok=reverify_ok,
            reverify_drop=reverify_drop,
            reverify_max_fails=reverify_max_fails,
        )
        self.bbox_lost_timeout = bbox_lost_timeout  # seconds
        self.tracked_persons: dict[int, TrackedPerson] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame, person_detections, face_results, face_recognizer):
        """
        Process one frame through ByteTrack + state machine.

        Returns:
            dict:
                tracked  – list of per-track info dicts for drawing
                alerts   – list of newly triggered LOCKED alerts
                active   – list of all currently LOCKED persons
        """
        now = time.monotonic()
        alerts = []

        if not person_detections:
            self._tick_lost_tracks(now)
            return {
                "tracked": [],
                "alerts": [],
                "active": self._active_tracks(),
            }

        # --- ByteTrack association ---
        bboxes = np.array([d["bbox"] for d in person_detections], dtype=np.float32)
        confs = np.array([d["confidence"] for d in person_detections], dtype=np.float32)
        detections = sv.Detections(xyxy=bboxes, confidence=confs)
        tracked_dets = self.byte_tracker.update_with_detections(detections)

        face_map = self._build_face_map(face_results)

        tracked_output = []
        active_ids = set()

        if tracked_dets.tracker_id is not None:
            for i, tid in enumerate(tracked_dets.tracker_id):
                tid = int(tid)
                active_ids.add(tid)
                bbox = tuple(tracked_dets.xyxy[i].astype(int))

                tp = self._get_or_create(tid)
                tp.update_appearance(frame, bbox)
                tp.last_bbox_seen_time = now  # bbox is visible this frame

                # Find best face match for this tracked bbox
                similarity, name, pid, face_info = self._best_face_for_track(
                    bbox, face_map, face_recognizer
                )
                face_detected = face_info is not None

                # Freeze identity once LOCKED — don't update name/similarity
                if tp.state != TrackState.LOCKED:
                    if name and (tp.person_name is None or similarity > tp.best_similarity):
                        tp.person_name = name
                        tp.person_id = pid
                    if similarity > tp.best_similarity:
                        tp.best_similarity = similarity

                transition = tp.update_state(similarity, face_detected)

                if transition == "LOCKED":
                    alerts.append({
                        "track_id": tid,
                        "person_name": tp.person_name,
                        "person_id": tp.person_id,
                        "bbox": bbox,
                        "similarity": tp.best_similarity,
                        "face_info": face_info,
                    })

                tracked_output.append({
                    "track_id": tid,
                    "bbox": bbox,
                    "state": tp.state,
                    "person_name": tp.person_name,
                    "person_id": tp.person_id,
                    "similarity": similarity,
                    "best_similarity": tp.best_similarity,
                    "face_info": face_info,
                    "velocity": tp.velocity,
                })

        # --- Recover lost LOCKED tracks via appearance + velocity ---
        for tid, tp in list(self.tracked_persons.items()):
            if tid in active_ids or tp.state != TrackState.LOCKED:
                continue

            recovered_bbox = self._try_recover(tp, frame, person_detections)
            if recovered_bbox:
                tp.update_appearance(frame, recovered_bbox)
                tp.last_bbox_seen_time = now
                tracked_output.append({
                    "track_id": tid,
                    "bbox": recovered_bbox,
                    "state": tp.state,
                    "person_name": tp.person_name,
                    "person_id": tp.person_id,
                    "similarity": 0.0,
                    "best_similarity": tp.best_similarity,
                    "face_info": None,
                    "velocity": tp.velocity,
                })

        # --- Check bbox-lost timeout for LOCKED tracks ---
        lost_drops = self._tick_lost_tracks(now, exclude=active_ids)
        for drop_tid in lost_drops:
            tp = self.tracked_persons.get(drop_tid)
            if tp:
                print(f"  [Tracker] T{drop_tid} LOCKED -> IDLE "
                      f"(bbox lost > {self.bbox_lost_timeout:.1f}s)")

        self._cleanup(active_ids)

        return {
            "tracked": tracked_output,
            "alerts": alerts,
            "active": self._active_tracks(),
        }

    def get_predicted_boxes(self):
        """Return predicted bboxes for all LOCKED persons (for skipped frames)."""
        results = []
        for tp in self.tracked_persons.values():
            if tp.state == TrackState.LOCKED:
                predicted = tp.predict_next_bbox()
                if predicted:
                    tp.last_bbox = predicted
                    results.append(self._prediction_payload(tp, predicted))
        return results

    def peek_predicted_boxes(self):
        """Return predicted bboxes without advancing tracker state."""
        results = []
        for tp in self.tracked_persons.values():
            if tp.state == TrackState.LOCKED:
                predicted = tp.predict_next_bbox()
                if predicted:
                    results.append(self._prediction_payload(tp, predicted))
        return results

    @staticmethod
    def _prediction_payload(tp, bbox):
        return {
            "track_id": tp.track_id,
            "bbox": bbox,
            "person_name": tp.person_name,
            "person_id": tp.person_id,
            "best_similarity": tp.best_similarity,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_or_create(self, track_id):
        if track_id not in self.tracked_persons:
            self.tracked_persons[track_id] = TrackedPerson(track_id, **self._sm_cfg)
        return self.tracked_persons[track_id]

    @staticmethod
    def _build_face_map(face_results):
        face_map: dict[tuple, list] = {}
        for fi in face_results:
            key = tuple(fi["person_bbox"])
            face_map.setdefault(key, []).append(fi)
        return face_map

    def _best_face_for_track(self, tracked_bbox, face_map, face_recognizer):
        best_sim = 0.0
        best_name = None
        best_pid = None
        best_fi = None

        for person_bbox, faces in face_map.items():
            if _iou(tracked_bbox, person_bbox) < 0.3:
                continue
            for fi in faces:
                emb = fi.get("embedding")
                if emb is None:
                    continue
                name, sim, pid = face_recognizer.match_raw(emb)
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
                    best_pid = pid
                    best_fi = fi

        return best_sim, best_name, best_pid, best_fi

    def _try_recover(self, tp, frame, person_detections):
        predicted = tp.predict_next_bbox()
        if predicted is None:
            return None

        best_score = 0.5
        best_bbox = None

        for det in person_detections:
            bbox = det["bbox"]
            iou_score = _iou(predicted, bbox)
            app_score = tp.appearance_similarity(frame, bbox)
            combined = 0.6 * iou_score + 0.4 * app_score
            if combined > best_score:
                best_score = combined
                best_bbox = bbox

        return best_bbox

    def _tick_lost_tracks(self, now, exclude=None):
        """Check bbox-lost timeout for LOCKED tracks. Returns list of dropped track IDs."""
        exclude = exclude or set()
        dropped = []
        for tid, tp in list(self.tracked_persons.items()):
            if tid in exclude or tp.state != TrackState.LOCKED:
                continue
            if tp.last_bbox_seen_time is None:
                continue
            elapsed = now - tp.last_bbox_seen_time
            if elapsed > self.bbox_lost_timeout:
                tp._reset_to_idle()
                dropped.append(tid)
        return dropped

    def _active_tracks(self):
        return [
            {
                "track_id": tp.track_id,
                "person_name": tp.person_name,
                "person_id": tp.person_id,
                "bbox": tp.last_bbox,
                "velocity": tp.velocity,
                "best_similarity": tp.best_similarity,
            }
            for tp in self.tracked_persons.values()
            if tp.state == TrackState.LOCKED
        ]

    def _cleanup(self, active_ids):
        stale = []
        for tid, tp in self.tracked_persons.items():
            if tid in active_ids:
                continue
            if tp.state == TrackState.LOCKED:
                # LOCKED tracks are cleaned by bbox-lost timeout, not here
                continue
            if tp.state == TrackState.IDLE and tp.total_frames > 0:
                if tp.last_bbox_seen_time is None:
                    stale.append(tid)
                elif time.monotonic() - tp.last_bbox_seen_time > 10.0:
                    stale.append(tid)
        for tid in stale:
            del self.tracked_persons[tid]


# ======================================================================
# Helpers
# ======================================================================

def _iou(box_a, box_b):
    """Compute IoU between two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
