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
        # last_detection_bbox = actual bbox from YOLO/ByteTrack (never a prediction).
        # last_bbox        = current drawing position; may be a prediction on skipped frames.
        self.last_detection_bbox = None
        self.bbox_history = []
        # Face bbox offset relative to the person bbox top-left, captured when
        # a face was last detected. Lets us re-project a face box onto any
        # predicted person bbox so the green face overlay tracks the person
        # between face-detection frames instead of disappearing.
        self.last_face_offset = None  # (dx1, dy1, dx2, dy2) or None

        # --- Bbox-lost tracking (wall-clock) ---
        self.last_bbox_seen_time = None   # set by PersonTracker
        self.frames_since_face = 0
        self.total_frames = 0

    # ------------------------------------------------------------------
    # Appearance model
    # ------------------------------------------------------------------

    def update_appearance(self, frame, bbox):
        """Update body appearance model with current crop.

        Called on processed frames when a real detection is available.
        Velocity is computed from actual-detection positions only, so predicted
        positions on skipped frames never contaminate the motion model.
        """
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

        actual_bbox = (x1, y1, x2, y2)

        # Velocity: compute from last ACTUAL detection, not last drawn/predicted bbox.
        # Value is displacement per processed frame (one full frame_skip cycle).
        if self.last_detection_bbox is not None:
            prev_cx = (self.last_detection_bbox[0] + self.last_detection_bbox[2]) / 2
            prev_cy = (self.last_detection_bbox[1] + self.last_detection_bbox[3]) / 2
            curr_cx = (x1 + x2) / 2
            curr_cy = (y1 + y2) / 2
            self.velocity = (curr_cx - prev_cx, curr_cy - prev_cy)

        self.last_detection_bbox = actual_bbox
        self.last_bbox = actual_bbox
        self.bbox_history.append(actual_bbox)
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

    def update_face_offset(self, face_bbox, person_bbox):
        """Store face bbox as an offset from the person bbox top-left."""
        if face_bbox is None or person_bbox is None:
            return
        px1, py1, _, _ = person_bbox
        fx1, fy1, fx2, fy2 = face_bbox
        self.last_face_offset = (
            int(fx1) - int(px1),
            int(fy1) - int(py1),
            int(fx2) - int(px1),
            int(fy2) - int(py1),
        )

    def current_face_bbox(self):
        """Project the stored face offset onto the current person bbox."""
        if self.last_face_offset is None or self.last_bbox is None:
            return None
        px1, py1, _, _ = self.last_bbox
        dx1, dy1, dx2, dy2 = self.last_face_offset
        return (int(px1 + dx1), int(py1 + dy1), int(px1 + dx2), int(py1 + dy2))

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
        """Hard reset: clear identity, appearance, and all state counters.

        When a LOCKED track drops (out-of-frame, re-verify fail, bbox-lost
        timeout), we must NOT let the previous identity leak into whatever
        person next uses this track slot. Otherwise a new face can be
        mis-labeled with the old locked person's name.
        """
        self.state = TrackState.IDLE
        # Identity
        self.person_name = None
        self.person_id = None
        self.best_similarity = 0.0
        # State-machine counters
        self.consecutive_above = 0
        self.watching_scores = []
        self.has_high_frame = False
        self.frames_since_verify = 0
        self.consecutive_verify_fails = 0
        # Appearance model (so stale colors don't haunt future matches)
        self.color_histogram = None
        self.velocity = (0.0, 0.0)


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
                 bbox_lost_timeout=2.0,
                 predict_draw_timeout=0.5,
                 out_of_frame_margin=0.6):
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
        self.bbox_lost_timeout = bbox_lost_timeout        # seconds — full drop after occlusion
        self.predict_draw_timeout = predict_draw_timeout  # seconds — stop drawing phantom boxes
        self.out_of_frame_margin = out_of_frame_margin    # drop if >this fraction of bbox exits the frame
        self._frame_shape = None                          # (h, w) – learned on each update
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
        self._frame_shape = frame.shape[:2] if frame is not None else self._frame_shape

        if not person_detections:
            self._tick_lost_tracks(now)
            self._drop_out_of_frame_tracks(now)
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

        tracked_output = []
        active_ids = set()

        # Pre-compute a 1:1 assignment from faces to tracked bboxes. Using
        # IoU per-track independently (old behaviour) lets a single face match
        # multiple overlapping tracks, which causes identity bleed — e.g. when
        # someone holds up a photo of a missing person, both the user and the
        # photo would get labeled with the photo's identity. A greedy bipartite
        # assignment (best containment wins, each face used once) fixes that.
        track_entries = []
        if tracked_dets.tracker_id is not None:
            for i, tid in enumerate(tracked_dets.tracker_id):
                bbox = tuple(tracked_dets.xyxy[i].astype(int))
                track_entries.append((int(tid), bbox))
        face_by_track = self._assign_faces_to_tracks(track_entries, face_results)

        if tracked_dets.tracker_id is not None:
            for i, tid in enumerate(tracked_dets.tracker_id):
                tid = int(tid)
                active_ids.add(tid)
                bbox = tuple(tracked_dets.xyxy[i].astype(int))

                # Detect ByteTrack id reuse: the same track_id can be recycled
                # for a different person if the original track is lost for a
                # while. A huge spatial jump after a visual gap is a strong
                # signal that this is not the same person — wipe the slot so
                # the old identity doesn't stamp itself onto the newcomer.
                tp = self.tracked_persons.get(tid)
                if tp is not None and self._is_track_reused(tp, bbox, now):
                    del self.tracked_persons[tid]
                    tp = None

                if tp is None:
                    tp = self._get_or_create(tid)
                tp.update_appearance(frame, bbox)
                tp.last_bbox_seen_time = now  # bbox is visible this frame

                # Face match for this track (at most one face per track)
                face_info = face_by_track.get(tid)
                if face_info is not None and face_info.get("embedding") is not None:
                    name, similarity, pid = face_recognizer.match_raw(
                        face_info["embedding"]
                    )
                    # Remember where the face sits inside the person bbox so we
                    # can redraw a face overlay on predicted frames even when
                    # face detection is skipped.
                    tp.update_face_offset(face_info["face_bbox"], bbox)
                else:
                    name, similarity, pid = None, 0.0, None
                    face_info = None
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
                    "face_bbox_predicted": tp.current_face_bbox(),
                    "velocity": tp.velocity,
                })

        # --- Recover lost LOCKED tracks via appearance + velocity ---
        # Bboxes that ByteTrack already attached to a track this frame are off
        # limits for recovery — otherwise a missing LOCKED track could "steal"
        # a detection that belongs to a completely different person and
        # mis-label them with the old locked identity.
        taken_bboxes = set()
        for tr in tracked_output:
            taken_bboxes.add(tuple(tr["bbox"]))

        for tid, tp in list(self.tracked_persons.items()):
            if tid in active_ids or tp.state != TrackState.LOCKED:
                continue

            candidates = [d for d in person_detections if tuple(d["bbox"]) not in taken_bboxes]
            recovered_bbox = self._try_recover(tp, frame, candidates)
            if recovered_bbox:
                taken_bboxes.add(tuple(recovered_bbox))
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
                    "face_bbox_predicted": tp.current_face_bbox(),
                    "velocity": tp.velocity,
                })

        # --- Drop LOCKED tracks whose predicted bbox left the frame ---
        self._drop_out_of_frame_tracks(now, exclude=active_ids)

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

    def get_predicted_boxes(self, locked_only=True):
        """Return predicted bboxes for tracked persons (for skipped frames).

        Advances `last_bbox` by velocity so consecutive skipped frames step forward.
        Velocity itself is NOT mutated here — it stays locked to the last two actual
        detections, so the prediction cadence matches the object's real motion.

        Skips phantom predictions: if a track hasn't had any real detection for
        longer than `predict_draw_timeout`, stop drawing it even while the track
        is still waiting on `bbox_lost_timeout` to fully drop.

        Args:
            locked_only: if True, only LOCKED persons (back-compat, smoother overlay).
                         If False, returns all active tracks with their state.
        """
        now = time.monotonic()
        results = []
        for tp in self.tracked_persons.values():
            if locked_only and tp.state != TrackState.LOCKED:
                continue
            if tp.last_bbox is None:
                continue
            # Don't draw phantom boxes for tracks whose actual detection has
            # been missing too long — they're almost certainly out of frame.
            if (tp.last_bbox_seen_time is not None
                    and (now - tp.last_bbox_seen_time) > self.predict_draw_timeout):
                continue
            predicted = tp.predict_next_bbox()
            if predicted is None:
                continue
            tp.last_bbox = predicted
            results.append(self._prediction_payload(tp, predicted))
        return results

    def peek_predicted_boxes(self, locked_only=True):
        """Return predicted bboxes without advancing tracker state."""
        results = []
        for tp in self.tracked_persons.values():
            if locked_only and tp.state != TrackState.LOCKED:
                continue
            predicted = tp.predict_next_bbox()
            if predicted:
                results.append(self._prediction_payload(tp, predicted))
        return results

    @staticmethod
    def _prediction_payload(tp, bbox):
        # Project the stored face offset onto the predicted person bbox so the
        # face overlay follows the person between face-detection frames.
        face_bbox = None
        if tp.last_face_offset is not None:
            dx1, dy1, dx2, dy2 = tp.last_face_offset
            face_bbox = (
                int(bbox[0] + dx1),
                int(bbox[1] + dy1),
                int(bbox[0] + dx2),
                int(bbox[1] + dy2),
            )
        return {
            "track_id": tp.track_id,
            "bbox": bbox,
            "state": tp.state,
            "person_name": tp.person_name,
            "person_id": tp.person_id,
            "similarity": 0.0,
            "best_similarity": tp.best_similarity,
            "face_bbox_predicted": face_bbox,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_or_create(self, track_id):
        if track_id not in self.tracked_persons:
            self.tracked_persons[track_id] = TrackedPerson(track_id, **self._sm_cfg)
        return self.tracked_persons[track_id]

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

    def _assign_faces_to_tracks(self, track_entries, face_results,
                                min_containment=0.5):
        """Greedy 1:1 assignment of face detections to tracked bboxes.

        For each candidate (track, face) pair we compute how much of the face
        bbox lies inside the track bbox (containment ratio). Best containment
        wins; ties are broken by the smaller face-center → track-center
        distance. Each face and each track can only be used once.

        Returns:
            dict[track_id, face_info] — tracks without a viable face are absent.
        """
        if not track_entries or not face_results:
            return {}

        scored = []
        for tid, tbbox in track_entries:
            tx1, ty1, tx2, ty2 = tbbox
            tcx = (tx1 + tx2) / 2.0
            tcy = (ty1 + ty2) / 2.0
            for fi in face_results:
                fx1, fy1, fx2, fy2 = fi["face_bbox"]
                ix1, iy1 = max(tx1, fx1), max(ty1, fy1)
                ix2, iy2 = min(tx2, fx2), min(ty2, fy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
                contain = inter / face_area
                if contain < min_containment:
                    continue
                fcx = (fx1 + fx2) / 2.0
                fcy = (fy1 + fy2) / 2.0
                dist = ((fcx - tcx) ** 2 + (fcy - tcy) ** 2) ** 0.5
                # Sort ascending: negate containment so best comes first
                scored.append((-contain, dist, tid, id(fi), fi))

        scored.sort()
        assigned = {}
        used_face_ids = set()
        for _, _, tid, fid, fi in scored:
            if tid in assigned or fid in used_face_ids:
                continue
            assigned[tid] = fi
            used_face_ids.add(fid)
        return assigned

    def _is_track_reused(self, tp, new_bbox, now):
        """Detect whether ByteTrack has recycled an id for a different person.

        Heuristic: a visual gap of >= reuse_gap_seconds combined with a big
        spatial jump (centroid displacement > reuse_jump_factor * bbox diag)
        almost certainly means we're looking at a new person — the previous
        occupant is long gone and ByteTrack just happened to reassign the slot.
        """
        if tp.last_detection_bbox is None or tp.last_bbox_seen_time is None:
            return False
        gap = now - tp.last_bbox_seen_time
        if gap < 0.4:
            return False
        px1, py1, px2, py2 = tp.last_detection_bbox
        nx1, ny1, nx2, ny2 = new_bbox
        prev_cx, prev_cy = (px1 + px2) / 2, (py1 + py2) / 2
        curr_cx, curr_cy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
        dist = ((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2) ** 0.5
        prev_diag = ((px2 - px1) ** 2 + (py2 - py1) ** 2) ** 0.5
        # Even a fast walk rarely jumps > 1.5 bbox-diagonals across a short
        # visual gap. If it does, ByteTrack almost certainly recycled this id.
        return dist > max(1.0, prev_diag) * 1.5

    def _drop_out_of_frame_tracks(self, now, exclude=None):
        """Drop LOCKED tracks whose predicted/last bbox has substantially left the frame.

        A person walking out of the frame shouldn't linger for the full
        `bbox_lost_timeout` — we can tell immediately by checking whether the
        most-recent known position sits outside the frame boundaries.
        """
        if self._frame_shape is None:
            return []
        exclude = exclude or set()
        h, w = self._frame_shape
        dropped = []
        for tid, tp in list(self.tracked_persons.items()):
            if tid in exclude or tp.state != TrackState.LOCKED:
                continue
            if tp.last_bbox is None:
                continue
            x1, y1, x2, y2 = tp.last_bbox
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            # Area clamped inside frame
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            inside_w = max(0, cx2 - cx1)
            inside_h = max(0, cy2 - cy1)
            inside_ratio = (inside_w * inside_h) / (bw * bh)
            if inside_ratio < (1.0 - self.out_of_frame_margin):
                tp._reset_to_idle()
                # Fully remove the track slot so any future track_id reuse starts fresh
                # and can't inherit this person's name / appearance.
                del self.tracked_persons[tid]
                dropped.append(tid)
                print(f"  [Tracker] T{tid} LOCKED -> DROPPED "
                      f"(out of frame, inside={inside_ratio:.0%})")
        return dropped

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
                del self.tracked_persons[tid]
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
                "bbox_history": list(tp.bbox_history),
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
