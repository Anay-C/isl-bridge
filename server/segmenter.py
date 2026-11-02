"""Split a live landmark stream into individual signs.

A sign is "active" while at least one hand is visible and raised out of the rest position. A sign ends after
END_GAP inactive frames, or when hands hold still for HOLD frames mid-signing (lets people chain signs without
dropping hands).

Rest position (all in shoulder widths below the shoulder line): a wrist lower than REST_DROP is resting, a wrist
higher than LOW_DROP is signing, and in between it is resting only when it hangs below its elbow (arm down by
the side). With the phone close, resting hands leave the frame anyway; with the signer standing back (whole body
in view), resting hands stay visible at hip height (~1.6 shoulder widths down): the old rule (wrist above 2.2)
counted 91% of INCLUDE's rest frames as signing, so the hands-down pause was never seen and the sentence waited
for the 12 s length cap. The elbow rule counts 0.6% of rest frames as signing and keeps 98-100% of the signing
frames of CISLR / ISLRTC / CSLTR (measured 1 Nov on 6,000 clips).
"""
import os

import numpy as np

REST_DROP = 2.2
LOW_DROP = 1.2
HANG = 0.5
START_FRAMES = 3
END_GAP = 8
HOLD = 10
HOLD_SPEED = 0.025  # shoulder-widths per frame
MIN_LEN, MAX_LEN = int(os.environ.get("SEG_MIN_LEN", 8)), 120  # 10 -> 8 (2 Nov): silent clips 3.4% -> 2.4%
# A stretch starts after START_FRAMES raised frames among the last START_WINDOW (not necessarily in a row): the moving
# hand of a fast sign drops out of hand detection for single frames (INDIA, FRIEND, MARKET in INCLUDE), and a strict
# run of 3 never happened, so the app did not react at all.
START_WINDOW = int(os.environ.get("SEG_START_WINDOW", 5))
# Context around a stretch. Training clips run from hand-visible to hand-visible, so they include the arm rising
# from rest and falling back; a stretch cut exactly at "raised" misses both (held-out clips: median 11 hand-visible
# frames before and 18 after). Classifying the stretch with that context back: 73.2% -> 80.3% top-1 (2 Nov,
# 624 held-out INCLUDE + CISLR clips). PRE_ROLL frames from before the start are already buffered; the END_GAP
# hands-down frames that end a stretch are kept instead of trimmed (KEEP_TAIL). Neither adds latency.
PRE_ROLL = int(os.environ.get("SEG_PRE_ROLL", 12))
KEEP_TAIL = os.environ.get("SEG_KEEP_TAIL", "1") == "1"
# The hand landmarker loses a fast-moving hand for many frames (motion blur: INCLUDE TODAY / SUNDAY / MARKET, dim
# phone video), while the pose model keeps tracking the wrist. A hand missing from the hand landmarker is judged by
# its pose wrist when the pose model sees it clearly (visibility >= POSE_WRIST_VIS; out-of-frame wrists are guessed
# with low visibility). A stretch still needs MIN_HAND_FRAMES frames with real hand landmarks to be recognised.
POSE_WRIST = os.environ.get("SEG_POSE_WRIST", "1") == "1"
POSE_WRIST_VIS = float(os.environ.get("SEG_POSE_WRIST_VIS", 0.6))
MIN_HAND_FRAMES = int(os.environ.get("SEG_MIN_HAND_FRAMES", 3))
# ...but only to BRIDGE a dropout: the pose wrist counts within POSE_BRIDGE frames of the last real hand detection,
# so a stretch never runs on pose alone (arms crossed / hands clasped after signing would otherwise hold it open).
POSE_BRIDGE = int(os.environ.get("SEG_POSE_BRIDGE", 10))


class Segmenter:
    def __init__(self, hold=HOLD, max_len=MAX_LEN, fps=None):  # fps: accepted, unused (frame counts measured best)
        """hold=None: never cut on a hold, return the whole stretch of signing (for sign spotting)."""
        self.hold, self.max_len = hold, max_len
        self.buf = []
        self.active = False
        self.on = self.off = self.still = self.extra = 0
        self.recent = []  # raised flags of the last START_WINDOW frames while waiting for a start
        self.since_hand = 10 ** 6  # frames since the hand landmarker last found a hand
        self.prev_wrists = None
        self.need_motion = False  # after a hold-ended sign, wait for movement before starting the next

    @staticmethod
    def _frame_state(pose, lh, rh, pose_ok=True):
        use_pose = POSE_WRIST and pose_ok
        if pose is None or (lh is None and rh is None and not use_pose):
            return False, None
        sw = np.linalg.norm(pose[11, :2] - pose[12, :2]) + 1e-6
        sh_y = (pose[11, 1] + pose[12, 1]) / 2

        def wrist_y(h, pw):  # hand landmarker's wrist; else the pose model's wrist when it is clearly visible
            if h is not None:
                return h[0, 1]
            if use_pose and pose.shape[1] > 3 and pose[pw, 3] >= POSE_WRIST_VIS:
                return pose[pw, 1]
            return None

        def up(y, elbow):  # elbow: pose 13 = left, 14 = right (hands are assigned by pose wrists 15 / 16)
            if y is None:
                return False
            drop = (y - sh_y) / sw
            return drop < LOW_DROP or (drop < REST_DROP and y - pose[elbow, 1] < HANG * sw)

        raised = up(wrist_y(lh, 15), 13) or up(wrist_y(rh, 16), 14)
        wrists = np.concatenate([(h[0, :2] if h is not None else pose[pw, :2]) for h, pw in ((lh, 15), (rh, 16))]) / sw
        return raised, wrists

    def push(self, frame, fps=None):
        """frame: dict(pose (33,4)|None, lh (21,3)|None, rh (21,3)|None, in pixel units). Returns a finished
        segment (list of frames) or None."""
        self.since_hand = 0 if (frame["lh"] is not None or frame["rh"] is not None) else self.since_hand + 1
        raised, wrists = self._frame_state(frame["pose"], frame["lh"], frame["rh"], self.since_hand <= POSE_BRIDGE)
        speed = 0.0 if wrists is None or self.prev_wrists is None else float(np.abs(wrists - self.prev_wrists).max())
        self.prev_wrists = wrists
        if not self.active:
            if self.need_motion:
                self.need_motion = raised and speed < HOLD_SPEED * 2
            self.recent = (self.recent + [bool(raised and not self.need_motion)])[-START_WINDOW:]
            self.on = sum(self.recent) if START_WINDOW > START_FRAMES else (self.on + 1 if self.recent[-1] else 0)
            self.buf = (self.buf + [frame])[-max(PRE_ROLL, START_FRAMES + 2):]
            if self.on >= START_FRAMES:
                self.active, self.off, self.still = True, 0, 0
                self.extra = max(0, len(self.buf) - (START_FRAMES + 2))  # context frames beyond the old pre-roll
            return None
        self.buf.append(frame)
        self.off = 0 if raised else self.off + 1
        self.still = self.still + 1 if (raised and speed < HOLD_SPEED) else 0
        held = self.hold is not None and self.still >= self.hold and len(self.buf) > 20
        done = self.off >= END_GAP or len(self.buf) >= self.max_len or held
        if not done:
            return None
        core = len(self.buf) - self.off - self.extra  # what the old segmenter returned: the length check is unchanged
        seg = self.buf if (KEEP_TAIL or not self.off) else self.buf[: len(self.buf) - self.off]
        self.need_motion = self.off < END_GAP
        self.buf, self.active, self.on, self.still, self.extra, self.recent = [], False, 0, 0, 0, []
        # a stretch found through the pose wrists alone carries no hand shape to recognise: never guess from it
        hands = sum(f["lh"] is not None or f["rh"] is not None for f in seg)
        return seg if core >= MIN_LEN and hands >= MIN_HAND_FRAMES else None

    @property
    def signing(self):
        return self.active

    def current(self):
        """The utterance so far (for live guesses while the person is still signing)."""
        return self.buf if self.active else []
