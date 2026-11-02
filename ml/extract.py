"""Extract MediaPipe Tasks keypoints (hands + pose lite) from every downloaded INCLUDE video.

Output per video: data/keypoints/<label>/<video stem>.npz with raw landmarks
    pose (T,33,4) x,y,z,visibility   lh/rh (T,21,3)   lh_p/rh_p (T,) presence   fps, width, height
Coordinates stay in MediaPipe's normalised image space; ml/features.py does the
shoulder-centred normalisation so Python and the browser share one implementation.

Rules: same .task models as the browser, unmirrored frames, left/right hand assigned by
distance to pose wrists (15 = left, 16 = right), strictly increasing VIDEO-mode timestamps per landmarker.

    python ml/extract.py [--workers 12] [--watch]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VIDEOS = DATA / "videos"
KEYPOINTS = DATA / "keypoints"
MODELS = DATA / "models"
MAX_SIDE = 640  # browser camera frames are ~640 px; match that scale

_hand = _pose = None
_ts_ms = 0  # running timestamp offset across videos for this worker's landmarkers


HAND_CONF = float(os.environ.get("HAND_CONF", "0.3"))


def init_worker(videos=None, keypoints=None):
    global _hand, _pose, VIDEOS, KEYPOINTS
    if videos: VIDEOS = Path(videos)
    if keypoints: KEYPOINTS = Path(keypoints)
    os.environ.setdefault("GLOG_minloglevel", "2")
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    _hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        # must match web/src/vision.ts: 0.5 defaults drop the 2nd hand when hands touch (HELP)
        min_hand_detection_confidence=HAND_CONF, min_hand_presence_confidence=HAND_CONF,
        min_tracking_confidence=HAND_CONF))
    _pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))


def assign_hands(hands, pose):
    """Return (left, right) hand arrays (21,3) or None, matched to pose wrists 15/16."""
    left = right = None
    if not hands:
        return left, right
    if pose is None:  # no pose: fall back to image side (unmirrored: signer's right hand is on image left)
        for h in hands:
            if h[0, 0] < 0.5: right = h
            else: left = h
        return left, right
    lw, rw = pose[15, :2], pose[16, :2]
    if len(hands) == 1:
        h = hands[0]
        if np.linalg.norm(h[0, :2] - lw) < np.linalg.norm(h[0, :2] - rw): left = h
        else: right = h
        return left, right
    a, b = hands[0], hands[1]
    cost_ab = np.linalg.norm(a[0, :2] - lw) + np.linalg.norm(b[0, :2] - rw)
    cost_ba = np.linalg.norm(b[0, :2] - lw) + np.linalg.norm(a[0, :2] - rw)
    return (a, b) if cost_ab <= cost_ba else (b, a)


def process(video_path):
    global _ts_ms
    import cv2
    import mediapipe as mp

    rel = Path(video_path).relative_to(VIDEOS)
    out = (KEYPOINTS / rel).with_suffix(".npz")
    if out.exists():
        return "skip", str(rel), 0
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step_ms = 1000.0 / fps
    poses, lhs, rhs, lps, rps = [], [], [], [], []
    w = h = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fh, fw = frame.shape[:2]
        s = MAX_SIDE / max(fh, fw)
        if s < 1:
            frame = cv2.resize(frame, (round(fw * s), round(fh * s)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        _ts_ms += int(round(step_ms))
        pr = _pose.detect_for_video(img, _ts_ms)
        hr = _hand.detect_for_video(img, _ts_ms)
        pose = None
        if pr.pose_landmarks:
            pose = np.array([[p.x, p.y, p.z, p.visibility] for p in pr.pose_landmarks[0]], np.float32)
        hands = [np.array([[p.x, p.y, p.z] for p in hl], np.float32) for hl in hr.hand_landmarks]
        lh, rh = assign_hands(hands, pose)
        poses.append(pose if pose is not None else np.zeros((33, 4), np.float32))
        lhs.append(lh if lh is not None else np.zeros((21, 3), np.float32))
        rhs.append(rh if rh is not None else np.zeros((21, 3), np.float32))
        lps.append(lh is not None)
        rps.append(rh is not None)
    cap.release()
    _ts_ms += 1000  # gap between videos
    if not poses:
        return "empty", str(rel), 0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, pose=np.stack(poses), lh=np.stack(lhs), rh=np.stack(rhs),
                        lh_p=np.array(lps), rh_p=np.array(rps), fps=fps, width=w, height=h)
    tmp.replace(out)
    return "ok", str(rel), len(poses)


def pending():
    vids = [p for p in VIDEOS.rglob("*") if p.suffix.lower() in (".mov", ".mp4", ".avi", ".mkv")]
    return sorted(p for p in vids if not (KEYPOINTS / p.relative_to(VIDEOS)).with_suffix(".npz").exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--watch", action="store_true", help="keep polling for new downloads")
    ap.add_argument("--videos", default=None, help="input root (<label>/<video>), default data/videos")
    ap.add_argument("--keypoints", default=None, help="output root, default data/keypoints")
    args = ap.parse_args()
    global VIDEOS, KEYPOINTS
    if args.videos: VIDEOS = Path(args.videos).resolve()
    if args.keypoints: KEYPOINTS = Path(args.keypoints).resolve()
    with Pool(args.workers, initializer=init_worker, initargs=(str(VIDEOS), str(KEYPOINTS))) as pool:
        while True:
            todo = pending()
            if todo:
                t0, frames = time.time(), 0
                print(f"{len(todo)} videos to extract with {args.workers} workers", flush=True)
                for i, (status, rel, n) in enumerate(pool.imap_unordered(process, todo, chunksize=2), 1):
                    frames += n
                    if status != "ok" or i % 25 == 0 or i == len(todo):
                        el = time.time() - t0
                        print(f"[{i}/{len(todo)}] {status} {rel}  {frames / el:.0f} frames/s  "
                              f"ETA {(len(todo) - i) * el / i / 60:.1f} min", flush=True)
            if not args.watch:
                break
            time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
