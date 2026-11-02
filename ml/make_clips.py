"""Build one playable ISL clip per sign for the speech -> sign direction: data/clips/<label>.mp4

Picks the most typical clip of each sign (embedding closest to its trained class weight), trims it to the
signing portion (hands visible, from keypoints), crops around the signer, encodes H.264 for browsers.
Source order: ISLRTC dictionary (one consistent HD signer, the official reference), then INCLUDE (AI4Bharat),
then CISLR (IIT Kanpur). Covers the trained vocabulary plus the extra hospital words in data/islrtc/index.json.

    python ml/make_clips.py [--force]
"""
import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from features import npz_frames
from recognizer import Recognizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "clips"
SRC = [("islrtc", DATA / "islrtc" / "keypoints", DATA / "islrtc" / "videos"),
       ("include", DATA / "keypoints", DATA / "videos"),
       ("cislr", DATA / "cislr" / "keypoints", DATA / "cislr" / "videos")]
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
VIDEO_EXT = (".mov", ".mp4", ".avi", ".mkv", ".webm")
MAX_SIGN_S = 4.0  # ISLRTC dictionary entries longer than this are a sign + signed definition, not one sign


def signing_seconds(npz):
    d = np.load(npz)
    h = np.flatnonzero(d["lh_p"] | d["rh_p"])
    return (h[-1] - h[0] + 1) / float(d["fps"]) if len(h) else 0.0


def find_video(vroot, rel_npz):
    base = vroot / rel_npz.with_suffix("")
    for ext in VIDEO_EXT + tuple(e.upper() for e in VIDEO_EXT):
        if base.with_suffix(ext).exists():
            return base.with_suffix(ext)
    return None


def trim_and_crop(npz):
    d = np.load(npz)
    fps = float(d["fps"])
    hands = np.flatnonzero(d["lh_p"] | d["rh_p"])
    s, e = (hands[0], hands[-1]) if len(hands) else (0, len(d["pose"]) - 1)
    s, e = max(0, s - int(0.3 * fps)), min(len(d["pose"]) - 1, e + int(0.3 * fps))
    pose = d["pose"][s:e + 1]
    ok = pose[:, 11, 3] > 0.3
    p = pose[ok] if ok.any() else pose
    sh = np.abs(p[:, 11, 0] - p[:, 12, 0]).mean()
    cx = (p[:, 11, 0] + p[:, 12, 0]).mean() / 2
    top = max(0.0, float(p[:, 0, 1].mean() - 1.0 * sh))
    # frame: ~3.4 shoulder widths wide (arms out), from above head to below waist
    w = min(1.0, 3.4 * sh)
    x0 = float(np.clip(cx - w / 2, 0, 1 - w))
    return s / fps, (e - s + 1) / fps, (x0, top, w, 1.0 - top)


def encode(src, dst, start, dur, crop):
    x0, y0, w, h = crop
    vf = (f"crop=iw*{w:.4f}:ih*{h:.4f}:iw*{x0:.4f}:ih*{y0:.4f},"
          "scale=-2:480,fps=25,format=yuv420p")
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(src),
                    "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                    "-movflags", "+faststart", str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    rec = Recognizer(DATA / "model.pt")
    cid = {c: i for i, c in enumerate(rec.classes)}
    v = json.loads((ROOT / "ml" / "vocab.json").read_text())
    made = {}
    idx = DATA / "islrtc" / "index.json"
    extra = list(json.loads(idx.read_text())) if idx.exists() else []
    for label in dict.fromkeys(v["include_signs"] + v["cislr_signs"] + v.get("common_signs", []) + extra):
        dst = OUT / f"{label}.mp4"
        if dst.exists() and not args.force:
            made[label] = "exists"
            continue
        for name, kroot, vroot in SRC:
            npzs = sorted((kroot / label).glob("*.npz")) if (kroot / label).is_dir() else []
            if name == "islrtc":
                npzs = [p for p in npzs if signing_seconds(p) <= MAX_SIGN_S]
            if not npzs:
                continue
            if label in cid:  # most typical = closest to trained class weight
                embs = rec.embed_frames([npz_frames(p) for p in npzs])
                best = npzs[int(np.argmax(embs @ rec.weights[cid[label]]))]
            else:  # prefer the main entry over "(Sign 2)" variants
                best = sorted(npzs, key=lambda p: ("(Sign" in p.name, p.name))[0]
            src = find_video(vroot, best.relative_to(kroot))
            if src is None:
                continue
            start, dur, crop = trim_and_crop(best)
            encode(src, dst, start, dur, crop)
            made[label] = f"{name}:{src.name}"
            break
        print(f"{label:14s} {made.get(label, 'MISSING')}", flush=True)
    (OUT / "sources.json").write_text(json.dumps(made, indent=1))
    print(f"{sum(1 for x in made.values())} clips -> {OUT}")


if __name__ == "__main__":
    main()
