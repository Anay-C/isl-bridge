"""Pick a clean reference recording per sign for the ghost (Talk-mode ghost signer + Learn ghost + coach scoring)
when data/clips/sources.json does not say which recording the clip was cut from ("exists").

The old fallback took the first file found, which for some signs was a long ISLRTC dictionary video (Hurt 45 s,
Patient 24 s), not the recording the clip shows. Per sign:
 1. recover the clip's real source: make_clips.py's source order (ISLRTC, INCLUDE, CISLR), and the trimmed length
    of the recording must match the clip's length (+-0.1 s) -> the Learn ghost stays in sync with the video;
 2. otherwise: a clean recording (hand span 0.8-4 s, hands tracked in most frames) that the recogniser reads most
    confidently as that sign (= a typical way of signing it).
Writes data/clips/ghost_sources.json ({label: "root:file.npz"}), read by coach._source.

    python ml/pick_ghost_refs.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

from coach import KP
from features import npz_frames
from recognizer import Recognizer

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "data" / "clips"
MAX_PER_ROOT = 40


def trimmed_seconds(p):
    """Length of the clip make_clips.py would cut from this recording (hand frames +- 0.3 s)."""
    d = np.load(p)
    fps, n = float(d["fps"]), len(d["pose"])
    h = np.flatnonzero(d["lh_p"] | d["rh_p"])
    s, e = (h[0], h[-1]) if len(h) else (0, n - 1)
    s, e = max(0, s - int(0.3 * fps)), min(n - 1, e + int(0.3 * fps))
    return (e - s + 1) / fps


def clip_seconds(label):
    v = cv2.VideoCapture(str(CLIPS / f"{label}.mp4"))
    n, fps = v.get(cv2.CAP_PROP_FRAME_COUNT), v.get(cv2.CAP_PROP_FPS)
    v.release()
    return n / fps if fps else None


def span(p):
    d = np.load(p)
    hp = d["lh_p"] | d["rh_p"]
    idx = np.flatnonzero(hp)
    if not len(idx):
        return 0.0, 0.0
    s, e = idx[0], idx[-1]
    return (e - s) / float(d["fps"]), float(hp[s:e + 1].mean())


def main():
    rec = Recognizer(str(ROOT / "data" / "model.pt"))
    cid = {c: i for i, c in enumerate(rec.classes)}
    sources = json.loads((CLIPS / "sources.json").read_text())
    out = {}
    for label, src in sorted(sources.items()):
        if ":" in src:
            continue
        cands = []
        for name, root in KP.items():
            d = root / label
            if d.is_dir():
                cands += [(name, p) for p in sorted(d.glob("*.npz"))]
        dur = clip_seconds(label)
        match = []
        for name in KP:  # make_clips.py order: the first source with candidates made the clip
            match = [(name, p) for n, p in cands if n == name and dur and abs(trimmed_seconds(p) - dur) <= 0.1]
            if match:
                break
        if match:
            if label in cid and len(match) > 1:
                embs = rec.embed_frames([npz_frames(p) for _, p in match])
                name, p = match[int(np.argmax(embs @ rec.weights[cid[label]]))]
            else:
                name, p = sorted(match, key=lambda x: ("(Sign" in x[1].name, x[1].name))[0]
            out[label] = f"{name}:{p.name}"
            print(f"  {label}: clip source {name}/{p.name} ({len(match)} length match)")
            continue
        cands = [c for name in KP for c in [x for x in cands if x[0] == name][:MAX_PER_ROOT]]
        ok = []
        for name, p in cands:
            sec, cover = span(p)
            if 0.8 <= sec <= 4.0 and cover >= 0.8:
                ok.append((name, p, sec))
        if not ok:
            print(f"  {label}: no clean candidate among {len(cands)}, keeping the old fallback")
            continue
        if label in cid:
            embs = rec.embed_frames([npz_frames(p) for _, p, _ in ok])
            conf = [float(rec.weights[cid[label]] @ e) for e in embs]
        else:
            conf = [-abs(sec - 2.0) for _, _, sec in ok]
        best = int(np.argmax(conf))
        name, p, sec = ok[best]
        out[label] = f"{name}:{p.name}"
        print(f"  {label}: {name}/{p.name} ({sec:.1f} s, cos {conf[best]:.2f}, {len(ok)}/{len(cands)} clean)")
    (CLIPS / "ghost_sources.json").write_text(json.dumps(out, indent=1))
    print(f"{len(out)} signs -> {CLIPS / 'ghost_sources.json'}")


if __name__ == "__main__":
    main()
