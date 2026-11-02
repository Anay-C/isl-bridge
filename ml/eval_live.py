"""Live-pipeline test on signers the model has never seen (ISL-CSLTR, CC BY 4.0, 7 signers).

    python ml/eval_live.py [--model data/model.pt]

Replays each video's keypoints frame by frame through the SAME Segmenter + Recognizer the server runs, so the
numbers are what the app does live (segmentation included), not clean pre-cut clips:
  words      word videos whose word is in our vocabulary: is the first detected sign right (top-1 / top-3)?
             also: did the segmenter find exactly one sign?
  sentences  sentence videos: which of the sentence's vocabulary words show up in the detected sign sequence
             (top-1 / anywhere in top-3), and how many signs were detected vs words in the sentence.
Keypoints come from data/csltr/keypoints (ml/extract.py), i.e. the same MediaPipe Tasks models as the browser.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "server"))
from features import clip_frames  # noqa: E402
from recognizer import Recognizer  # noqa: E402
from segmenter import Segmenter  # noqa: E402
import spotting  # noqa: E402

KP = ROOT / "data" / "csltr" / "keypoints"
# CSLTR word -> our label (only where the meaning is the same)
ALIAS = {"hi": "Hello", "thank": "Thank you", "thanks": "Thank you", "fine": "good", "medicines": "Medicine",
         "you": "you", "i": "I", "me": "I", "my": "I", "he": "he", "she": "she", "we": "we", "they": "they"}


DROPOUT = 0.0  # set by --dropout


def vocab_map(labels):
    m = {l.lower(): l for l in labels}
    m.update({k: v for k, v in ALIAS.items() if v in labels})
    return m


def frames_of(npz):
    """npz keypoints (normalised) -> list of server-style frames in pixel units."""
    d = np.load(npz)
    w, h = float(d["width"]), float(d["height"])
    px = lambda a, n: (a * [w, h, w] + [0, 0, 0]) if n == 3 else a
    out = []
    for t in range(len(d["pose"])):
        pose = d["pose"][t].copy()
        pose[:, 0] *= w
        pose[:, 1] *= h
        pose[:, 2] *= w
        f = {"pose": pose, "w": w, "h": h}
        for k in ("lh", "rh"):
            f[k] = (d[k][t] * [w, h, w]).astype(np.float32) if d[k + "_p"][t] else None
        out.append(f)
    if DROPOUT:  # phone-like tracking loss: hands vanish for 1-7 frames (seen in live logs)
        rng = np.random.default_rng(abs(hash(str(npz))) % 2**32)
        t = 0
        while t < len(out):
            if rng.random() < DROPOUT:
                for u in range(t, min(len(out), t + int(rng.integers(1, 8)))):
                    out[u] = dict(out[u], lh=None, rh=None)
                t += 8
            t += 1
    return out + [hands_down(out[-1])] * 20  # signer drops hands at the end


def hands_down(f):
    """The frame `f` with the arms hanging at rest (what the camera sees when the signer drops the hands): no hand
    landmarks, and the pose's elbows / wrists / hand points moved below the shoulders, hanging (the segmenter also
    reads pose wrists, so keeping raised arms in the pose would look like signing)."""
    pose = f["pose"].copy()
    sw = float(np.linalg.norm(pose[11, :2] - pose[12, :2])) or 1.0
    for sh, el, idx in ((11, 13, (15, 17, 19, 21)), (12, 14, (16, 18, 20, 22))):
        pose[el, :2] = pose[sh, :2] + [0.0, 1.0 * sw]
        for i in idx:
            pose[i, :2] = pose[sh, :2] + [0.0, 1.9 * sw]
    return dict(f, pose=pose, lh=None, rh=None)


def segments(frames, fps=25.0):
    seg, out = Segmenter(fps=fps), []
    for f in frames:
        s = seg.push(f)
        if s:
            out.append(s)
    return out


def spot(rec, frames, lens=(16, 24, 32, 44), stride=3, thr=0.35, max_overlap=0.3):
    """Sign spotting for continuous signing: score sliding windows of several lengths over the hands-visible
    stretch, then keep confident, non-overlapping peaks (greedy NMS), in time order -> [[top3 labels], ...]."""
    z = lambda k: np.zeros(k, np.float32)
    active = [i for i, f in enumerate(frames) if f["lh"] is not None or f["rh"] is not None]
    if not active:
        return []
    a, b = active[0], active[-1] + 1
    wins = [(s, s + L) for L in lens for s in range(a, max(a + 1, b - L + 1), stride) if s + L <= b + 2] or [(a, b)]
    feats = []
    for s, e in wins:
        seg = frames[s:e]
        pose = np.stack([f["pose"] for f in seg])
        lh = np.stack([f["lh"] if f["lh"] is not None else z((21, 3)) for f in seg])
        rh = np.stack([f["rh"] if f["rh"] is not None else z((21, 3)) for f in seg])
        lp = np.array([f["lh"] is not None for f in seg])
        rp = np.array([f["rh"] is not None for f in seg])
        feats.append(clip_frames(pose, lh, rh, lp, rp, 1, 1))
    embs = rec.embed_frames(feats)
    scored = []
    for (s, e), emb in zip(wins, embs):
        top = rec.predict(emb, k=3, personalise=False)
        scored.append((top[0]["p"], s, e, [t["label"] for t in top]))
    keep = []
    for p, s, e, labs in sorted(scored, reverse=True):
        if p < thr:
            break
        if all(min(e, e2) - max(s, s2) <= max_overlap * min(e - s, e2 - s2) for _, s2, e2, _ in keep):
            keep.append((p, s, e, labs))
    return [labs for _, s, _, labs in sorted(keep, key=lambda k: k[1])]


def server_words(rec, frames, fps):
    """Server pipeline (Segmenter(hold=None, max_len=360) + spotting.spot, fallback = whole segment).
    Returns the top-3 of the most confident detected sign (a single-word clip should yield that word)."""
    seg, outs = Segmenter(hold=None, max_len=360, fps=fps), []
    for f in frames:
        s = seg.push(f)
        if s:
            outs.append(s)
    best = None
    for s in outs:
        spots = spotting.spot(rec, s, fps=fps) or [(0, len(s), None)]
        for a, b, top in spots:
            top = top or [{"label": l, "p": 0.0} for l in predict(rec, s)]
            if best is None or top[0]["p"] > best[0]["p"]:
                best = top
    return [t["label"] for t in best] if best else []


def predict(rec, s):
    z = lambda k: np.zeros(k, np.float32)
    pose = np.stack([f["pose"] for f in s])
    lh = np.stack([f["lh"] if f["lh"] is not None else z((21, 3)) for f in s])
    rh = np.stack([f["rh"] if f["rh"] is not None else z((21, 3)) for f in s])
    lp = np.array([f["lh"] is not None for f in s])
    rp = np.array([f["rh"] is not None for f in s])
    emb = rec.embed_frames([clip_frames(pose, lh, rh, lp, rp, 1, 1)])[0]
    return [p["label"] for p in rec.predict(emb, k=3, personalise=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "data" / "model.pt"))
    ap.add_argument("--decoder", choices=["segmenter", "spot"], default="spot")
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--signers", default="", help="only these CSLTR signer ids (data/csltr/signer_clusters.json)")
    ap.add_argument("--dropout", type=float, default=0.0, help="per-frame chance of a 1-7 frame hand drop-out")
    ap.add_argument("--exemplars", default="", help="exemplar bank (ml/exemplars.py) made with the SAME model")
    ap.add_argument("--out", default="", help="report path (default data/eval_live.json)")
    args = ap.parse_args()
    global DROPOUT
    DROPOUT = args.dropout
    rec = Recognizer(args.model)
    if args.exemplars:
        rec.set_exemplars(args.exemplars)
    sig = json.loads((ROOT / "data" / "csltr" / "signer_clusters.json").read_text()) if args.signers else {}
    want_s = {int(x) for x in args.signers.split(",") if x}
    norm = lambda f: str(f.relative_to(KP)).replace("/", "\\")
    keep = lambda f: not want_s or sig.get(norm(f), sig.get(str(f.relative_to(KP)))) in want_s
    v = json.loads((ROOT / "ml" / "vocab.json").read_text())
    rec.active = set(v["include_signs"]) | set(v["cislr_signs"]) | set(v.get("common_signs", []))
    vm = vocab_map(rec.sign_labels())
    report = {}

    # ---- words --------------------------------------------------------------------------------------
    per = defaultdict(lambda: [0, 0, 0, 0])  # n, top1, top3, exactly-one-segment
    for d in sorted((KP / "words").iterdir()) if (KP / "words").exists() else []:
        label = vm.get(d.name)
        if not label:
            continue
        for f in sorted(d.glob("*.npz")):
            if not keep(f):
                continue
            fr = frames_of(f)
            segs = segments(fr, float(np.load(f)["fps"]))
            if args.decoder == "spot":  # exactly what server/app.py does: whole utterance, then spotting
                p = server_words(rec, fr, float(np.load(f)["fps"]))
            else:
                p = predict(rec, max(segs, key=len)) if segs else []
            r = per[label]
            r[0] += 1
            r[1] += bool(p) and p[0] == label
            r[2] += label in p
            r[3] += len(segs) == 1
    n = sum(r[0] for r in per.values())
    if n:
        report["words"] = {"clips": n, "signs": len(per), "top1": sum(r[1] for r in per.values()) / n,
                           "top3": sum(r[2] for r in per.values()) / n,
                           "one_segment": sum(r[3] for r in per.values()) / n,
                           "per_sign": {l: {"n": r[0], "top1": r[1], "top3": r[2]} for l, r in sorted(per.items())}}
        print(f"WORDS (unseen signers, live segmentation): {n} clips, {len(per)} signs | top-1 {report['words']['top1']:.1%}"
              f" top-3 {report['words']['top3']:.1%} | segmenter found exactly 1 sign in {report['words']['one_segment']:.0%}")
        for l, r in sorted(per.items(), key=lambda x: x[1][1] / x[1][0]):
            print(f"   {l:12s} {r[1]}/{r[0]} top-1   {r[2]}/{r[0]} top-3")

    # ---- sentences ----------------------------------------------------------------------------------
    hit1 = hit3 = want = 0
    nseg, nword, rows = [], [], []
    for d in sorted((KP / "sentences").iterdir()) if (KP / "sentences").exists() else []:
        words = re.findall(r"[a-z]+", d.name.lower())
        targets = [vm[w] for w in words if w in vm]
        for f in sorted(d.glob("*.npz")):
            if not keep(f):
                continue
            fr = frames_of(f)
            if args.decoder == "spot":  # what the live server does (ml/spotting.py)
                fps = float(np.load(f)["fps"])
                preds = [[t["label"] for t in top] for _, _, top in spotting.spot(rec, fr, fps=fps, thr=args.thr)]
                segs = preds
            else:
                segs = segments(fr)
                preds = [predict(rec, s) for s in segs]
            top1 = {p[0] for p in preds}
            top3 = {x for p in preds for x in p}
            hit1 += sum(t in top1 for t in targets)
            hit3 += sum(t in top3 for t in targets)
            want += len(targets)
            nseg.append(len(segs))
            nword.append(len(words))
            rows.append({"sentence": d.name, "file": f.name, "targets": targets, "detected": [p[0] for p in preds]})
    if rows:
        report["sentences"] = {"videos": len(rows), "target_words": want, "recall_top1": hit1 / max(1, want),
                               "recall_top3": hit3 / max(1, want), "signs_detected_per_video": float(np.mean(nseg)),
                               "words_per_sentence": float(np.mean(nword))}
        print(f"\nSENTENCES (continuous, live segmentation): {len(rows)} videos | vocabulary words found: "
              f"top-1 {hit1}/{want} ({hit1 / max(1, want):.1%}), in top-3 {hit3}/{want} | signs detected per video "
              f"{np.mean(nseg):.1f} vs {np.mean(nword):.1f} words")
        for r in rows[:12]:
            print(f"   {r['sentence'][:34]:34s} want {r['targets']} got {r['detected']}")
    out = Path(args.out) if args.out else ROOT / "data" / "eval_live.json"
    out.write_text(json.dumps({**report, "rows": rows}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
