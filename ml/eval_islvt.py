"""Unseen-signer, unseen-REGION sentence test: ISLVT (Mendeley 10.17632/98mzk82wbb.1, CC BY 4.0).

One professional Maharashtra signer; file names are the ISL gloss in signing order ("11 grand mother market go").
No training data comes from this signer or region (unless a model was trained with ISLVT mined clips: train.py
source "islvt"; use --no-islvt models only). Each front-view video is replayed through the live spotting decoder
(as ml/eval_live.py does for CSLTR sentences):
  recall     the sentence's vocabulary signs found among the detected signs (top-1 / anywhere in top-3)
  precision  detected signs that are in the sentence
Duplicate takes ("x - Copy") and side views (the "(2)" takes are mostly filmed from the side: shoulders ~5% of
the frame wide, nose 1.5 shoulder widths sideways) are skipped: the app asks people to face the camera.

    python ml/eval_islvt.py --model data/model.pt [--out data/islvt_eval.json]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "server"))
import eval_live as E  # noqa: E402
import spotting  # noqa: E402
from fetch_isign import sign_matcher  # noqa: E402
from mine_isign import side_view  # noqa: E402
from recognizer import Recognizer  # noqa: E402

KP = ROOT / "data" / "islvt" / "keypoints"
PRONOUN = {"i": "I", "me": "I", "my": "I", "you": "you", "he": "he", "him": "he", "she": "she", "her": "she",
           "we": "we", "they": "they", "it": "it"}


def targets_of(stem, match, labels):
    """Gloss words -> app signs in order (pronouns included: they are real signs in the sentence)."""
    out = []
    for w in stem.lower().split():
        if w in PRONOUN and PRONOUN[w] in labels:
            out.append(PRONOUN[w])
    words = [t for t in match.ordered(stem) if t in labels]
    return list(dict.fromkeys(words + out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "data" / "model.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rec = Recognizer(args.model, device=args.device)
    v = json.loads((ROOT / "ml" / "vocab.json").read_text())
    rec.active = set(v["include_signs"]) | set(v["cislr_signs"]) | set(v.get("common_signs", []))
    labels = set(rec.sign_labels())
    match = sign_matcher(sorted(labels))
    seen, rows = set(), []
    hit1 = hit3 = want = det = det_in = 0
    per = Counter()
    per_hit = Counter()
    for f in sorted(KP.glob("*.npz")):
        d = np.load(f)
        sig = (len(d["pose"]), round(float(d["pose"].sum()), 2))
        if sig in seen:
            continue
        seen.add(sig)
        targets = targets_of(f.stem, match, labels)
        if not targets:
            continue
        fps = float(d["fps"]) or 25.0
        frames = E.frames_of(f)
        if side_view(frames) > 0.35:
            continue
        preds = [[t["label"] for t in top] for _, _, top in spotting.spot(rec, frames, fps=fps)]
        top1 = {p[0] for p in preds}
        top3 = {x for p in preds for x in p}
        hit1 += sum(t in top1 for t in targets)
        hit3 += sum(t in top3 for t in targets)
        want += len(targets)
        det += len(preds)
        det_in += sum(p[0] in targets for p in preds)
        for t in targets:
            per[t] += 1
            per_hit[t] += t in top1
        rows.append({"video": f.stem, "targets": targets, "detected": [p[0] for p in preds]})
    rep = {"model": args.model, "videos": len(rows), "target_signs": want, "recall_top1": hit1 / max(1, want),
           "recall_top3": hit3 / max(1, want), "precision": det_in / max(1, det), "detected": det,
           "per_sign": {t: [per_hit[t], per[t]] for t in sorted(per)}, "rows": rows}
    print(f"ISLVT (Maharashtra signer, never trained on): {len(rows)} videos, {want} vocabulary signs | recall top-1 "
          f"{rep['recall_top1']:.1%} top-3 {rep['recall_top3']:.1%} | precision {rep['precision']:.0%} ({det_in}/{det})")
    for r in rows[:10]:
        print(f"   {r['video'][:40]:40s} want {r['targets']} got {r['detected']}")
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
