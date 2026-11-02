"""Cross-dataset evaluation on CISLR (different signers, different cameras than INCLUDE).

 1. Cross-signer: CISLR clips of signs the model was trained on (INCLUDE), generic model, no personalisation.
 2. One-shot new signs: signs NOT in INCLUDE. For each sign with >= 2 clips, enrol 1 clip as its prototype,
    test on the others, competing against every trained sign + every other enrolled new sign.

    python ml/eval_cislr.py [--model data/model.pt]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from features import npz_frames
from recognizer import Recognizer

ROOT = Path(__file__).resolve().parent.parent
KP = ROOT / "data" / "cislr" / "keypoints"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "data" / "model.pt"))
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()
    rec = Recognizer(args.model)
    vocab = json.loads((ROOT / "ml" / "vocab.json").read_text())
    rec.active = set(vocab["include_signs"])

    labels, paths = [], []
    for d in sorted(p for p in KP.iterdir() if p.is_dir()):
        for f in sorted(d.glob("*.npz")):
            labels.append(d.name)
            paths.append(f)
    embs = rec.embed_frames([npz_frames(p) for p in paths])
    labels = np.array(labels)
    report = {}

    # 1. cross-signer on trained signs
    known = np.isin(labels, rec.classes)
    top1 = top3 = 0
    for i in np.flatnonzero(known):
        pred = [p["label"] for p in rec.predict(embs[i], k=3, personalise=False)]
        top1 += pred[0] == labels[i]
        top3 += labels[i] in pred
    n = int(known.sum())
    report["cross_signer"] = {"n": n, "signs": len(set(labels[known])), "top1": top1 / n, "top3": top3 / n}
    print(f"1. cross-signer (CISLR clips of {len(set(labels[known]))} trained signs, n={n}): "
          f"top1 {top1 / n:.1%}  top3 {top3 / n:.1%}   (chance {1 / len(rec.labels()):.1%})")

    # 2. one-shot new signs
    new = sorted(set(labels[~known]))
    multi = [s for s in new if (labels == s).sum() >= 2]
    rng = np.random.default_rng(0)
    top1 = top3 = n = 0
    for _ in range(args.trials):
        rec.clear_samples()
        test = []
        for s in new:
            idx = np.flatnonzero(labels == s)
            shot = rng.choice(idx)
            rec.add_sample(s, embs[shot])
            if s in multi:
                test += [i for i in idx if i != shot]
        for i in test:
            pred = [p["label"] for p in rec.predict(embs[i], k=3)]
            top1 += pred[0] == labels[i]
            top3 += labels[i] in pred
            n += 1
    n_cls = len(rec.labels())
    report["one_shot_new"] = {"n": n, "new_signs": len(new), "tested_signs": len(multi), "candidates": n_cls,
                              "top1": top1 / n, "top3": top3 / n}
    print(f"2. one-shot new signs ({len(new)} enrolled from 1 clip, {len(multi)} testable, "
          f"{n_cls} candidates): top1 {top1 / n:.1%}  top3 {top3 / n:.1%}   (chance {1 / n_cls:.1%})")
    out = ROOT / "data" / "eval_cislr.json"
    out.write_text(json.dumps(report, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
