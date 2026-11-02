"""Before vs after personalisation on a NEW signer (never seen in training).

Uses data/recordings/<user>/<INCLUDE sign>/*.npz (Record mode, 5 takes per sign).
For k = 0..K personal shots: repeat over random draws -> use k takes per sign as personal samples,
test on the remaining takes. Restricted to the signs the user recorded (the scenario vocabulary).

    python ml/eval_personal.py --user anay
"""
import argparse
import json
from pathlib import Path

import numpy as np

from features import npz_frames
from recognizer import Recognizer

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="anay")
    ap.add_argument("--model", default=str(ROOT / "data" / "model.pt"))
    ap.add_argument("--max-k", type=int, default=3)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--closed", action="store_true",
                    help="only choose among the recorded signs (easier; default = the full app vocabulary)")
    args = ap.parse_args()

    rec = Recognizer(args.model)
    d = ROOT / "data" / "recordings" / args.user
    labels, frames = [], []
    for sd in sorted(p for p in d.iterdir() if p.is_dir() and p.name in rec.classes):
        for f in sorted(sd.glob("*.npz")):
            labels.append(sd.name)
            frames.append(npz_frames(f))
    if not labels:
        raise SystemExit(f"no recordings of trained signs in {d}")
    embs = rec.embed_frames(frames)
    labels = np.array(labels)
    signs = sorted(set(labels))
    v = json.loads((ROOT / "ml" / "vocab.json").read_text())
    rec.active = set(signs) if args.closed else (
        set(v["include_signs"]) | set(v["cislr_signs"]) | set(v.get("common_signs", [])))
    print(f"choosing among {len(rec.sign_labels())} signs")
    print(f"{args.user}: {len(labels)} takes of {len(signs)} signs")

    rng = np.random.default_rng(0)
    results = {}
    for k in range(args.max_k + 1):
        top1 = top3 = n = 0
        for _ in range(args.trials if k else 1):
            rec.clear_samples()
            test = np.ones(len(labels), bool)
            for s in signs:
                idx = np.flatnonzero(labels == s)
                if len(idx) <= k:
                    continue
                for i in rng.choice(idx, k, replace=False):
                    rec.add_sample(s, embs[i])
                    test[i] = False
            for i in np.flatnonzero(test):
                pred = [p["label"] for p in rec.predict(embs[i], k=3)]
                top1 += pred[0] == labels[i]
                top3 += labels[i] in pred
                n += 1
        results[k] = {"top1": top1 / n, "top3": top3 / n, "n_test": n}
        tag = "before (generic model)" if k == 0 else f"after {k} personal take{'s' * (k > 1)} / sign"
        print(f"  {tag:32s} top1 {top1 / n:6.1%}   top3 {top3 / n:6.1%}")
    out = ROOT / "data" / f"eval_personal_{args.user}.json"
    out.write_text(json.dumps({"user": args.user, "signs": signs, "vocab_size": len(rec.sign_labels()),
                               "results": results}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
