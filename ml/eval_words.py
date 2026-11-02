"""Per-word accuracy table, scored through the app's own Recognizer (same vocabulary + scoring as the server).

Test clips:
  include  INCLUDE recording sessions held out by train.py (stored in the checkpoint)
  cislr    one CISLR clip per sign held out by train.py (other signers / cameras)
  csltr    ISL-CSLTR word videos of signers 0,3 (only meaningful for models trained WITHOUT them: model_v10*)
Caveat: train.py picks the best epoch on the include + cislr test sets, so those two are slightly optimistic.

    python ml/eval_words.py --models data/model.pt [--vocab hospital] [--tta] [--sets include,cislr]
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from features import npz_frames
from recognizer import Recognizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ROOTS = {"include": DATA / "keypoints", "cislr": DATA / "cislr" / "keypoints"}


def vocab_set(name):
    v = json.loads((ROOT / "ml" / "vocab.json").read_text())
    if name == "hospital":
        return set(v["include_signs"]) | set(v["cislr_signs"])
    return set(v["include_signs"]) | set(v["cislr_signs"]) | set(v.get("common_signs", []))


def crops(f):
    """Test-time augmentation: the same clip with slightly different start / end points."""
    n = len(f)
    c = max(1, int(round(n * 0.08)))
    out = [f]
    if n > 12:
        out += [f[c:], f[:-c], f[c // 2: n - c // 2]]
    return out


class Ensemble:
    """Averages the per-label cosine scores of several recognisers (each model has its own embedding space)."""

    def __init__(self, paths, active):
        self.recs = [Recognizer(p) for p in paths]
        for r in self.recs:
            r.active = active

    def predict_frames(self, frames, k=3, tta=False):
        tot = None
        for r in self.recs:
            fs = crops(frames) if tta else [frames]
            embs = r.embed_frames(fs)
            sc = None
            for e in embs:
                labs, s = r.scores(e)
                sc = s if sc is None else sc + s
            sc = sc / len(embs)
            tot = sc if tot is None else tot + sc
        tot = tot / len(self.recs)
        order = np.argsort(-tot)
        return [labs[i] for i in order if labs[i] != "_none_"][:k]


def test_clips(ck_path, sets):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    out = []
    for key in (ck.get("test_include", []) if "include" in sets else []) + (ck.get("test_cislr", []) if "cislr" in sets else []):
        src, rel = key.split("/", 1)
        out.append((src, key.split("/")[1], ROOTS[src] / rel))
    if "csltr" in sets:
        kp = DATA / "csltr" / "keypoints" / "words"
        sig = json.loads((DATA / "csltr" / "signer_clusters.json").read_text())
        alias = {"hi": "Hello", "thank": "Thank you", "thanks": "Thank you", "medicines": "Medicine", "fine": "good",
                 "me": "I", "my": "I"}
        lower = {c.lower(): c for c in ck["classes"]}
        for d in sorted(kp.iterdir()):
            lab = alias.get(d.name) or lower.get(d.name)
            if not lab:
                continue
            for f in sorted(d.glob("*.npz")):
                rel = f"words\\{d.name}\\{f.name}"
                if sig.get(rel) in (0, 3):
                    out.append(("csltr", lab, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(DATA / "model.pt"), help="comma-separated = ensemble")
    ap.add_argument("--vocab", choices=["all", "hospital"], default="all")
    ap.add_argument("--sets", default="include,cislr")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--show", type=int, default=40, help="print the N worst words")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    paths = args.models.split(",")
    active = vocab_set(args.vocab)
    ens = Ensemble(paths, active)
    clips = [c for c in test_clips(paths[0], args.sets.split(",")) if c[1] in active]
    per = defaultdict(lambda: [0, 0, 0])
    per_set = defaultdict(lambda: [0, 0])
    for src, lab, p in clips:
        pred = ens.predict_frames(npz_frames(p), tta=args.tta)
        r = per[lab]
        r[0] += 1
        r[1] += pred[0] == lab
        r[2] += lab in pred
        per_set[src][0] += 1
        per_set[src][1] += pred[0] == lab
    n = sum(r[0] for r in per.values())
    t1 = sum(r[1] for r in per.values()) / n
    t3 = sum(r[2] for r in per.values()) / n
    word_acc = {l: r[1] / r[0] for l, r in per.items()}
    ge90 = sum(a >= 0.9 for a in word_acc.values())
    print(f"models {[Path(p).stem for p in paths]} vocab={args.vocab} ({len(active)} signs) tta={args.tta}")
    print(f"  clips {n}, words {len(per)} | top-1 {t1:.1%} top-3 {t3:.1%} | mean per-word {np.mean(list(word_acc.values())):.1%}"
          f" | words >= 90%: {ge90}/{len(per)}  ({', '.join(f'{s} {v[1]}/{v[0]}' for s, v in per_set.items())})")
    if args.show:
        worst = sorted(per.items(), key=lambda x: (x[1][1] / x[1][0], -x[1][0]))[:args.show]
        print("  worst: " + ", ".join(f"{l} {r[1]}/{r[0]}" for l, r in worst))
    if args.out:
        Path(args.out).write_text(json.dumps({"top1": t1, "top3": t3, "per_word": {l: {"n": r[0], "top1": r[1], "top3": r[2]}
                                                                                   for l, r in sorted(per.items())}}, indent=1))


if __name__ == "__main__":
    main()
