"""Exemplar bank: the embedding of every training clip, so the recogniser can match a sign against the closest
real recording of it (one variant) instead of only the class average (all variants blended).

    python ml/exemplars.py --model data/model.pt --out data/exemplars.npz          # live model (CSLTR_ALL=1 split)
    python ml/exemplars.py --model data/model_v10.pt --out data/exemplars_v10.npz   # held-out model, for evaluation

Uses train.py's exact data loading + split (seed 0 by default), so held-out test clips are never in the bank.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import train
from recognizer import Recognizer

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--no-mined", action="store_true", help="leave clips mined from CSLTR sentences out of the bank")
    args = ap.parse_args()
    keys, frames, labels = train.load_frames()
    v = json.loads((ROOT / "ml" / "vocab.json").read_text())
    vocab = set(v["include_signs"]) | set(v["cislr_signs"]) | set(v.get("common_signs", []))
    keep = [i for i, l in enumerate(labels) if l in vocab]
    keys, frames, labels = [keys[i] for i in keep], [frames[i] for i in keep], [labels[i] for i in keep]
    tr, _, _ = train.split(keys, labels, 0.2, args.split_seed, frames)
    if args.no_mined:
        tr = [i for i in tr if not keys[i].startswith("mined/")]
    rec = Recognizer(args.model)
    tr = [i for i in tr if labels[i] in rec._cid]
    embs = []
    for s in range(0, len(tr), 256):
        embs.append(rec.embed_frames([frames[i] for i in tr[s:s + 256]]))
    E = np.concatenate(embs).astype(np.float32)
    L = np.array([labels[i] for i in tr])
    K = np.array([keys[i] for i in tr])
    np.savez_compressed(args.out, emb=E, label=L, key=K, model=os.path.basename(args.model))
    print(f"{len(E)} exemplars of {len(set(L))} signs -> {args.out} "
          f"(csltr {sum(k.startswith('csltr/') for k in K)}, mined {sum(k.startswith('mined/') for k in K)})")


if __name__ == "__main__":
    main()
