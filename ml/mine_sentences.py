"""Self-training from ISL-CSLTR sentence videos: we know WHICH vocabulary words each sentence contains (from its
text), not WHERE. The current model finds, in sentence order, the window that best matches each word; confident
matches become new training clips -> data/csltr/keypoints/mined/<Label>/<video>__<i>.npz (pixel units, w=h=1).

    python ml/mine_sentences.py [--model data/model.pt] [--min-rank 5]

Only the training signers' sentences are mined: signers in train.CSLTR_TEST_SIGNERS stay a clean test set.
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
import eval_live as E  # noqa: E402
import spotting  # noqa: E402
from features import clip_frames  # noqa: E402
from recognizer import Recognizer  # noqa: E402
from train import CSLTR_SIGNERS, CSLTR_TEST_SIGNERS  # noqa: E402

OUT = ROOT / "data" / "csltr" / "keypoints" / ("mined_all" if os.environ.get("CSLTR_ALL") else "mined")


def windows(frames, fps):
    active = [i for i, f in enumerate(frames) if f["lh"] is not None or f["rh"] is not None]
    if not active:
        return []
    a, b = active[0], active[-1] + 1
    sc = fps / spotting.BASE_FPS
    lens = sorted({max(6, int(round(L * sc))) for L in spotting.LENS})
    st = max(1, int(round(spotting.STRIDE * sc)))
    return [(s, s + L) for L in lens for s in range(a, max(a + 1, b - L + 1), st) if s + L <= b + 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "data" / "model.pt"))
    ap.add_argument("--min-rank", type=int, default=5, help="keep a match only if the word is in the window's top-N")
    args = ap.parse_args()
    rec = Recognizer(args.model)
    v = json.loads((ROOT / "ml" / "vocab.json").read_text())
    rec.active = set(v["include_signs"]) | set(v["cislr_signs"]) | set(v.get("common_signs", []))
    labs = rec.labels()
    li = {l: i for i, l in enumerate(labs)}
    vm = E.vocab_map(rec.sign_labels())
    signer = json.loads(CSLTR_SIGNERS.read_text())
    if OUT.exists():
        shutil.rmtree(OUT)  # regenerate from scratch with the given model
    kept, tried = {}, 0
    for d in sorted((E.KP / "sentences").iterdir()):
        targets = [vm[w] for w in re.findall(r"[a-z]+", d.name.lower()) if w in vm]
        if not targets:
            continue
        for f in sorted(d.glob("*.npz")):
            rel = str(f.relative_to(E.KP)).replace("/", "\\")
            if signer.get(rel) in CSLTR_TEST_SIGNERS or signer.get(rel) is None:
                continue
            frames = E.frames_of(f)
            fps = float(np.load(f)["fps"])
            wins = windows(frames, fps)
            if not wins:
                continue
            embs = rec.embed_frames([clip_frames(*spotting._arrays(frames[s:e]), 1, 1) for s, e in wins])
            P = []
            for emb in embs:  # full probability vector per window
                _, sc = rec.scores(emb)
                p = np.exp(16.0 * (sc - sc.max()))
                P.append(p / p.sum())
            P = np.stack(P)
            order = np.argsort(-P, axis=1)
            cursor = wins[0][0]
            for k, word in enumerate(targets):  # left-to-right: each word after the previous match starts
                tried += 1
                cand = [j for j, (s, e) in enumerate(wins) if s >= cursor]
                if not cand:
                    break
                j = max(cand, key=lambda j: P[j, li[word]])
                rank = int(np.flatnonzero(order[j] == li[word])[0])
                if rank >= args.min_rank:
                    continue
                s, e = wins[j]
                cursor = s + (e - s) // 2
                seg = frames[s:e]
                pose, lh, rh, lp, rp = spotting._arrays(seg)
                out = OUT / word / f"{d.name}__{f.stem}__{k}.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, pose=pose, lh=lh, rh=rh, lh_p=lp, rh_p=rp, fps=fps, width=1, height=1)
                kept[word] = kept.get(word, 0) + 1
    print(f"mined {sum(kept.values())} clips from {tried} (word, sentence) pairs for {len(kept)} words")
    print(sorted(kept.items(), key=lambda x: -x[1]))


if __name__ == "__main__":
    main()
