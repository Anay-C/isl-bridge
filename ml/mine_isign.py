"""Self-training on real-world iSign signing (clips from ml/fetch_isign.py, keypoints from ml/extract.py).

The English text says WHICH of our signs are probably in a clip, not WHERE (and ISL may skip or reorder them). For
each such sign the model scans the clip with sliding windows (as mine_sentences.py) and keeps the best window ONLY
if it is sure: that sign is the window's top-1 of all vocabulary signs and its cosine >= --cmin. Kept windows ->
data/isign/mined/<Label>/<uid>__<k>.npz (pixel units, w=h=1, same format as CSLTR mined clips; train only).

Negative control (printed): the same test for random signs NOT in the text. Its pass rate bounds how often a sign
that is not actually there gets mined, so  precision >= 1 - neg_rate * pos_tried / pos_kept.
Clips whose text is exactly one sign (dictionary-style entries) are kept whole if signing lasts <= 4 s ("__d").

    python ml/mine_isign.py --model data/model_v10.pt [--cmin 0.6] [--device cuda]
"""
import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
import eval_live as E  # noqa: E402
import spotting  # noqa: E402
from features import clip_frames  # noqa: E402
from fetch_isign import SKIP, sign_matcher  # noqa: E402
from mine_sentences import windows  # noqa: E402
from recognizer import Recognizer  # noqa: E402

KP = ROOT / "data" / "isign" / "keypoints"
OUT = ROOT / "data" / "isign" / "mined"


def save(path, seg, fps):
    pose, lh, rh, lp, rp = spotting._arrays(seg)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pose=pose, lh=lh, rh=rh, lh_p=lp, rh_p=rp, fps=fps, width=1, height=1)


def window_scores(rec, W, frames, wins):
    embs = np.concatenate([rec.embed_frames([clip_frames(*spotting._arrays(frames[s:e]), 1, 1)
                                             for s, e in wins[b:b + 512]]) for b in range(0, len(wins), 512)])
    return embs @ W.T  # (windows, signs) cosines


def side_view(frames):
    """Median |nose - shoulder midpoint| / shoulder width: ~0-0.2 facing the camera, > 0.35 turned sideways."""
    r = [abs(f["pose"][0, 0] - (f["pose"][11, 0] + f["pose"][12, 0]) / 2) /
         (abs(f["pose"][11, 0] - f["pose"][12, 0]) + 1e-6) for f in frames if np.abs(f["pose"][[11, 12], :2]).sum() > 0]
    return float(np.median(r)) if r else 9.0


def mine_islvt(rec, labs, li, W, match, min_rank):
    """ISLVT file names are the ISL gloss in signing order, so mine like mine_sentences.py: left to right, each sign
    in the window (after the previous match) where it is most probable, kept if it is in that window's top-N."""
    kp, out = ROOT / "data" / "islvt" / "keypoints", ROOT / "data" / "islvt" / "mined"
    if out.exists():
        shutil.rmtree(out)
    st, kept, seen = Counter(), Counter(), set()
    for f in sorted(kp.glob("*.npz")):
        targets = [t for t in match.ordered(f.stem) if t in li and t not in SKIP]
        sig = (len(np.load(f)["pose"]), round(float(np.load(f)["pose"].sum()), 2))
        if sig in seen:  # "x - Copy.MOV" files are byte copies of another take
            st["duplicate"] += 1
            continue
        seen.add(sig)
        frames = E.frames_of(f)
        fps = float(np.load(f)["fps"]) or 25.0
        sv = side_view(frames)
        if sv > 0.35:
            st["side view"] += 1
            continue
        wins = windows(frames, fps)
        if not targets or not wins:
            continue
        S = window_scores(rec, W, frames, wins)
        P = np.exp(16.0 * (S - S.max(1, keepdims=True)))
        P /= P.sum(1, keepdims=True)
        order = np.argsort(-S, axis=1)
        cursor = wins[0][0]
        for k, t in enumerate(targets):
            st["tried"] += 1
            cand = [j for j, (s, e) in enumerate(wins) if s >= cursor]
            if not cand:
                break
            j = max(cand, key=lambda j: P[j, li[t]])
            if int(np.flatnonzero(order[j] == li[t])[0]) >= min_rank:
                continue
            s, e = wins[j]
            cursor = s + (e - s) // 2
            save(out / t / f"{f.stem.strip()}__{k}.npz", frames[s:e], fps)
            kept[t] += 1
            st["kept"] += 1
    print(f"ISLVT: {dict(st)}; mined {sum(kept.values())} clips for {len(kept)} signs")
    print("per sign:", sorted(kept.items(), key=lambda x: -x[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "data" / "model_v10.pt"))
    ap.add_argument("--set", choices=("isign", "islvt"), default="isign")
    ap.add_argument("--cmin", type=float, default=0.6, help="iSign: min cosine of the mined sign in its best window")
    ap.add_argument("--min-rank", type=int, default=5, help="ISLVT: keep a match if the sign is in the window's top-N")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    rec = Recognizer(args.model, device=args.device)
    v = json.loads((ROOT / "ml" / "vocab.json").read_text())
    vocab = set(v["include_signs"]) | set(v["cislr_signs"]) | set(v.get("common_signs", []))
    labs = [l for l in rec.classes if l in vocab]
    li = {l: i for i, l in enumerate(labs)}
    W = np.stack([rec.weights[rec._cid[l]] for l in labs])
    match = sign_matcher(sorted(vocab))
    if args.set == "islvt":
        return mine_islvt(rec, labs, li, W, match, args.min_rank)
    plan = json.loads((ROOT / "data" / "isign" / "plan.json").read_text())
    if OUT.exists():
        shutil.rmtree(OUT)  # regenerate from scratch with the given model
    rng = random.Random(0)
    st, kept, neg_hits = Counter(), Counter(), Counter()
    cos_kept = []
    for n, (uid, p) in enumerate(plan.items(), 1):
        f = KP / f"{uid}.npz"
        if not f.exists():
            st["no keypoints"] += 1
            continue
        frames = E.frames_of(f)
        fps = float(np.load(f)["fps"]) or 25.0
        st["clips"] += 1
        pose_ok = np.mean([np.abs(fr["pose"][[11, 12], :2]).sum() > 0 for fr in frames])
        hands = np.mean([fr["lh"] is not None or fr["rh"] is not None for fr in frames])
        if pose_ok < 0.8 or hands < 0.2:  # shoulders cut off by the crop, or almost no hands found
            st["bad framing"] += 1
            continue
        targets = [t for t in p["signs"] if t in li and t not in SKIP]
        words = re.findall(r"[a-z]+", p["text"].lower())
        # dictionary-style entry: the whole text is one sign ("Doctor", "thank you")
        if len(targets) == 1 and len(words) <= 3 and match(p["text"]) == {targets[0]}:
            act = [i for i, fr in enumerate(frames) if fr["lh"] is not None or fr["rh"] is not None]
            if act and (act[-1] - act[0] + 1) / fps <= 4.0:
                save(OUT / targets[0] / f"{uid}__d.npz", frames[act[0]:act[-1] + 1], fps)
                kept[targets[0]] += 1
                st["direct"] += 1
                continue
        wins = windows(frames, fps)
        if not targets or not wins:
            continue
        S = window_scores(rec, W, frames, wins)
        top = S.argmax(1)
        for k, t in enumerate(targets):
            j = int(np.argmax(S[:, li[t]]))
            st["pos tried"] += 1
            if top[j] == li[t] and S[j, li[t]] >= args.cmin:
                s, e = wins[j]
                save(OUT / t / f"{uid}__{k}.npz", frames[s:e], fps)
                kept[t] += 1
                st["pos kept"] += 1
                cos_kept.append(float(S[j, li[t]]))
        others = [l for l in labs if l not in p["signs"] and l not in SKIP]
        for t in rng.sample(others, min(len(others), len(targets))):
            j = int(np.argmax(S[:, li[t]]))
            st["neg tried"] += 1
            if top[j] == li[t] and S[j, li[t]] >= args.cmin:
                st["neg kept"] += 1
                neg_hits[t] += 1
        if n % 250 == 0:
            print(f"[{n}/{len(plan)}] {dict(st)}", flush=True)
    pos_rate = st["pos kept"] / max(1, st["pos tried"])
    neg_rate = st["neg kept"] / max(1, st["neg tried"])
    prec = 1 - neg_rate * st["pos tried"] / max(1, st["pos kept"])
    print(f"\n{dict(st)}")
    print(f"mined {sum(kept.values())} clips for {len(kept)} signs (direct {st['direct']}); "
          f"text signs kept {pos_rate:.1%}, random signs kept {neg_rate:.1%} -> est. precision >= {prec:.0%}")
    if cos_kept:
        print(f"cosine of mined windows: median {np.median(cos_kept):.2f}")
    print("per sign:", sorted(kept.items(), key=lambda x: -x[1]))
    print("negative-control hits (attractor signs):", neg_hits.most_common(10))
    (ROOT / "data" / "isign" / "mine_report.json").write_text(json.dumps(
        {"model": args.model, "cmin": args.cmin, "stats": dict(st), "per_sign": dict(kept),
         "pos_rate": pos_rate, "neg_rate": neg_rate, "est_precision": prec, "neg_hits": dict(neg_hits)}, indent=1))


if __name__ == "__main__":
    main()
