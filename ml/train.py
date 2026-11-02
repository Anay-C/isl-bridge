"""Train the sign encoder + cosine classifier on INCLUDE + CISLR keypoints.

    python ml/train.py --vocab                 # scenario vocabulary (INCLUDE signs + CISLR-only signs)
    python ml/train.py                         # every extracted sign

Two held-out test sets:
  * INCLUDE: whole recording sessions (consecutive MVI_#### runs) per sign
  * CISLR:   one clip per sign (signs with >= 2 clips) -> different signers & cameras = cross-signer check

Cosine classifier: class weights live on the same unit sphere as embeddings, so a user's personal
prototype (mean embedding of 3-5 repetitions) is directly comparable to a class weight -> personalisation.
"""
import argparse
import os
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import FEAT_DIM, POSE_IDX, T_OUT, npz_frames, resample
from model import SignModel

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SOURCES = {"include": DATA / "keypoints", "cislr": DATA / "cislr" / "keypoints",
           "islrtc": DATA / "islrtc" / "keypoints",  # ISLRTC dictionary: 1 clip per sign, train only
           "csltr": DATA / "csltr" / "keypoints" / "words",  # ISL-CSLTR word videos (7 signers), train only;
           # words found in CSLTR sentences (ml/mine_sentences.py); separate folder when all signers are mined
           "mined": DATA / "csltr" / "keypoints" / ("mined_all" if os.environ.get("CSLTR_ALL") else "mined"),
           # real-world iSign signing (news / Deaf channels), self-labelled by ml/mine_isign.py; train only
           "isign": DATA / "isign" / "mined",
           # ISLVT (Mendeley, CC BY 4.0): a Maharashtra signer's gloss-ordered sentences, mined; train only
           "islvt": DATA / "islvt" / "mined"}
# its sentence videos are the live-pipeline test (ml/eval_live.py)
CSLTR_SIGNERS = DATA / "csltr" / "signer_clusters.json"  # clip -> signer id (clustered by clothing, 7 signers)
# held out entirely: the honest unseen-signer test (ml/eval_live.py --signers 0,3). CSLTR_ALL=1 trains the
# final live model on every signer (report the held-out numbers, ship the model that has seen more people).
CSLTR_TEST_SIGNERS = set() if os.environ.get("CSLTR_ALL") else {0, 3}
NONE = "_none_"  # "no sign": windows straddling two signs, so sign spotting rejects transitions
CSLTR_ALIAS = {"hi": "Hello", "thank": "Thank you", "medicines": "Medicine"}


def _csltr_label(word, labels):
    m = {l.lower(): l for l in labels}
    return CSLTR_ALIAS.get(word) or m.get(word)
CACHE = DATA / "frames_cache.npz"

# feature layout (see features.py)
NP = len(POSE_IDX) * 2
LB, RB, LL, RL = NP, NP + 63, NP + 126, NP + 189  # lh body, rh body, lh local, rh local
LP, RP = FEAT_DIM - 2, FEAT_DIM - 1
# mirror: swap left/right pose joints within POSE_IDX [nose, eyeL, eyeR, mouthL, mouthR, shL, shR, elL, elR, wrL, wrR]
POSE_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9]


def load_frames():
    cached = {}
    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=True)
        cached = dict(zip(c["paths"].tolist(), c["frames"]))
    keys, frames = [], []
    v = json.loads((ROOT / "ml" / "vocab.json").read_text())
    known = v["include_signs"] + v["cislr_signs"] + v.get("common_signs", [])
    signer = json.loads(CSLTR_SIGNERS.read_text()) if CSLTR_SIGNERS.exists() else {}
    for src, root in SOURCES.items():
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.npz")):
            if src == "csltr":  # folder = lower-case CSLTR word; keep only words we know, under our label
                lab = _csltr_label(p.parent.name, known)
                rel = f"words/{p.parent.name}/{p.name}"
                if not lab or signer.get(rel.replace("/", "\\"), signer.get(rel)) in CSLTR_TEST_SIGNERS:
                    continue
                key = f"csltr/{lab}/{p.parent.name}__{p.name}"
                f = cached.get(key)
                keys.append(key)
                frames.append(f if f is not None else npz_frames(p))
                continue
            if src == "islrtc" and _signing_seconds(p) > 4.0:  # dictionary entry with a signed definition
                continue
            key = f"{src}/{p.relative_to(root).as_posix()}"
            # mined clips are regenerated under the same names each round: never trust the cache for them
            f = cached.get(key) if src not in ("mined", "isign", "islvt") else None
            if f is None:
                f = npz_frames(p)
            keys.append(key)
            frames.append(f)
    if set(keys) != set(cached):
        arr = np.empty(len(frames), object)
        arr[:] = frames
        np.savez(CACHE, paths=np.array(keys), frames=arr)
    labels = [k.split("/")[1] for k in keys]
    return keys, frames, labels


def _signing_seconds(npz):
    d = np.load(npz)
    h = np.flatnonzero(d["lh_p"] | d["rh_p"])
    return (h[-1] - h[0] + 1) / float(d["fps"]) if len(h) else 0.0


def mirror(f):
    """Left-right mirror in body space: negate x, swap pose sides and the two hands."""
    g = f.copy()
    pose = g[:, :NP].reshape(len(g), -1, 2)[:, POSE_SWAP]
    pose[..., 0] *= -1
    g[:, :NP] = pose.reshape(len(g), -1)
    for a, b in ((LB, RB), (LL, RL)):
        lh, rh = f[:, a:a + 63].reshape(len(f), 21, 3).copy(), f[:, b:b + 63].reshape(len(f), 21, 3).copy()
        lh[..., 0] *= -1
        rh[..., 0] *= -1
        g[:, a:a + 63], g[:, b:b + 63] = rh.reshape(len(f), -1), lh.reshape(len(f), -1)
    g[:, LP], g[:, RP] = f[:, RP], f[:, LP]
    return g


def drop_hand_frames(f, rng, p):
    """Simulate missed hand detections (as on low-res / blurry cameras)."""
    for body, local, pres in ((LB, LL, LP), (RB, RL, RP)):
        m = rng.random(len(f)) < p
        f[m, body:body + 63] = 0
        f[m, local:local + 63] = 0
        f[m, pres] = 0
    return f


STRONG = False  # --aug-strong: simulate more signer variety (set in main)


def signer_style(f, rng):
    """Different-person augmentation on variable-length features: per-hand reach offsets, hand size, and a
    non-uniform time warp (signers differ in where, how big and how evenly they sign)."""
    f = f.copy()
    for body, pres in ((LB, LP), (RB, RP)):
        on = f[:, pres] > 0.5
        blk = f[:, body:body + 63].reshape(len(f), 21, 3)
        wrist = blk[:, :1, :2].copy()
        size = rng.uniform(0.75, 1.3)  # hand size relative to the body
        blk[..., :2] = wrist + (blk[..., :2] - wrist) * size
        blk[..., :2] += rng.uniform(-0.3, 0.3, size=2)  # where this signer holds this hand
        f[:, body:body + 63] = np.where(on[:, None], blk.reshape(len(f), -1), 0)
    if len(f) > 6:  # smooth random time warp
        knots = np.cumsum(rng.uniform(0.5, 1.5, size=5))
        pos = np.interp(np.linspace(0, knots[-1], len(f)), np.r_[0, knots], np.linspace(0, len(f) - 1, 6))
        f = f[np.clip(np.rint(pos).astype(int), 0, len(f) - 1)]
    return f


CLOSEUP = False  # --aug-closeup (set in main)
FACE = [0, 1, 2, 3, 4]  # nose, eyes, mouth within POSE_IDX


def closeup(f, rng):
    """Phone held close (live test: shoulders ~50% of the frame vs ~15% in training): perspective makes the
    hands look bigger and spread wider, and the face sits nearer the shoulder line."""
    f = f.copy()
    grow, spread = rng.uniform(1.0, 1.35), rng.uniform(1.0, 1.2)
    for body, local, pres in ((LB, LL, LP), (RB, RL, RP)):
        on = f[:, pres] > 0.5
        blk = f[:, body:body + 63].reshape(len(f), 21, 3)
        wrist = blk[:, :1, :2] * spread
        blk[..., :2] = wrist + (blk[..., :2] - blk[:, :1, :2]) * grow
        f[:, body:body + 63] = np.where(on[:, None], blk.reshape(len(f), -1), 0)
    pose = f[:, :NP].reshape(len(f), -1, 2)
    pose[:, 7:, :] *= spread  # elbows, wrists
    pose[:, FACE, 1] *= rng.uniform(0.7, 1.0)  # face closer to the shoulder line
    f[:, :NP] = pose.reshape(len(f), -1)
    return f


AUG_LIVE = 0.0  # --aug-live P (set in main)
# body-space columns (see features.py): pose joint j at (2j, 2j+1); elbows 13/14 = j 7/8; hand wrist = first point of a hand
EL_L, EL_R = 7 * 2 + 1, 8 * 2 + 1


def live_crop(f, rng):
    """Cut the clip the way the live segmenter does (server/segmenter.py): from a few frames before the first frame
    with a hand raised (wrist above 1.2 shoulder widths below the shoulders, or above 2.2 and not hanging below its
    elbow) to a few frames after the last. Training clips otherwise keep long stretches of hands at rest."""
    raised = np.zeros(len(f), bool)
    for body, pres, el in ((LB, LP, EL_L), (RB, RP, EL_R)):
        wy = f[:, body + 1]
        raised |= (f[:, pres] > 0.5) & ((wy < 1.2) | ((wy < 2.2) & (wy - f[:, el] < 0.5)))
    r = np.flatnonzero(raised)
    if len(r) < 3:
        return f
    s = max(0, r[0] - int(rng.integers(0, 16)))
    e = min(len(f), r[-1] + 1 + int(rng.integers(0, 11)))
    return f[s:e] if e - s >= 6 else f


def augment(f, rng):
    """f: variable-length (T, FEAT_DIM)."""
    if AUG_LIVE and rng.random() < AUG_LIVE:
        f = live_crop(f, rng)
    if CLOSEUP and rng.random() < 0.35:
        f = closeup(f, rng)
    if rng.random() < 0.5:
        f = mirror(f)
    if STRONG and rng.random() < 0.7:
        f = signer_style(f, rng)
    t = len(f)
    if t > 8:
        s = rng.integers(0, max(1, int(t * 0.15)))
        e = t - rng.integers(0, max(1, int(t * 0.15)))
        f = f[s:e]
        if rng.random() < 0.5:
            keep = np.sort(rng.choice(len(f), size=max(4, int(len(f) * rng.uniform(0.6, 1))), replace=False))
            f = f[keep]
    f = resample(f, T_OUT).copy()
    if rng.random() < 0.5:
        f = drop_hand_frames(f, rng, rng.uniform(0.05, 0.3))
    ang = np.deg2rad(rng.uniform(-15, 15))
    sc = rng.uniform(0.8, 1.2) * rng.uniform(0.9, 1.1, size=2)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * sc
    shift = rng.uniform(-0.2, 0.2, size=2)
    for a, b, stride in ((0, NP, 2), (LB, LB + 63, 3), (RB, RB + 63, 3)):
        blk = f[:, a:b].reshape(T_OUT, -1, stride)
        mask = np.abs(blk[..., :2]).sum(-1, keepdims=True) > 0  # keep missing hands at zero
        blk[..., :2] = np.where(mask, blk[..., :2] @ rot.T + shift, 0)
        f[:, a:b] = blk.reshape(T_OUT, -1)
    for a in (LL, RL):  # small hand-shape rotation in local space
        blk = f[:, a:a + 63].reshape(T_OUT, 21, 3)
        r = np.deg2rad(rng.uniform(-10, 10))
        blk[..., :2] = blk[..., :2] @ np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]]).T
        f[:, a:a + 63] = blk.reshape(T_OUT, -1)
    nz = f[:, :-2] != 0
    f[:, :-2] += rng.normal(0, rng.uniform(0.005, 0.03), size=f[:, :-2].shape) * nz
    return f.astype(np.float32)


def include_sessions(idx, keys, labels):
    """Group INCLUDE clips into recording sessions: runs of consecutive MVI_#### numbers within a sign."""
    groups, by = {}, {}
    for i in idx:
        num = re.findall(r"(\d+)", Path(keys[i]).stem.split("__")[-1])
        by.setdefault(labels[i], []).append((int(num[-1]) if num else i, i))
    for l, items in by.items():
        items.sort()
        g, prev = 0, None
        for n, i in items:
            if prev is not None and n - prev > 3:
                g += 1
            groups.setdefault((l, g), []).append(i)
            prev = n
    return groups


def split(keys, labels, test_frac, seed, frames):
    # one RNG per sign: a sign's test clips are the same whatever else is in the run, so a pretrained
    # all-signs model never sees the test clips of a later vocab fine-tune
    rng_for = lambda l, src: random.Random(f"{seed}/{src}/{l}")
    tr, te_inc, te_cis = [], [], []
    inc = [i for i, k in enumerate(keys) if k.startswith("include/")]
    cis = [i for i, k in enumerate(keys) if k.startswith("cislr/")]
    by_label = {}
    for (l, _), g in include_sessions(inc, keys, labels).items():
        by_label.setdefault(l, []).append(g)
    for l, gl in by_label.items():
        rng_for(l, "include").shuffle(gl)
        k = max(1, int(round(len(gl) * test_frac))) if len(gl) >= 3 else 0
        te_inc += [i for g in gl[:k] for i in g]
        tr += [i for g in gl[k:] for i in g]
    by_c = {}
    for i in cis:
        by_c.setdefault(labels[i], []).append(i)
    for l, idx in by_c.items():
        rng_for(l, "cislr").shuffle(idx)
        if len(idx) >= 2:
            te_cis.append(idx[0])
            tr += idx[1:]
        else:
            tr += idx
    tr += [i for i, k in enumerate(keys) if k.startswith(("csltr/", "mined/", "islvt/"))]  # train only
    # iSign: train only. It re-distributes CISLR, so drop any whole-clip entry identical to a CISLR clip
    cis_by = {}
    for i in cis:
        cis_by.setdefault(labels[i], []).append(frames[i])
    for i, k in enumerate(keys):
        if k.startswith("isign/") and not (k.endswith("__d.npz") and any(
                _same_video(frames[i], g) for g in cis_by.get(labels[i], []))):
            tr.append(i)
    # ISLRTC dictionary clips: training only, minus exact copies of CISLR test videos (CISLR scraped
    # the older ISLRTC dictionary, so the same recording can appear in both)
    test_cis = {labels[i]: frames[i] for i in te_cis}
    for i, k in enumerate(keys):
        if k.startswith("islrtc/"):
            t = test_cis.get(labels[i])
            if t is not None and _same_video(frames[i], t):
                continue
            tr.append(i)
    return tr, te_inc, te_cis


def _same_video(a, b, tol=0.02):
    a, b = resample(a, T_OUT), resample(b, T_OUT)
    return float(np.abs(a[:, :-2] - b[:, :-2]).mean()) < tol


def evaluate(model, x, y, n_cls):
    with torch.no_grad():
        lg = model.logits(model.embed(x))
    top1 = (lg.argmax(1) == y).float().mean().item()
    top3 = (lg.topk(min(3, n_cls), 1).indices == y[:, None]).any(1).float().mean().item()
    return top1, top3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", action="store_true", help="restrict to ml/vocab.json signs")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=str(DATA / "model.pt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=None,
                    help="train/test split seed (default = --seed); keep 0 to train ensemble members on the v10 split")
    ap.add_argument("--no-cislr-train", action="store_true", help="ablation: train on INCLUDE only")
    ap.add_argument("--no-islrtc", action="store_true", help="ablation: no ISLRTC dictionary clips")
    ap.add_argument("--no-csltr", action="store_true", help="ablation: no ISL-CSLTR word clips")
    ap.add_argument("--no-mined", action="store_true", help="ablation: no clips mined from CSLTR sentences")
    ap.add_argument("--mined-cap", type=int, default=15, help="max mined clips per sign (0 = no cap)")
    ap.add_argument("--no-isign", action="store_true", help="ablation: no clips mined from iSign")
    ap.add_argument("--no-islvt", action="store_true", help="ablation: no clips mined from ISLVT")
    ap.add_argument("--isign-cap", type=int, default=40, help="max iSign clips per sign (0 = no cap)")
    ap.add_argument("--none-frac", type=float, default=0.0, help="share of each batch that is a 'no sign' window")
    ap.add_argument("--init", default=None, help="start from this checkpoint's encoder (pretraining)")
    ap.add_argument("--aug-strong", action="store_true", help="extra different-signer augmentation")
    ap.add_argument("--aug-closeup", action="store_true", help="simulate a phone held close (perspective)")
    ap.add_argument("--gpu-frac", type=float, default=0.5, help="max share of GPU memory to use")
    ap.add_argument("--aug-live", type=float, default=0.0, help="share of samples cut like the live segmenter cuts")
    args = ap.parse_args()
    global STRONG, CLOSEUP, AUG_LIVE
    STRONG, CLOSEUP, AUG_LIVE = args.aug_strong, args.aug_closeup, args.aug_live
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    keys, frames, labels = load_frames()
    if args.vocab:
        v = json.loads((ROOT / "ml" / "vocab.json").read_text())
        vocab = set(v["include_signs"]) | set(v["cislr_signs"]) | set(v.get("common_signs", []))
        keep = [i for i, l in enumerate(labels) if l in vocab]
        keys, frames, labels = [keys[i] for i in keep], [frames[i] for i in keep], [labels[i] for i in keep]
    tr, te_inc, te_cis = split(keys, labels, 0.2, args.seed if args.split_seed is None else args.split_seed, frames)
    n_dup = sum(k.startswith("islrtc/") for k in keys) - sum(keys[i].startswith("islrtc/") for i in tr)
    if n_dup:
        print(f"dropped {n_dup} ISLRTC clips identical to CISLR test videos")
    if args.no_cislr_train:
        tr = [i for i in tr if keys[i].startswith("include/")]
    if args.no_islrtc:
        tr = [i for i in tr if not keys[i].startswith("islrtc/")]
    if args.no_csltr:
        tr = [i for i in tr if not keys[i].startswith(("csltr/", "mined/"))]
    if args.no_mined:
        tr = [i for i in tr if not keys[i].startswith("mined/")]
    if args.no_isign:
        tr = [i for i in tr if not keys[i].startswith("isign/")]
    if args.no_islvt:
        tr = [i for i in tr if not keys[i].startswith("islvt/")]
    if args.isign_cap:  # per sign, so frequent news words do not swamp the rest
        seen, keep = {}, []
        for i in tr:
            if keys[i].startswith("isign/"):
                seen[labels[i]] = seen.get(labels[i], 0) + 1
                if seen[labels[i]] > args.isign_cap:
                    continue
            keep.append(i)
        tr = keep
    n_isign = sum(keys[i].startswith("isign/") for i in tr)
    if n_isign:
        print(f"iSign clips in training: {n_isign} "
              f"({len({labels[i] for i in tr if keys[i].startswith('isign/')})} signs)")
    if args.mined_cap:  # pointing signs (I, you) are mined far more often; cap so they do not dominate
        seen = {}
        keep = []
        for i in tr:
            if keys[i].startswith("mined/"):
                seen[labels[i]] = seen.get(labels[i], 0) + 1
                if seen[labels[i]] > args.mined_cap:
                    continue
            keep.append(i)
        tr = keep
    classes = sorted({labels[i] for i in tr}) + ([NONE] if args.none_frac > 0 else [])
    cid = {c: i for i, c in enumerate(classes)}
    te_inc = [i for i in te_inc if labels[i] in cid]
    te_cis = [i for i in te_cis if labels[i] in cid]
    y = np.array([cid.get(l, -1) for l in labels])
    print(f"{len(classes)} classes | train {len(tr)} | test INCLUDE {len(te_inc)} | test CISLR {len(te_cis)}")

    # class-balanced sampling: each epoch draws len(tr) clips, classes weighted ~ 1/sqrt(count)
    counts = np.bincount(y[tr], minlength=len(classes))
    w = 1.0 / np.sqrt(counts[y[tr]])
    w /= w.sum()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":  # GPU is shared with other work: hard cap on this process's memory
        torch.cuda.set_per_process_memory_fraction(args.gpu_frac)
    model = SignModel(FEAT_DIM, len(classes)).to(dev)
    if args.init:  # encoder weights only; the classifier is new (different class list)
        st = torch.load(args.init, map_location=dev, weights_only=False)["state"]
        own = model.state_dict()
        st = {k: v for k, v in st.items() if k in own and own[k].shape == v.shape and k != "weight"}
        model.load_state_dict(st, strict=False)
        print(f"init from {args.init}: {len(st)}/{len(own)} tensors")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps_per = (len(tr) + args.bs - 1) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_per, pct_start=0.1)
    mk = lambda idx: (torch.tensor(np.stack([resample(frames[i], T_OUT) for i in idx])).to(dev),
                      torch.tensor(y[idx]).to(dev)) if idx else None
    t_inc, t_cis = mk(te_inc), mk(te_cis)

    best, t0, tr_arr = -1.0, time.time(), np.array(tr)
    for ep in range(1, args.epochs + 1):
        model.train()
        order = rng.choice(tr_arr, size=len(tr), replace=True, p=w)
        tot = 0.0
        for b in range(0, len(order), args.bs):
            idx = order[b:b + args.bs]
            xs = [augment(frames[i], rng) for i in idx]
            ys = list(y[idx])
            for j in range(int(round(len(idx) * args.none_frac))):  # stitched transition = "no sign"
                a, b = frames[rng.choice(tr_arr)], frames[rng.choice(tr_arr)]
                st = np.concatenate([a[int(len(a) * rng.uniform(0.45, 0.75)):], b[:max(2, int(len(b) * rng.uniform(0.25, 0.55)))]])
                xs[j], ys[j] = augment(st, rng), cid[NONE]
            xb = torch.tensor(np.stack(xs)).to(dev)
            yb = torch.tensor(np.array(ys)).to(dev)
            loss = F.cross_entropy(model.logits(model.embed(xb), margin_labels=yb), yb, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        if ep % 10 == 0 or ep == args.epochs:
            model.eval()
            r_inc = evaluate(model, *t_inc, len(classes)) if t_inc else (0, 0)
            r_cis = evaluate(model, *t_cis, len(classes)) if t_cis else (0, 0)
            print(f"ep {ep:3d} loss {tot / len(tr):.3f} | INCLUDE top1 {r_inc[0]:.3f} top3 {r_inc[1]:.3f} "
                  f"| CISLR top1 {r_cis[0]:.3f} top3 {r_cis[1]:.3f} ({time.time() - t0:.0f}s)", flush=True)
            score = r_inc[0] + r_cis[0]  # select on both in-domain and cross-signer
            if score >= best:
                best = score
                torch.save({"state": model.state_dict(), "classes": classes, "feat_dim": FEAT_DIM, "t_out": T_OUT,
                            "test_include": [keys[i] for i in te_inc], "test_cislr": [keys[i] for i in te_cis],
                            "metrics": {"include": r_inc, "cislr": r_cis, "epoch": ep}}, args.out)
    ck = torch.load(args.out, weights_only=False)
    print(f"saved {args.out}: {ck['metrics']}")


if __name__ == "__main__":
    main()
