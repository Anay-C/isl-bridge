"""Ghost Coach: reference trajectories for the learning page + scoring a learner's attempt with tips.

The reference is the exact recording that data/clips/<label>.mp4 was cut from (data/clips/sources.json),
trimmed the same way (hand frames +- 0.3 s), so the ghost skeleton stays in sync with the clip.
Everything is in body space (centred on the shoulder midpoint, scaled by shoulder width), so the ghost
can be drawn onto any learner's body and attempts can be compared regardless of camera distance.
"""
import json
from pathlib import Path

import numpy as np

from features import POSE_IDX, T_OUT, clip_frames, npz_frames, resample

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KP = {"islrtc": DATA / "islrtc" / "keypoints", "include": DATA / "keypoints", "cislr": DATA / "cislr" / "keypoints"}
ARM = [11, 12, 13, 14, 15, 16]  # shoulders, elbows, wrists

# feature layout (features.py): pose (x,y) for POSE_IDX, then lh body, rh body, lh local, rh local (21x3), presence
NP = len(POSE_IDX) * 2
LB, RB, LL, RL = NP, NP + 63, NP + 126, NP + 189
LP, RP = -2, -1
NOSE_Y = 1  # y of pose joint 0 (nose) within the pose block

_sources = None


def _source(label):
    global _sources
    if _sources is None:
        _sources = {}
        # sources.json: the recording each clip was cut from ("exists" = unknown); ghost_sources.json
        # (ml/pick_ghost_refs.py) fills the unknown ones with a clean, typical recording
        for name in ("sources.json", "ghost_sources.json"):
            f = DATA / "clips" / name
            if f.exists():
                _sources.update({k: v for k, v in json.loads(f.read_text()).items() if ":" in v})
    src = _sources.get(label)
    if src and ":" in src:
        name, file = src.split(":", 1)
        p = KP[name] / label / (Path(file).stem + ".npz")
        if p.exists():
            return p
    for root in KP.values():  # no clip: any recording of the sign
        fs = sorted((root / label).glob("*.npz")) if (root / label).is_dir() else []
        if fs:
            return fs[0]
    return None


def reference(label):
    """Ghost trajectory for the web page: per frame arm joints + both hands (x,y) in body space, or None."""
    p = _source(label)
    if p is None:
        return None
    d = np.load(p)
    w, h, fps = float(d["width"]), float(d["height"]), float(d["fps"])
    hands = np.flatnonzero(d["lh_p"] | d["rh_p"])
    s, e = (hands[0], hands[-1]) if len(hands) else (0, len(d["pose"]) - 1)
    s, e = max(0, s - int(0.3 * fps)), min(len(d["pose"]) - 1, e + int(0.3 * fps))  # same trim as make_clips
    pose = d["pose"][:, :, :2] * [w, h]
    ok = d["pose"][:, 11, 3] > 0.3
    ref = pose[ok] if ok.any() else pose
    centre = np.median((ref[:, 11] + ref[:, 12]) / 2, axis=0)
    scale = float(np.median(np.linalg.norm(ref[:, 11] - ref[:, 12], axis=1)))
    body = lambda a: np.round((a[..., :2] * [w, h] - centre) / scale, 3)
    frames = []
    for t in range(s, e + 1):
        frames.append({
            "arm": body(d["pose"][t, ARM]).tolist(),
            "lh": body(d["lh"][t]).tolist() if d["lh_p"][t] else None,
            "rh": body(d["rh"][t]).tolist() if d["rh_p"][t] else None,
        })
    return {"label": label, "fps": fps, "frames": frames, "source": p.parent.parent.parent.name}


# ---- scoring ------------------------------------------------------------------------------------
def _mirror(f):
    """Left-right mirror of resampled features (same as train.mirror, without importing torch code)."""
    g = f.copy()
    swap = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9]
    pose = g[:, :NP].reshape(len(g), -1, 2)[:, swap]
    pose[..., 0] *= -1
    g[:, :NP] = pose.reshape(len(g), -1)
    for a, b in ((LB, RB), (LL, RL)):
        lh, rh = f[:, a:a + 63].reshape(len(f), 21, 3).copy(), f[:, b:b + 63].reshape(len(f), 21, 3).copy()
        lh[..., 0] *= -1
        rh[..., 0] *= -1
        g[:, a:a + 63], g[:, b:b + 63] = rh.reshape(len(f), -1), lh.reshape(len(f), -1)
    g[:, LP], g[:, RP] = f[:, RP], f[:, LP]
    return g


# Thresholds from data (ml/coach.py calibration, 2026-10-30): 90th percentile of correct INCLUDE signs vs the
# ISLRTC reference, so each check fires on <= ~10% of correct attempts. Signers and variants differ a lot, so
# tips are only shown when the recogniser says the attempt failed.
RAISED_Y = 1.6  # hand counts as "in use" above this (body space, shoulder widths below the shoulder line)
DY, DX = 0.55, 0.3
PATH_SMALL, PATH_BIG = 0.45, 4.7


def _hand(f, side):
    """(body (T,21,3), local (T,21,3), in-use mask): present AND raised (a resting hand often stays visible)."""
    b, l, p = (LB, LL, LP) if side == "left" else (RB, RL, RP)
    body = f[:, b:b + 63].reshape(len(f), 21, 3)
    return body, f[:, l:l + 63].reshape(len(f), 21, 3), (f[:, p] > 0.5) & (body[:, 0, 1] < RAISED_Y)


def _distance(u, r):
    """Rough trajectory distance used to pick plain vs mirrored reference."""
    d = 0.0
    for side in ("left", "right"):
        (ub, _, up), (rb, _, rp) = _hand(u, side), _hand(r, side)
        both = up & rp
        d += float(np.linalg.norm(ub[both, 0, :2] - rb[both, 0, :2], axis=1).mean()) if both.any() else 0.0
        d += 2.0 * abs(float(up.mean()) - float(rp.mean()))
    return d


def compare(user_frames, ref_frames):
    """Both: variable-length clip_frames() output. Returns (tips, bad_hands, mirrored, details)."""
    u, r = resample(user_frames, T_OUT), resample(ref_frames, T_OUT)
    mirrored = _distance(u, _mirror(r)) < _distance(u, r) * 0.8
    if mirrored:
        r = _mirror(r)
    tips, bad = [], set()  # (severity, text)
    used = {s: float(_hand(u, s)[2].mean()) for s in ("left", "right")}
    need = {s: float(_hand(r, s)[2].mean()) for s in ("left", "right")}
    two_u = min(used.values()) > 0.3
    two_r = min(need.values()) > 0.3
    if two_r and not two_u:
        tips.append((3.0, "This sign uses both hands."))
        bad.add(min(used, key=used.get))
    elif two_u and not two_r:
        tips.append((2.0, "This sign uses only one hand. Keep the other one down."))
        bad.add(min(need, key=need.get))
    nose_y = float(np.median(r[:, NOSE_Y]))
    for side in ("left", "right"):
        (ub, ul, up), (rb, rl, rp) = _hand(u, side), _hand(r, side)
        both = up & rp
        if both.sum() < 5:
            continue
        name = f"your {side} hand"
        dy = float(np.mean(ub[both, 0, 1] - rb[both, 0, 1]))  # + = learner lower (image y grows down)
        dx = float(np.mean(np.abs(ub[both, 0, 0]) - np.abs(rb[both, 0, 0])))  # + = further from the centre
        where = "near your face" if np.mean(rb[both, 0, 1]) < nose_y + 0.6 else "at chest height"
        if abs(dy) > DY:
            tips.append((abs(dy) * 2, f"Move {name} {'higher' if dy > 0 else 'lower'} ({where})."))
            bad.add(side)
        if abs(dx) > DX:
            tips.append((abs(dx) * 1.5, f"Keep {name} {'closer to your body' if dx > 0 else 'further out to the side'}."))
            bad.add(side)
        path = lambda b, m: float(np.linalg.norm(np.diff(b[m, 0, :2], axis=0), axis=1).sum())
        pu, pr = path(ub, both), path(rb, both)
        if pr > 0.5 and pu < PATH_SMALL * pr:
            tips.append((1.5 * (1 - pu / pr), f"Make the movement of {name} bigger."))
            bad.add(side)
        elif pr > 0.3 and pu > PATH_BIG * pr:
            tips.append((1.0, f"Make the movement of {name} smaller and calmer."))
        # finger shape varies a lot between signers, so only flag very large differences (and only
        # on failed attempts, see score_attempt)
        shape = float(np.abs(ul[both, 1:, :2] - rl[both, 1:, :2]).mean())
        if shape > 1.3:
            tips.append((0.5, f"Check the finger shape of {name}. Watch the fingers in slow motion."))
    ratio = len(user_frames) / max(1, len(ref_frames))
    if ratio < 0.5:
        tips.append((1.2, "Slow down a little and finish the whole movement."))
    elif ratio > 2.2:
        tips.append((0.8, "Try it a bit faster and smoother."))
    tips.sort(key=lambda x: -x[0])
    return [t for _, t in tips[:2]], sorted(bad), mirrored, {"duration_ratio": round(ratio, 2)}


def score_attempt(rec, label, pose, lh, rh, lh_p, rh_p):
    """Live attempt (pixel units) -> result dict for the web page."""
    user = clip_frames(pose, lh, rh, lh_p, rh_p, 1, 1)
    if not (lh_p.any() or rh_p.any()):
        return {"label": label, "stars": 0, "score": 0, "tips": ["I couldn't see your hands. Step back a little."],
                "bad_hands": [], "top": []}
    ref_path = _source(label)
    ref = npz_frames(ref_path) if ref_path else None
    emb = rec.embed_frames([user])[0]
    top = rec.predict(emb, k=3)
    sims = []
    cid = {c: i for i, c in enumerate(rec.classes)}
    if label in cid:
        sims.append(float(rec.weights[cid[label]] @ emb))
    if ref is not None:
        e_ref = rec.embed_frames([ref])[0]
        sims.append(float(e_ref @ emb))
        sims.append(float(rec.embed_frames([_mirror(resample(ref, T_OUT))])[0] @ emb))
    sim = max(sims) if sims else 0.0
    rank = next((i for i, t in enumerate(top) if t["label"] == label), None)
    score = int(round(100 * float(np.clip((sim - 0.15) / 0.65, 0, 1))))
    stars = 3 if (rank == 0 or sim > 0.8) else 2 if (rank is not None or sim > 0.6) else 1
    tips, bad, mirrored, extra = compare(user, ref) if ref is not None else ([], [], False, {})
    if stars == 3:  # recognised: geometric tips are too unreliable across signers to nitpick a success
        tips, bad = ["Great! That looks right."], []
    elif not tips:
        tips = ["Not quite. Watch the video in slow motion and follow the ghost hands."]
    return {"label": label, "stars": stars, "score": score, "tips": tips, "bad_hands": bad,
            "mirrored": mirrored, "top": [t["label"] for t in top], **extra}
