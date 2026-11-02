"""Keypoints -> fixed-size feature tensor. THE single source of truth; web/src/features.ts must match exactly.

Steps (per clip):
 1. Convert normalised image coords to pixel-aspect units: x*W, y*H, z*W (camera aspect stops mattering).
 2. Trim leading/trailing frames with no hand (signer at rest), keep MARGIN frames.
 3. Centre on shoulder midpoint (pose 11/12) and scale by shoulder width, both clip medians.
 4. Per frame: 11 upper-body pose joints (x,y) + both hands (x,y,z) in body space
    + both hands in hand-local space (minus wrist, / wrist->middle-MCP length) + 2 presence flags.
    Missing hands are zeros with presence 0.
 5. Linearly resample to T_OUT frames (presence: nearest frame).
"""
import numpy as np

T_OUT = 48
MARGIN = 2
POSE_IDX = [0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16]  # nose, eyes, mouth, shoulders, elbows, wrists
FEAT_DIM = len(POSE_IDX) * 2 + 21 * 3 * 4 + 2  # 276


def _to_pixels(a, w, h):
    a = a.astype(np.float32).copy()
    a[..., 0] *= w
    a[..., 1] *= h
    if a.shape[-1] >= 3:
        a[..., 2] *= w
    return a


def clip_features(pose, lh, rh, lh_p, rh_p, width, height):
    """pose (T,33,4) lh/rh (T,21,3) lh_p/rh_p (T,) bool -> (T_OUT, FEAT_DIM) float32."""
    return resample(clip_frames(pose, lh, rh, lh_p, rh_p, width, height), T_OUT)


def clip_frames(pose, lh, rh, lh_p, rh_p, width, height):
    """Steps 1-4: variable-length (T', FEAT_DIM) per-frame features."""
    pose = _to_pixels(pose[..., :3], width, height)
    lh, rh = _to_pixels(lh, width, height), _to_pixels(rh, width, height)
    lh_p, rh_p = np.asarray(lh_p, bool), np.asarray(rh_p, bool)

    any_hand = np.flatnonzero(lh_p | rh_p)
    if len(any_hand):
        s, e = max(0, any_hand[0] - MARGIN), min(len(pose), any_hand[-1] + 1 + MARGIN)
        pose, lh, rh, lh_p, rh_p = pose[s:e], lh[s:e], rh[s:e], lh_p[s:e], rh_p[s:e]

    pose_ok = np.abs(pose[:, 11, :2]).sum(1) > 0
    ref = pose[pose_ok] if pose_ok.any() else pose
    centre = np.median((ref[:, 11, :2] + ref[:, 12, :2]) / 2, axis=0)
    scale = float(np.median(np.linalg.norm(ref[:, 11, :2] - ref[:, 12, :2], axis=1)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(width) / 4

    def body(a):
        out = a.copy()
        out[..., :2] = (out[..., :2] - centre) / scale
        if out.shape[-1] == 3:
            out[..., 2] = out[..., 2] / scale
        return out

    def local(hand):
        rel = hand - hand[:, :1]
        size = np.linalg.norm(rel[:, 9, :2], axis=1, keepdims=True)[..., None]
        return rel / np.maximum(size, 1e-3)

    pose_f = body(pose[:, POSE_IDX, :2]).reshape(len(pose), -1)
    feats = np.concatenate([
        pose_f,
        body(lh).reshape(len(lh), -1) * lh_p[:, None],
        body(rh).reshape(len(rh), -1) * rh_p[:, None],
        local(lh).reshape(len(lh), -1) * lh_p[:, None],
        local(rh).reshape(len(rh), -1) * rh_p[:, None],
        lh_p[:, None].astype(np.float32),
        rh_p[:, None].astype(np.float32),
    ], axis=1).astype(np.float32)
    return feats


def resample(feats, t_out):
    """Linear time resampling; the last 2 columns (presence) use nearest frame."""
    t = len(feats)
    if t == 1:
        return np.repeat(feats, t_out, axis=0)
    pos = np.linspace(0, t - 1, t_out)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    frac = (pos - lo)[:, None].astype(np.float32)
    out = feats[lo] * (1 - frac) + feats[hi] * frac
    out[:, -2:] = feats[np.rint(pos).astype(int), -2:]
    return out.astype(np.float32)


def npz_frames(path):
    d = np.load(path)
    return clip_frames(d["pose"], d["lh"], d["rh"], d["lh_p"], d["rh_p"], int(d["width"]), int(d["height"]))
