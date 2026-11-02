"""Sign spotting: find the individual signs inside one continuous stretch of signing.

People do not drop their hands between signs, so cutting on "hands at rest" merges a whole sentence into one
segment (measured on ISL-CSLTR: 1.5 segments for 4.1-word sentences, 5% of words found). Instead, score sliding
windows of several lengths across the stretch with the recogniser and keep the confident, non-overlapping
peaks (greedy non-maximum suppression). Used by the live server and ml/eval_live.py.
"""
import numpy as np

from features import clip_frames

BASE_FPS = 25.0  # window lengths below are frames at this rate (training clips are 25-30 fps)
LENS = (16, 24, 32, 44)
STRIDE = 3
import os  # noqa: E402
# segmentation decoder (decode()): window score = seconds * (cos - C0) - PER_SIGN; best non-overlapping cover
C0 = float(os.environ.get("SPOT_C0", 0.55))
PER_SIGN = float(os.environ.get("SPOT_PER_SIGN", 0.0))
LONG_LENS = (56, 72, 90)
# Single-sign preference. A window scores seconds * (cos - C0) and unexplained frames are free, so a 1.2 s fragment
# that happens to match another sign well (cos 0.85, e.g. the open palms that start BOOK look like WHAT) beat the
# whole sign (2 s at cos 0.70). The whole stretch gets WHOLE_BONUS per second extra when it is short (<= WHOLE_MAX_S)
# or, longer, still matches one sign as a whole (cos >= WHOLE_MIN_COS; a sentence as a whole is a blend, median cos
# 0.38). Measured 2 Nov with the segmenter's new context frames (server/segmenter.py PRE_ROLL / KEEP_TAIL):
#   held-out isolated clips (624, live model, server path)             69.6% -> 78.2% top-1
#   unseen CSLTR signers 0,3 (model_v10, 3 seeds): words               61.8% -> 71.6% top-1 (top-3 71.6 -> 80.4%)
#                                                  sentences recall    61.8% -> 58.8%, precision 59% -> 58%
# (gate 0.55: words 74.5% but sentences 56.6%; 0.65: words 67.6%, sentences 61.4%.)
WHOLE_BONUS = float(os.environ.get("SPOT_WHOLE_BONUS", 0.15))
WHOLE_MAX_S = float(os.environ.get("SPOT_WHOLE_MAX_S", 2.0))
WHOLE_MIN_COS = float(os.environ.get("SPOT_WHOLE_MIN_COS", 0.60))
DECODER = os.environ.get("SPOT_DECODER", "dp")


def _arrays(seg):
    z = lambda k: np.zeros(k, np.float32)
    pose = np.stack([f["pose"] if f["pose"] is not None else z((33, 4)) for f in seg])
    lh = np.stack([f["lh"] if f["lh"] is not None else z((21, 3)) for f in seg])
    rh = np.stack([f["rh"] if f["rh"] is not None else z((21, 3)) for f in seg])
    lp = np.array([f["lh"] is not None for f in seg])
    rp = np.array([f["rh"] is not None for f in seg])
    return pose, lh, rh, lp, rp


def embed_segment(rec, seg):
    return rec.embed_frames([clip_frames(*_arrays(seg), 1, 1)])[0]


def spot(rec, frames, fps=BASE_FPS, thr=0.35, max_overlap=0.3, k=3):
    """frames: server-style dicts (pose/lh/rh in pixel units, None if missing).
    Returns [(start, end, top_k predictions)] in time order."""
    active = [i for i, f in enumerate(frames) if f["lh"] is not None or f["rh"] is not None]
    if not active:
        return []
    a, b = active[0], active[-1] + 1
    scale = fps / BASE_FPS
    lens = sorted({max(6, int(round(L * scale))) for L in LENS})
    stride = max(1, int(round(STRIDE * scale)))
    wins = [(s, s + L) for L in lens for s in range(a, max(a + 1, b - L + 1), stride) if s + L <= b + 2]
    if DECODER == "dp":
        return decode(rec, frames, a, b, scale, stride, k)
    embs = rec.embed_frames([clip_frames(*_arrays(frames[s:e]), 1, 1) for s, e in wins])
    scored = []
    for (s, e), emb in zip(wins, embs):
        top = rec.predict(emb, k=k)
        if top:
            scored.append((top[0]["p"], s, e, top))
    keep = []
    for p, s, e, top in sorted(scored, key=lambda x: -x[0]):
        if p < thr:
            break
        if all(min(e, e2) - max(s, s2) <= max_overlap * min(e - s, e2 - s2) for _, s2, e2, _ in keep):
            keep.append((p, s, e, top))
    # one long sign often wins two neighbouring windows ("you you"): merge back-to-back repeats of the same
    # top-1 sign into one (keep the more confident). Measured on unseen CSLTR signers: see HANDOFF.md.
    out = []
    for p, s, e, top in sorted(keep, key=lambda x: x[1]):
        if out and out[-1][3][0]["label"] == top[0]["label"]:
            q, s0, e0, t0 = out[-1]
            out[-1] = (max(p, q), s0, max(e, e0), t0 if q >= p else top)
            continue
        out.append((p, s, e, top))
    return [(s, e, top) for _, s, e, top in out]


def window_preds(rec, frames, a, b, scale, stride, k=3):
    """Candidate windows over the stretch [a, b) (short + long lengths and the whole stretch) with each window's
    top-k signs. This is the expensive part of decode() (one embedding per window)."""
    lens = sorted({max(6, int(round(L * scale))) for L in LENS + LONG_LENS})
    starts = list(range(a, b, stride))
    wins = {(s, min(b, s + L)) for L in lens for s in starts if s + L <= b + stride}
    wins.add((a, b))
    wins = sorted(w for w in wins if w[1] - w[0] >= max(6, int(10 * scale)) or w == (a, b))
    embs = rec.embed_frames([clip_frames(*_arrays(frames[s:e]), 1, 1) for s, e in wins])
    return [(s, e, rec.predict(emb, k=k)) for (s, e), emb in zip(wins, embs)]


def decode(rec, frames, a, b, scale, stride, k=3):
    """Explain the whole stretch [a, b) as a sequence of signs (dynamic programming over candidate windows).

    The softmax confidence is ~0.99 for almost every window (a fragment of a sign, or the hand coming up,
    still looks like *some* sign), so greedy peak picking cuts one slow sign into several wrong ones. Instead
    each window scores  seconds * (cos - C0) - PER_SIGN  (cos = match to its best sign): long windows that
    match well beat several short mediocre ones, frames may stay unexplained (transitions), and a new sign
    costs PER_SIGN. Window grid: short + long lengths and the whole stretch."""
    return dp(window_preds(rec, frames, a, b, scale, stride, k), a, b, scale)


def dp(preds, a, b, scale, c0=None, per_sign=None, whole_bonus=None, whole_max_s=None, whole_min_cos=None):
    """The decoder's dynamic programme over window_preds() output (parameters default to the module constants)."""
    c0 = C0 if c0 is None else c0
    per_sign = PER_SIGN if per_sign is None else per_sign
    whole_bonus = WHOLE_BONUS if whole_bonus is None else whole_bonus
    whole_max_s = WHOLE_MAX_S if whole_max_s is None else whole_max_s
    whole_min_cos = WHOLE_MIN_COS if whole_min_cos is None else whole_min_cos
    whole = next((t for s, e, t in preds if (s, e) == (a, b)), None)
    # the single-sign preference applies to a short stretch, or to a longer one that still matches one sign well as a
    # whole (slow signers); a sentence is long AND, as a whole, a blend of several signs (low cosine)
    if not whole or ((b - a) / (BASE_FPS * scale) > whole_max_s and whole[0]["cos"] < whole_min_cos):
        whole_bonus = 0.0
    by_end = {}
    for s, e, top in preds:
        if not top:
            continue
        sec = (e - s) / (BASE_FPS * scale)
        sc = sec * (top[0]["cos"] - c0 + (whole_bonus if (s, e) == (a, b) else 0.0)) - per_sign
        if sc > 0:
            by_end.setdefault(e, []).append((s, sc, top))
    # dp over frame positions: best[t] = best score explaining [a, t)
    best = {a: (0.0, None)}
    order = sorted(set([a, b] + [e for e in by_end] + [s for ws in by_end.values() for s, _, _ in ws]))
    run = (0.0, a)  # best (score, position) seen so far: frames may be skipped (unexplained)
    back = {}
    for t in order:
        if t in best and best[t][0] > run[0]:
            run = (best[t][0], t)
        cand = (run[0], ("skip", run[1]))
        for s, sc, top in by_end.get(t, []):
            prev = max(((best[q][0], q) for q in best if q <= s), default=(0.0, a))
            if prev[0] + sc > cand[0]:
                cand = (prev[0] + sc, ("sign", prev[1], s, top))
        if cand[0] > best.get(t, (-1.0,))[0]:
            best[t] = (cand[0], None)
            back[t] = cand[1]
        if best[t][0] > run[0]:
            run = (best[t][0], t)
    # backtrack from the best end position
    t = max(best, key=lambda q: best[q][0])
    out = []
    while t in back and t != a:
        step = back[t]
        if step[0] == "skip":
            if step[1] == t:
                break
            t = step[1]
        else:
            _, prev, s, top = step
            out.append((s, t, top))
            t = prev
    out.reverse()
    merged = []  # same sign twice in a row = one sign split in two
    for s, e, top in out:
        if merged and merged[-1][2][0]["label"] == top[0]["label"]:
            merged[-1] = (merged[-1][0], e, merged[-1][2])
        else:
            merged.append((s, e, top))
    return merged
