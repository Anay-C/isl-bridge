"""Live recogniser: model + personal prototypes. Used by the server and by ml/eval_personal.py.

Scoring (all on the unit sphere):
    base_c     = cos(e, w_c)                        trained class weight
    personal_c = mean of top-2 cos(e, sample_k)      user's own samples of sign c
    score_c    = (1-ALPHA)*base_c + ALPHA*personal_c  if the user enrolled c
               = personal_c                          if c is a brand-new user sign
               = base_c                              otherwise
    PERS_MODE "max" (default): a taught sign scores max(blend, trusted_c), trusted_c = personal_c over the
    TRUSTED samples only. A sample is trusted when the model itself reads it as that sign (cos to the class
    weight >= TRUST_BASE), or another of the user's samples of that sign resembles it (cos >= TRUST_PEER), or it
    is the sign's only sample. Untrusted samples still count through the blend. Why: on 1 Nov Anay's taught
    "Help" held 2 stray reps that the model reads as "I" (0.59 / 0.67) and that match none of his other Help
    reps (<= 0.53), so every "I"-like movement (me, pointing at yourself, hands on the chest) became "Help" at
    full strength. Calibrated on 26 unseen signers (Mendeley emergency, 2 takes each): two takes of the same
    sign agree at >= 0.6 in 59% of pairs, different signs only 3%.
Probabilities = softmax(SCALE * score) over the active vocabulary.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from features import T_OUT, clip_frames, resample
from model import SignModel

ALPHA = 0.6
PERS_MODE = os.environ.get("PERS_MODE", "max")  # "max": a taught sign also counts fully when it matches the user's own samples
TRUST_BASE = 0.4  # the model itself reads the sample as that sign
TRUST_PEER = 0.6  # another of the user's samples of the same sign resembles it
EX_MODE = os.environ.get("EX_MODE", "max")  # exemplar bank (set_exemplars): "max" | "blend" | "off"
EX_DELTA = float(os.environ.get("EX_DELTA", "0.1"))  # "max": an exemplar match counts as (cos - EX_DELTA)
EX_BETA = float(os.environ.get("EX_BETA", "0.5"))  # "blend": weight of the closest exemplar
NONE = "_none_"
SCALE = 16.0


class Recognizer:
    def __init__(self, model_path, device="cpu"):
        """model_path: one checkpoint, or several (list, or "a.pt,b.pt") = an ensemble. Ensemble members must
        share the class list; their unit embeddings are concatenated and divided by sqrt(N), so one cosine in
        the joint space is the MEAN of the members' cosines. Personal samples, prototypes and the scoring
        below work unchanged (embeddings just get longer)."""
        paths = [p for p in str(model_path).split(",") if p] if not isinstance(model_path, (list, tuple)) else list(model_path)
        self.models, weights = [], []
        for p in paths:
            ck = torch.load(p, map_location=device, weights_only=False)
            classes = list(ck["classes"])
            if self.models and classes != self.classes:
                raise ValueError(f"ensemble member {p} has a different class list")
            self.classes = classes
            m = SignModel(ck["feat_dim"], len(classes))
            m.load_state_dict(ck["state"])
            m.eval().to(device)
            self.models.append(m)
            with torch.no_grad():
                weights.append(F.normalize(m.weight, dim=-1).cpu().numpy())
        self.model = self.models[0]
        self.device = device
        self.weights = (np.concatenate(weights, axis=1) / np.sqrt(len(weights))).astype(np.float32)
        self.dim = self.weights.shape[1]
        self.samples: dict[str, list[np.ndarray]] = {}  # label -> personal embeddings
        self.active: set[str] | None = None  # restrict output to a scenario vocabulary
        self._cid = {c: i for i, c in enumerate(self.classes)}
        self._trust_cache: dict = {}  # per-object: the server gives each connection its own (see app.py)
        self._ex = None
        if os.environ.get("EXEMPLARS"):  # evaluation scripts: EXEMPLARS=data/exemplars_v10.npz
            self.set_exemplars(os.environ["EXEMPLARS"])

    # ---- embeddings -------------------------------------------------------------------------
    def embed_frames(self, frames_list):
        """frames_list: list of variable-length (T, FEAT_DIM) arrays -> (N, emb) numpy."""
        x = torch.tensor(np.stack([resample(f, T_OUT) for f in frames_list])).to(self.device)
        with torch.no_grad():
            if len(self.models) == 1:
                return self.model.embed(x).cpu().numpy()
            z = torch.cat([m.embed(x) for m in self.models], -1) / float(np.sqrt(len(self.models)))
            return z.cpu().numpy()

    def embed_raw(self, pose, lh, rh, lh_p, rh_p, width, height):
        return self.embed_frames([clip_frames(pose, lh, rh, lh_p, rh_p, width, height)])[0]

    # ---- personalisation --------------------------------------------------------------------
    def add_sample(self, label, emb):
        self.samples.setdefault(label, []).append(np.asarray(emb, np.float32))
        self._trust_cache.pop(label, None)

    def clear_samples(self, label=None):
        if label is None:
            self.samples.clear()
            self._trust_cache.clear()
        else:
            self.samples.pop(label, None)
            self._trust_cache.pop(label, None)

    def trust_mask(self, label, S=None):
        """Which samples of `label` may count at full strength (see module docstring)."""
        S = np.stack(self.samples[label]) if S is None else S
        if len(S) == 1:
            return np.ones(1, bool)
        pair = S @ S.T
        np.fill_diagonal(pair, -1.0)
        ok = pair.max(1) >= TRUST_PEER
        if label in self._cid:
            ok |= S @ self.weights[self._cid[label]] >= TRUST_BASE
        return ok

    def _trusted(self, label):
        """Trusted samples of `label`, cached until that sign's samples change (scores() runs per window)."""
        S = self.samples[label]
        key = tuple(float(x[0]) + 3.0 * float(x[1]) for x in S)  # content fingerprint (lists get reassigned)
        hit = self._trust_cache.get(label)
        if hit is None or hit[0] != key:
            M = np.stack(S)
            hit = (key, M, M[self.trust_mask(label, M)])
            self._trust_cache[label] = hit
        return hit[1], hit[2]

    def enroll_report(self, label):
        """Clean up and self-test a freshly taught sign. Drops stray reps (the model does not read them as
        the sign AND no other rep resembles them: usually a mis-cut rep, e.g. a "me" caught while teaching),
        else shaky reps far from the sign's mean; then leave-one-out: each kept rep must be recognised as
        `label` from the other reps alone."""
        S = np.stack(self.samples[label])
        mean = S.mean(0) / np.linalg.norm(S.mean(0))
        c = S @ mean
        trusted = np.flatnonzero(self.trust_mask(label, S)) if len(S) >= 3 else np.arange(len(S))
        if len(trusted) >= 2:
            keep = trusted
        else:
            keep = np.flatnonzero(c >= np.median(c) - 0.2) if len(S) >= 4 else np.arange(len(S))
            if len(keep) < 3:
                keep = np.argsort(-c)[:3]
        S = S[keep]
        hits, confused = 0, {}
        for i in range(len(S)):
            self.samples[label] = [s for j, s in enumerate(S) if j != i]
            top = self.predict(S[i], k=1)[0]["label"]
            hits += top == label
            if top != label:
                confused[top] = confused.get(top, 0) + 1
        self.samples[label] = list(S)
        pair = S @ S.T
        return {"label": label, "kept": len(S), "dropped": int(len(c) - len(S)), "self_test": hits,
                "consistency": round(float(pair[~np.eye(len(S), dtype=bool)].mean()), 3),
                "confused_with": sorted(confused, key=confused.get, reverse=True)}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({k: [v.tolist() for v in vs] for k, vs in self.samples.items()}))

    def load(self, path):
        if Path(path).exists():
            raw = json.loads(Path(path).read_text())
            # samples from a different model (other embedding size) cannot be compared: skip them (re-teach)
            self.samples = {k: [np.array(v, np.float32) for v in vs if len(v) == self.dim] for k, vs in raw.items()}
            stale = sorted(k for k, vs in self.samples.items() if not vs)
            for k in stale:
                del self.samples[k]
            if stale:
                print(f"[recognizer] {path}: samples of {stale} were taught with another model; re-teach them", flush=True)
            self._trust_cache = {}

    # ---- prediction -------------------------------------------------------------------------
    def labels(self):
        labs = self.classes + [s for s in self.samples if s not in self.classes]
        if self.active is not None:
            labs = [l for l in labs if l in self.active or l == NONE or l in self.samples]  # taught signs always count
        return labs

    def sign_labels(self):
        return [l for l in self.labels() if l != NONE]

    def set_exemplars(self, path):
        """Load an exemplar bank (ml/exemplars.py): every training clip's embedding. A sign then also scores by its
        closest real recording (one variant), not only by its class weight (all variants averaged)."""
        d = np.load(path)
        labs, E = d["label"], d["emb"].astype(np.float32)
        if E.shape[1] != self.dim:
            raise ValueError(f"exemplar bank {path} was made with another model")
        idx = sorted((i for i, l in enumerate(labs) if l in self._cid), key=lambda i: self._cid[labs[i]])
        cls = np.array([self._cid[labs[i]] for i in idx])
        starts = np.flatnonzero(np.r_[True, cls[1:] != cls[:-1]])
        self._ex = (E[idx], starts, cls[starts])

    def _base(self, emb):
        """Class-weight cosines, raised by the closest exemplar when a bank is loaded (EX_MODE)."""
        base = self.weights @ emb
        if getattr(self, "_ex", None) is not None and EX_MODE != "off":
            E, starts, cls = self._ex
            sims = E @ emb
            near = np.full(len(base), -1.0, np.float32)
            near[cls] = np.maximum.reduceat(sims, starts)
            if EX_MODE == "max":
                base = np.maximum(base, near - EX_DELTA)
            else:  # blend
                has = near > -1.0
                base = np.where(has, (1 - EX_BETA) * base + EX_BETA * near, base)
        return base

    def scores(self, emb, personalise=True):
        labs = self.labels()
        cid = self._cid
        out = np.empty(len(labs), np.float32)
        allbase = self._base(emb)
        for j, l in enumerate(labs):
            base = float(allbase[cid[l]]) if l in cid else None
            pers = trusted = None
            if personalise and self.samples.get(l):
                S, T = self._trusted(l)
                pers = float(np.sort(S @ emb)[::-1][:2].mean())
                if len(T):
                    trusted = float(np.sort(T @ emb)[::-1][:2].mean())
            if base is None: out[j] = pers
            elif pers is None: out[j] = base
            else:
                blend = (1 - ALPHA) * base + ALPHA * pers
                out[j] = max(blend, trusted) if PERS_MODE == "max" and trusted is not None else blend
        return labs, out

    def predict(self, emb, k=3, personalise=True):
        """Top-k signs. The "_none_" (transition) class, if the model has one, takes probability mass but is
        never returned, so a window between two signs gets a low p for every real sign."""
        labs, s = self.scores(emb, personalise)
        p = np.exp(SCALE * (s - s.max()))
        p /= p.sum()
        top = [i for i in np.argsort(-p) if labs[i] != NONE][:k]
        return [{"label": labs[i], "p": float(p[i]), "cos": float(s[i])} for i in top]
