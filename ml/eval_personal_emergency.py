"""Personalisation before/after on 26 signers the model has NEVER seen (Mendeley ISL emergency words, CC BY 4.0).

Each signer signs 8 words twice (<word>_<signer>_<take>). 4 words are in the app vocabulary (Doctor, Help,
Pain, hot) and 4 are not (Accident, Call, Lose, Thief = "brand-new personal signs").
Per signer, per direction (take 1 -> 2 and 2 -> 1):
  before  generic model, full app vocabulary (202 signs); new words cannot be recognised at all
  after   the signer teaches each of the 8 words with ONE take (private samples, as in the Teach sheet),
          then the other take is recognised among the full vocabulary + the new words.
Keypoints: data/emergency/keypoints (ml/extract.py, same MediaPipe Tasks models as the browser).

CAVEAT (checked 31 Oct): these videos show ONLY the hands in front of a black board (no face / shoulders), so
pose and the shoulder-based normalisation fail. "before" (~4%) is a framing failure, not a sign-variant
result. Quote it as "a camera setup the generic model cannot read; one example per sign recovers X%".
Do NOT add this dataset to training (bodies without shoulders). Result 31 Oct, model.pt (v10_all):
app signs 3.8% -> 55.3% top-1 (78.4% top-3) after 1 take; new signs 0% -> 65.3%; 26/26 signers improved.

    python ml/eval_personal_emergency.py [--model data/model.pt]
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from features import npz_frames
from recognizer import Recognizer

ROOT = Path(__file__).resolve().parent.parent
KP = ROOT / "data" / "emergency" / "keypoints"
LABEL = {"Doctor": "Doctor", "Help": "Help", "Pain": "Pain", "hot": "hot"}  # dataset folder -> app label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "data" / "model.pt"))
    args = ap.parse_args()
    rec = Recognizer(args.model)
    v = json.loads((ROOT / "ml" / "vocab.json").read_text())
    rec.active = set(v["include_signs"]) | set(v["cislr_signs"]) | set(v.get("common_signs", []))
    n_vocab = len(rec.sign_labels())

    clips = defaultdict(dict)  # signer -> {(label, take): frames}
    for d in sorted(p for p in KP.iterdir() if p.is_dir()):
        label = LABEL.get(d.name, d.name)
        for f in sorted(d.glob("*.npz")):
            m = re.match(r".*_(\d+)_(\d+)$", f.stem)
            if m:
                clips[int(m.group(1))][(label, int(m.group(2)))] = npz_frames(f)
    keys = [(s, k) for s in sorted(clips) for k in sorted(clips[s])]
    embs = dict(zip(keys, rec.embed_frames([clips[s][k] for s, k in keys])))
    known = set(LABEL.values())

    res = {g: {"before": [0, 0, 0], "after": [0, 0, 0]} for g in ("known", "new")}  # top1, top3, n
    per_signer = {}
    for s in sorted(clips):
        hit_b = hit_a = n_s = 0
        for teach, test in ((1, 2), (2, 1)):
            labels = sorted({l for l, t in clips[s] if t == teach and (l, test) in clips[s]})
            for phase in ("before", "after"):
                rec.clear_samples()
                if phase == "after":
                    for l in labels:
                        rec.add_sample(l, embs[(s, (l, teach))])
                for l in labels:
                    pred = [p["label"] for p in rec.predict(embs[(s, (l, test))], k=3)]
                    r = res["known" if l in known else "new"][phase]
                    r[0] += pred[0] == l
                    r[1] += l in pred
                    r[2] += 1
                    if l in known:
                        hit_b += phase == "before" and pred[0] == l
                        hit_a += phase == "after" and pred[0] == l
                        n_s += phase == "before"
        per_signer[s] = {"before": hit_b / max(1, n_s), "after": hit_a / max(1, n_s)}

    print(f"{len(clips)} unseen signers, {len(keys)} clips; recognised among {n_vocab} app signs (+ the new ones after)")
    out = {"signers": len(clips), "vocab": n_vocab}
    for g, title in (("known", "4 app signs (Doctor, Help, Pain, Hot)"), ("new", "4 brand-new signs (Accident, Call, Lose, Thief)")):
        b, a = res[g]["before"], res[g]["after"]
        print(f"  {title}: before top-1 {b[0] / b[2]:.1%} (top-3 {b[1] / b[2]:.1%})  ->  after ONE personal take "
              f"top-1 {a[0] / a[2]:.1%} (top-3 {a[1] / a[2]:.1%})   n={a[2]}")
        out[g] = {"before_top1": b[0] / b[2], "before_top3": b[1] / b[2], "after_top1": a[0] / a[2],
                  "after_top3": a[1] / a[2], "n": a[2]}
    better = sum(p["after"] > p["before"] for p in per_signer.values())
    worse = sum(p["after"] < p["before"] for p in per_signer.values())
    print(f"  per signer (app signs): improved {better}, worse {worse}, same {len(per_signer) - better - worse}")
    out["per_signer"] = per_signer
    dst = ROOT / "data" / "eval_personal_emergency.json"
    dst.write_text(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
