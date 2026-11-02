"""Organise CISLR (Exploration-Lab/CISLR, AFL-3.0) videos into data/cislr/videos/<Label>/<uid>.<ext>.

Two uses:
  * new scenario signs missing from INCLUDE (vocab.json "cislr_signs") -> few-shot enrolment + training
  * INCLUDE signs re-signed by different people -> cross-signer test set (label spelled as in INCLUDE)

    python ml/prep_cislr.py
"""
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CISLR = ROOT / "data" / "cislr"
OUT = CISLR / "videos"


def main():
    vocab = json.loads((ROOT / "ml" / "vocab.json").read_text())
    include = {s.lower(): s for s in vocab["include_signs"]}
    extra = {s.lower(): s for s in vocab["cislr_signs"]}
    wanted = {**include, **extra}

    files = {p.stem: p for p in (CISLR / "raw").rglob("*") if p.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov")}
    rows = list(csv.DictReader(open(CISLR / "dataset.csv", encoding="utf8")))
    n = missing = 0
    per = {}
    for r in rows:
        g = r["gloss"].strip().lower()
        if g not in wanted:
            continue
        src = files.get(r["uid"])
        if src is None:
            missing += 1
            continue
        label = wanted[g]
        dst = OUT / label / f"{r['uid']}{src.suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        per[label] = per.get(label, 0) + 1
        n += 1
    print(f"{n} videos for {len(per)} signs ({missing} listed but not in zip)")
    for k in sorted(per):
        tag = "TEST (INCLUDE sign)" if k.lower() in include else "new sign"
        print(f"  {k:14s} {per[k]}  {tag}")


if __name__ == "__main__":
    main()
