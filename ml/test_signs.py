"""Test EVERY sign of the app through the live server path; write a per-sign report.

Accuracy (clips the model never trained on):
  include  INCLUDE recording sessions held out by train.py (stored in the checkpoint)
  cislr    the CISLR clip held out per sign (other signers / cameras)
  csltr    ISL-CSLTR word videos of signers 0,3 (--unseen-csltr; only for models trained WITHOUT them, model_v10*)
Each clip is replayed frame by frame the way the phone sends it, + 20 hands-down frames, through the server's own
code: Segmenter(hold=None, max_len=360) -> spotting.spot on each stretch (fallback: whole stretch), with the
scenario vocabulary. Per sign: n, top-1 (most confident detected sign), top-3, exact (exactly ONE sign shown
and it is right = what a user wants to see), and what it was confused with.

Pipeline check (--pipeline): EVERY clip of every sign, training clips included, through the segmenter only:
  silent = no stretch of signing found at all (the app would never react to that sign), split = more than one
  stretch (the sentence would be cut in two). Structural, independent of the model.

    python ml/test_signs.py --model data/model.pt [--scenario general] [--device cuda] [--pipeline] [--tag live]
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "server"))
import eval_live as E  # noqa: E402
import spotting  # noqa: E402
from recognizer import Recognizer  # noqa: E402
from segmenter import Segmenter  # noqa: E402

DATA = ROOT / "data"
ROOTS = {"include": DATA / "keypoints", "cislr": DATA / "cislr" / "keypoints", "islrtc": DATA / "islrtc" / "keypoints",
         "csltr": DATA / "csltr" / "keypoints"}
VOCAB = json.loads((ROOT / "ml" / "vocab.json").read_text())
# server/app.py scenario_vocab (kept in sync by hand: importing the server would start its Meta API clients)
HOSPITAL_EXTRA = {"it", "you (plural)", "Man", "Woman", "Boy", "Girl", "Son", "Daughter", "Parent", "Grandfather",
                  "Grandmother", "Year", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
CSLTR_ALIAS = {"hi": "Hello", "thank": "Thank you", "thanks": "Thank you", "medicines": "Medicine", "fine": "good",
               "me": "I", "my": "I"}


def scenario_vocab(scenario):
    core = set(VOCAB["include_signs"]) | set(VOCAB["cislr_signs"])
    return core | HOSPITAL_EXTRA if scenario in ("hospital", "pharmacy") else core | set(VOCAB.get("common_signs", []))


def stretches(frames, fps):
    """The server's segmentation of a replayed clip: list of signing stretches."""
    seg, out = Segmenter(hold=None, max_len=360, fps=fps), []
    for f in frames:
        s = seg.push(f, fps=fps)
        if s:
            out.append(s)
    return out


def server_signs(rec, frames, fps):
    """Exactly Session.on_segment: every stretch -> spotted signs (top-3 each), in order."""
    out = []
    for s in stretches(frames, fps):
        spots = spotting.spot(rec, s, fps=fps)
        if not spots:
            spots = [(0, len(s), rec.predict(spotting.embed_segment(rec, s), k=3))]
        out += [top for _, _, top in spots]
    return out


def held_out(ck, unseen_csltr):
    clips = []
    for key in ck.get("test_include", []) + ck.get("test_cislr", []):
        src, rel = key.split("/", 1)
        clips.append((src, key.split("/")[1], ROOTS[src] / rel))
    if unseen_csltr:
        sig = json.loads((DATA / "csltr" / "signer_clusters.json").read_text())
        lower = {c.lower(): c for c in ck["classes"]}
        for d in sorted((ROOTS["csltr"] / "words").iterdir()):
            lab = CSLTR_ALIAS.get(d.name) or lower.get(d.name)
            if not lab:
                continue
            for f in sorted(d.glob("*.npz")):
                if sig.get(f"words\\{d.name}\\{f.name}") in (0, 3):
                    clips.append(("csltr", lab, f))
    return clips


def all_clips(labels):
    """Every keypoint file of every app sign (training clips included) for the pipeline check."""
    out = []
    for src in ("include", "cislr", "islrtc"):
        for lab in labels:
            d = ROOTS[src] / lab
            out += [(src, lab, f) for f in sorted(d.glob("*.npz"))] if d.exists() else []
    lower = {l.lower(): l for l in labels}
    for d in sorted((ROOTS["csltr"] / "words").iterdir()):
        lab = CSLTR_ALIAS.get(d.name) or lower.get(d.name)
        if lab:
            out += [("csltr", lab, f) for f in sorted(d.glob("*.npz"))]
    return out


def train_counts():
    keys = np.load(DATA / "frames_cache.npz", allow_pickle=True)["paths"].tolist()
    per = defaultdict(Counter)
    for k in keys:
        per[k.split("/")[1]][k.split("/")[0]] += 1
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DATA / "model.pt"))
    ap.add_argument("--scenario", default="general", choices=["general", "hospital"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--unseen-csltr", action="store_true", help="add CSLTR signers 0,3 (models trained without them)")
    ap.add_argument("--pipeline", action="store_true", help="also run the segmenter check on every clip")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rec = Recognizer(args.model, device=args.device)
    rec.active = scenario_vocab(args.scenario)
    labels = sorted(rec.sign_labels())
    ck = torch.load(args.model.split(",")[0], map_location="cpu", weights_only=False)
    tc = train_counts()
    tag = args.tag or Path(args.model.split(",")[0]).stem
    out_dir = DATA / "sign_test"
    out_dir.mkdir(exist_ok=True)

    # ---- accuracy on held-out clips, through the server path ---------------------------------------
    clips = [c for c in held_out(ck, args.unseen_csltr) if c[1] in rec.active]
    per = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0, "exact": 0, "silent": 0, "conf": Counter(), "by_src": Counter()})
    t0 = time.time()
    for i, (src, lab, p) in enumerate(clips, 1):
        fps = float(np.load(p)["fps"]) or 25.0
        signs = server_signs(rec, E.frames_of(p), fps)
        r = per[lab]
        r["n"] += 1
        r["by_src"][src] += 1
        if not signs:
            r["silent"] += 1
            r["conf"]["(nothing)"] += 1
            continue
        best = max(signs, key=lambda t: t[0]["p"])
        top = [t["label"] for t in best]
        r["top1"] += top[0] == lab
        r["top3"] += lab in top
        r["exact"] += len(signs) == 1 and signs[0][0]["label"] == lab
        if top[0] != lab:
            r["conf"][top[0]] += 1
        if i % 100 == 0:
            print(f"  [{i}/{len(clips)}] {time.time() - t0:.0f}s", flush=True)
    n = sum(r["n"] for r in per.values())
    acc = {l: r["top1"] / r["n"] for l, r in per.items()}
    summ = {"model": args.model, "scenario": args.scenario, "vocab": len(rec.active), "clips": n, "signs_tested": len(per),
            "signs_untested": sorted(set(labels) - set(per)),
            "top1": sum(r["top1"] for r in per.values()) / max(1, n), "top3": sum(r["top3"] for r in per.values()) / max(1, n),
            "exact": sum(r["exact"] for r in per.values()) / max(1, n), "silent": sum(r["silent"] for r in per.values()) / max(1, n),
            "macro_top1": float(np.mean(list(acc.values()))) if acc else 0.0,
            "signs_ge90": sum(a >= 0.9 for a in acc.values()), "signs_zero": sorted(l for l, a in acc.items() if a == 0)}
    print(f"\n{tag}: {n} held-out clips, {len(per)} signs tested, {len(summ['signs_untested'])} without any held-out clip"
          f" | top-1 {summ['top1']:.1%} top-3 {summ['top3']:.1%} exact-one-sign {summ['exact']:.1%}"
          f" silent {summ['silent']:.1%} | macro {summ['macro_top1']:.1%} | signs >= 90%: {summ['signs_ge90']}/{len(per)}")
    print(f"{'sign':16s} {'n':>3s} {'top1':>5s} {'top3':>5s} {'exact':>5s} {'train':>5s}  confused with")
    for l, r in sorted(per.items(), key=lambda x: (x[1]["top1"] / x[1]["n"], -x[1]["n"])):
        if r["top1"] == r["n"]:
            continue
        print(f"{l:16s} {r['n']:3d} {r['top1'] / r['n']:5.0%} {r['top3'] / r['n']:5.0%} {r['exact'] / r['n']:5.0%} "
              f"{sum(tc[l].values()):5d}  {dict(r['conf'].most_common(3))}")
    report = {"summary": summ, "per_sign": {l: {**{k: v for k, v in r.items() if k not in ("conf", "by_src")},
                                               "confused_with": dict(r["conf"].most_common()), "by_source": dict(r["by_src"]),
                                               "train_clips": dict(tc[l])} for l, r in sorted(per.items())},
              "untested_train_clips": {l: dict(tc[l]) for l in summ["signs_untested"]}}

    # ---- pipeline check on every clip -----------------------------------------------------------------
    if args.pipeline:
        pc = defaultdict(lambda: {"n": 0, "silent": 0, "split": 0})
        allc = all_clips(labels)
        t0 = time.time()
        for i, (src, lab, p) in enumerate(allc, 1):
            fps = float(np.load(p)["fps"]) or 25.0
            k = len(stretches(E.frames_of(p), fps))
            r = pc[f"{lab}|{src}"]
            r["n"] += 1
            r["silent"] += k == 0
            r["split"] += k > 1
            if i % 1000 == 0:
                print(f"  pipeline [{i}/{len(allc)}] {time.time() - t0:.0f}s", flush=True)
        tot = Counter()
        sign_tot = defaultdict(Counter)
        for key, r in pc.items():
            lab, src = key.split("|")
            for k2 in ("n", "silent", "split"):
                tot[(src, k2)] += r[k2]
                sign_tot[lab][k2] += r[k2]
        print("\nPIPELINE (segmenter only, every clip):")
        for src in ("include", "cislr", "islrtc", "csltr"):
            if tot[(src, "n")]:
                print(f"  {src:8s} {tot[(src, 'n')]:5d} clips | silent {tot[(src, 'silent')] / tot[(src, 'n')]:.1%}"
                      f" | split {tot[(src, 'split')] / tot[(src, 'n')]:.1%}")
        bad = sorted(((r["silent"] / r["n"], l, dict(r)) for l, r in sign_tot.items() if r["silent"]), reverse=True)
        print("  signs with silent clips:", [(l, f"{s:.0%}", r["n"]) for s, l, r in bad[:30]])
        report["pipeline"] = {"per_source": {src: {k: tot[(src, k)] for k in ("n", "silent", "split")}
                                             for src in ("include", "cislr", "islrtc", "csltr")},
                              "per_sign": {l: dict(r) for l, r in sorted(sign_tot.items())}}
    path = out_dir / f"{tag}_{args.scenario}.json"
    path.write_text(json.dumps(report, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
