"""Summarise eval_live reports: words top-1/top-3, sentence recall, detections, precision; mean per group."""
import json, sys, re
from collections import defaultdict
groups = defaultdict(list)
for path in sys.argv[1:]:
    d = json.load(open(path))
    w, s, rows = d.get("words", {}), d.get("sentences", {}), d.get("rows", [])
    det = sum(len(r["detected"]) for r in rows)
    ins = sum(sum(x in r["targets"] for x in r["detected"]) for r in rows)
    m = (w.get("top1", 0), w.get("top3", 0), s.get("recall_top1", 0), s.get("recall_top3", 0), ins / max(1, det))
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    groups[re.sub(r"_s\d+\.json$|\.json$", "", name)].append(m)
    print(f"{name:34s} words {m[0]:.1%}/{m[1]:.1%} | sentences recall {m[2]:.1%}/{m[3]:.1%} | det {det:3d} prec {m[4]:.0%}")
for g, ms in groups.items():
    if len(ms) > 1:
        mean = [sum(x[i] for x in ms) / len(ms) for i in range(5)]
        print(f"MEAN {g:29s} words {mean[0]:.1%}/{mean[1]:.1%} | sentences recall {mean[2]:.1%}/{mean[3]:.1%} | prec {mean[4]:.0%}  (n={len(ms)})")
