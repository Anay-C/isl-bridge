"""Focus the iSign download on the signs that need data: hospital words and every sign with few training clips get up
to 40 clips, well-covered signs 8. Short texts first (the plan is already ordered short-first per sign)."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(r"C:\Meta Hackathon")
plan = json.loads((ROOT / "data" / "isign" / "plan.json").read_text(encoding="utf-8"))
idx = json.loads((ROOT / "data" / "isign" / "zip_index.json").read_text())
v = json.loads((ROOT / "ml" / "vocab.json").read_text())
keys = np.load(ROOT / "data" / "frames_cache.npz", allow_pickle=True)["paths"].tolist()
have = Counter(k.split("/")[1] for k in keys)
weak = set(v["cislr_signs"]) | {s for s in v["include_signs"] + v.get("common_signs", []) if have[s] <= 12}
quota = {s: (40 if s in weak else 8) for s in v["include_signs"] + v["cislr_signs"] + v.get("common_signs", [])}
items = sorted(plan.items(), key=lambda kv: len(kv[1]["text"].split()))
got, keep = Counter(), {}
for uid, p in items:
    need = [s for s in p["signs"] if got[s] < quota.get(s, 0)]
    if need:
        keep[uid] = p
        for s in p["signs"]:
            got[s] += 1
gb = sum(idx[p["file"]][1] for p in keep.values()) / 1e9
(ROOT / "data" / "isign" / "plan_focus.json").write_text(json.dumps(keep, ensure_ascii=False))
print(f"focused plan: {len(keep)} clips ({gb:.2f} GB) of {len(plan)}; weak signs covered: "
      f"{sum(1 for s in weak if got[s])}/{len(weak)}; weak signs with >= 10 clips: {sum(1 for s in weak if got[s] >= 10)}")
print("weak signs and clip counts:", sorted(((s, got[s]) for s in weak), key=lambda x: x[1])[:60])
