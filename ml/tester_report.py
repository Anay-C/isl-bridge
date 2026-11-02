"""What did testers run into? Summary of everything the public link collected in data/.

    .venv-ml/Scripts/python ml/tester_report.py              # testers only (profiles t-*)
    .venv-ml/Scripts/python ml/tester_report.py --all        # include anay / test users
    .venv-ml/Scripts/python ml/tester_report.py --since 2026-11-03

Reads data/feedback/<user>.jsonl ("✏️ Wrong?" reports + feedback notes), data/feedback/<user>/*.npz (keypoints of
corrected signs = labelled live data for training/eval), data/live_log/<user>/ (every sign of testers who left
"help improve" on) and data/users/<user>.json (taught signs). Prints the most common confusions first.
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
TEST_USERS = ("wstest", "autotest")


def is_tester(user, everyone):
    return everyone or (user.startswith("t-") and not user.startswith(TEST_USERS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every profile, not just testers (t-*)")
    ap.add_argument("--since", default="", help="YYYY-MM-DD: only reports from this day on")
    a = ap.parse_args()

    rows = []
    for f in sorted((DATA / "feedback").glob("*.jsonl")):
        if not is_tester(f.stem, a.all):
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not a.since or r.get("time", "") >= a.since:
                rows.append(r)
    reports = [r for r in rows if r.get("fixes")]  # "✏️ Wrong?" with corrected signs
    notes = [r for r in rows if r.get("note")]  # free text (feedback sheet, or added to a report)

    users = defaultdict(lambda: {"signs": 0, "first": "", "last": "", "taught": 0, "reports": 0})
    for d in (DATA / "live_log").glob("*"):
        if d.is_dir() and is_tester(d.name, a.all):
            fs = sorted(d.glob("*.npz"))
            if fs:
                u = users[d.name]
                u["signs"] = len(fs)
                u["first"], u["last"] = fs[0].stem[:13], fs[-1].stem[:13]
    for f in (DATA / "users").glob("*.json"):
        if is_tester(f.stem, a.all):
            try:
                prof = json.loads(f.read_text(encoding="utf-8"))
                users[f.stem]["taught"] = sum(1 for k in prof if not k.startswith("_"))
            except (OSError, ValueError):
                pass
    for r in rows:
        users[r["user"]]["reports"] += 1

    print(f"Testers: {len(users)}  |  '✏️ Wrong?' reports: {len(reports)}  |  notes: {len(notes)}")
    try:
        led = json.loads((DATA / "meta_spend.json").read_text())
        print(f"Meta spend so far (laptop ledger): ${led.get('_total_usd', 0):.2f}")
    except (OSError, ValueError):
        pass
    if users:
        print("\nprofile        signs logged  taught  reports  active (first -> last)")
        for name, u in sorted(users.items(), key=lambda x: -x[1]["signs"]):
            print(f"  {name:<14} {u['signs']:>6}  {u['taught']:>9}  {u['reports']:>7}  {u['first']} -> {u['last']}")

    pairs = Counter()
    for r in reports:
        for f in r["fixes"]:
            pairs[(f.get("said", "?"), f.get("meant", "?"))] += 1
    if pairs:
        print("\nMost common mistakes (read as -> really signed):")
        for (said, meant), n in pairs.most_common(25):
            print(f"  {n:>3} x  {said:<16} -> {meant}")
        meant = Counter(f["meant"] for r in reports for f in r["fixes"])
        print("\nSigns testers had to correct most:", ", ".join(f"{m} ({n})" for m, n in meant.most_common(12)))
        kp = sum(len(r.get("keypoints", [])) for r in reports)
        print(f"Labelled live clips saved (keypoints of corrected signs): {kp}  -> data/feedback/<user>/*.npz")
    if notes:
        print("\nNotes from testers (newest last):")
        for r in notes[-30:]:
            print(f"  [{r.get('time', '')[:16]} {r['user']}] {r['note'][:300]}")
    if not (users or reports or notes):
        print("\nNothing from testers yet. Share the link printed by deploy/share.py (also in data/share_url.txt).")


if __name__ == "__main__":
    main()
