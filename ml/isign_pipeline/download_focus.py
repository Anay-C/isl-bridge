import json, sys
sys.path.insert(0, r"C:\Meta Hackathon\ml")
import fetch_isign as F
plan = json.load(open(r"C:\Meta Hackathon\data\isign\plan_focus.json", encoding="utf-8"))
F.download(plan, F.zip_index(), workers=6)
