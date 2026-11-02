"""Download ISLRTC dictionary clips for our vocabulary from the Hugging Face mirror of the
data.gov.in "Indian Sign Language Dictionary" -> data/islrtc/videos/<Label>/<file>.mp4

ISLRTC terms (islrtc.nic.in/faq Q.23): research / ISL technology use allowed, no resale or profiteering,
acknowledge ISLRTC, DEPwD and the Ministry of Social Justice and Empowerment.

    python ml/fetch_islrtc.py [--dry]
"""
import argparse
import json
import re
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPO = "silentone0725/Indian_Sign_Language_Data.gov_Rencoded"
OUT = DATA / "islrtc" / "videos"
CACHE = DATA / "islrtc" / "raw"

# our label -> ISLRTC file stems, where the names differ
ALIASES = {
    "I": ["I_Me"], "House": ["House_Home"], "Store or Shop": ["Shop"], "cheap": ["Cheap_Inexpensive"],
    "old": ["Old_(Things_Time)", "Old_Aged_Elderly_(Person)"], "big large": ["Big", "Large"],
    "small little": ["Small"], "Week": ["Week_One_Week", "Week_One_Week_(Sign_2)"], "Telephone": ["Phone"],
    "Cell phone": ["Phone"], "When": ["When_(For_Days)"], "Vomit": ["Vomiting"],
    "Tired": ["Tired_Exhausted_Fatigued"], "Understand": ["Understand_Comprehend"],
    "Again": ["Again_Repeat", "Again_Repeat_(Sign_2)"], "Alright": ["Okay", "Okay_(Sign_2)"],
}
# hospital words beyond the trained vocab: speech -> sign clips and one-shot prototypes
EXTRA = ["Chest pain", "Blood pressure", "X-ray", "Bed rest", "Operation", "Surgery", "Wheelchair", "Thermometer",
         "Temperature", "Weight", "Pregnant", "Heart", "Heart attack", "Cancer", "Infection", "Wound", "Bandage",
         "Ointment", "Syrup", "Report", "Test", "Appointment", "Wait", "Sit", "Come", "Go", "Stop", "Pharmacy",
         "Chemist", "Ear", "Eye", "Nose", "Throat", "Tooth", "Teeth", "Hand", "Leg", "Back", "Back pain", "Skin",
         "Itching", "Swelling", "Burn", "Bleeding", "Fracture", "Diarrhoea", "Diarrhea", "Cold", "Sneeze",
         "Breath", "Unconscious", "Faint", "Weakness", "Doctor", "Hospital", "Medicine", "Ambulance", "Nurse",
         "Injection", "Pain", "Fever", "Headache", "Diabetes", "Emergency", "Cough", "Tablet", "Blood"]


def norm(s):
    return re.sub(r"[\s_]+", " ", s).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    files = [f for f in json.loads((DATA / "islrtc_hf_files.json").read_text()) if f.endswith(".mp4")]
    by_stem = {}
    for f in files:
        by_stem.setdefault(norm(Path(f).stem), []).append(f)

    v = json.loads((ROOT / "ml" / "vocab.json").read_text())
    wanted = {}
    for label in dict.fromkeys(v["include_signs"] + v["cislr_signs"] + EXTRA):
        stems = ALIASES.get(label, [label])
        hits = []
        for s in stems:
            n = norm(s)
            hits += by_stem.get(n, [])
            hits += [f for k, fs in by_stem.items() if re.fullmatch(re.escape(n) + r" \(sign \d\)", k) for f in fs]
        if hits:
            wanted[label] = sorted(set(hits))
    missing = [l for l in v["include_signs"] + v["cislr_signs"] if l not in wanted]
    print(f"{len(wanted)} labels, {sum(map(len, wanted.values()))} clips; vocab without ISLRTC clip: {missing}")
    if args.dry:
        return
    for i, (label, fs) in enumerate(wanted.items(), 1):
        for f in fs:
            dst = OUT / label / Path(f).name
            if dst.exists():
                continue
            src = hf_hub_download(REPO, f, repo_type="dataset", local_dir=CACHE)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        print(f"[{i}/{len(wanted)}] {label}: {len(fs)}", flush=True)
    (DATA / "islrtc" / "index.json").write_text(json.dumps(wanted, indent=1))
    print("done")


if __name__ == "__main__":
    main()
