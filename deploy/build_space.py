"""Build and deploy ISL Bridge to a public Hugging Face Space (Docker SDK, free CPU), so testers get one https link
that works without the laptop.

    .venv-ml/Scripts/python deploy/build_space.py --dry-run   # build the web app + stage files, show sizes
    .venv-ml/Scripts/python deploy/build_space.py             # ... + upload, set secrets, wait until running

What goes up (data/space_build/, ~115 MB): the runtime code (server/, the ml/ modules the server imports), the
built web app, the live model, the sign clips + SAM cut-outs + picture cards + cached dictionary clips, and the
227 reference recordings the ghost coach draws. Nothing else from data/ (no training data, no user logs).

Secrets come from .env and are never printed: MODEL_API_KEY (Meta) and HF_TOKEN (tester data -> the PRIVATE
dataset LOG_REPO, see server/sync.py). Spend: the Space may spend META_SPEND_CAP_USD - META_SPEND_BASE_USD
($50 - $20 = $30); the laptop's .env caps its own ledger at $20, so both together stay within Anay's $50.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STAGE = DATA / "space_build"
HERE = Path(__file__).resolve().parent / "hf"
# not "isl-bridge": that free static Space is the fixed tester address (deploy/forward/, updated by share.py).
# Once this app Space runs, point testers at it (or run share.py with --no-forward and edit deploy/forward/).
SPACE = os.environ.get("SPACE", "AnayChauhan/isl-bridge-app")
LOG_REPO = os.environ.get("LOG_REPO", "AnayChauhan/isl-bridge-testlogs")
CAP, BASE = os.environ.get("SPACE_CAP_USD", "50"), os.environ.get("SPACE_BASE_USD", "20")

SERVER = ["app.py", "muse.py", "dictionary.py", "pictures.py", "segmenter.py", "sync.py"]
ML = ["recognizer.py", "model.py", "features.py", "spotting.py", "coach.py", "vocab.json"]
CLIP_JSON = ["sources.json", "ghost_sources.json", "descriptions.json"]
MEDIA_DIRS = {"clips_sam": "*.mp4", "clips_dict": "*.mp4", "pictures": "*.png"}


def host():
    owner, name = SPACE.split("/")
    return f"{owner.lower()}-{name.lower().replace('_', '-')}.hf.space"


def version():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    return f"{sha or 'nogit'}{'+' if dirty else ''}-{time.strftime('%m%d-%H%M')}"


def build_web():
    print("building the web app ...", flush=True)
    subprocess.run("npm run build", cwd=ROOT / "web", shell=True, check=True)


def stage(ver):
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for sub, names in (("server", SERVER), ("ml", ML)):
        (STAGE / sub).mkdir(parents=True)
        for n in names:
            shutil.copy2(ROOT / sub / n, STAGE / sub / n)
    shutil.copytree(ROOT / "web" / "dist", STAGE / "web" / "dist")
    d = STAGE / "data"
    (d / "clips").mkdir(parents=True)
    shutil.copy2(DATA / "model.pt", d / "model.pt")
    for p in (DATA / "clips").glob("*.mp4"):
        shutil.copy2(p, d / "clips" / p.name)
    for n in CLIP_JSON:
        if (DATA / "clips" / n).exists():
            shutil.copy2(DATA / "clips" / n, d / "clips" / n)
    for sub, pat in MEDIA_DIRS.items():
        (d / sub).mkdir()
        for p in (DATA / sub).glob(pat):
            shutil.copy2(p, d / sub / p.name)
    shutil.copy2(DATA / "clips_sam" / "boxes.json", d / "clips_sam" / "boxes.json")
    for n in ("islrtc_hf_files.json", "islrtc_hf_sizes.json"):
        shutil.copy2(DATA / n, d / n)
    # the ghost coach's reference recordings: exactly the file ml/coach.py picks for each clip
    sys.path.insert(0, str(ROOT / "ml"))
    import coach
    refs = [coach._source(p.stem) for p in sorted((DATA / "clips").glob("*.mp4"))]
    for src in [r for r in refs if r is not None]:
        dst = d / src.relative_to(DATA)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    shutil.copy2(HERE / "Dockerfile", STAGE / "Dockerfile")
    shutil.copy2(HERE / "requirements.txt", STAGE / "requirements.txt")
    readme = (HERE / "README.md").read_text(encoding="utf-8").replace("{SPACE_HOST}", host()).replace("{VERSION}", ver)
    (STAGE / "README.md").write_text(readme, encoding="utf-8")
    files = [p for p in STAGE.rglob("*") if p.is_file()]
    mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"staged {len(files)} files, {mb:.0f} MB ({sum(r is not None for r in refs)} ghost references) in {STAGE}")


def deploy(ver):
    from dotenv import dotenv_values
    from huggingface_hub import HfApi
    env = dotenv_values(ROOT / ".env")
    token, key = env.get("HF_TOKEN") or os.environ.get("HF_TOKEN"), env.get("MODEL_API_KEY")
    if not token or not key:
        sys.exit("HF_TOKEN and MODEL_API_KEY must be in .env")
    api = HfApi(token=token)
    api.create_repo(SPACE, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.create_repo(LOG_REPO, repo_type="dataset", private=True, exist_ok=True)
    api.add_space_secret(SPACE, "MODEL_API_KEY", key, description="Meta Model API (Muse Spark / Voice / Image)")
    api.add_space_secret(SPACE, "HF_TOKEN", token, description="writes tester data to the private dataset")
    for k, v in {"LOG_REPO": LOG_REPO, "META_SPEND_CAP_USD": CAP, "META_SPEND_BASE_USD": BASE,
                 "ISL_VERSION": ver}.items():
        api.add_space_variable(SPACE, k, v)
    print(f"uploading to https://huggingface.co/spaces/{SPACE} ...", flush=True)
    t0 = time.time()
    api.upload_folder(repo_id=SPACE, repo_type="space", folder_path=STAGE, commit_message=f"Deploy {ver}",
                      delete_patterns=["server/**", "ml/**", "web/**", "data/**"])
    print(f"uploaded in {time.time() - t0:.0f} s; waiting for the build ...", flush=True)
    wait(api)


def wait(api, minutes=25):
    last = None
    for _ in range(minutes * 4):
        rt = api.get_space_runtime(SPACE)
        if rt.stage != last:
            print(f"  {time.strftime('%H:%M:%S')} {rt.stage}", flush=True)
            last = rt.stage
        if rt.stage == "RUNNING":
            print(f"\nRUNNING: https://{host()}  (Space page: https://huggingface.co/spaces/{SPACE})")
            return True
        if rt.stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"):
            print(f"FAILED: {rt.stage} {getattr(rt, 'raw', {}).get('errorMessage', '')}")
            print(f"build logs: https://huggingface.co/spaces/{SPACE}?logs=build")
            sys.exit(1)
        time.sleep(15)
    print("still not running - check the Space page")
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build + stage only, upload nothing")
    ap.add_argument("--no-web", action="store_true", help="use the existing web/dist")
    ap.add_argument("--wait", action="store_true", help="only wait for the current build")
    a = ap.parse_args()
    if a.wait:
        from dotenv import dotenv_values
        from huggingface_hub import HfApi
        wait(HfApi(token=dotenv_values(ROOT / ".env").get("HF_TOKEN")))
        sys.exit(0)
    v = version()
    if not a.no_web:
        build_web()
    stage(v)
    if not a.dry_run:
        deploy(v)
