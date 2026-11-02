"""Tester data on a public deployment (Hugging Face Space): keep it in a PRIVATE Hugging Face dataset.

A Space's disk is wiped on every restart, so without this the testers' taught signs, their mistake reports, the
keypoint logs (only for testers who left "help improve" on) and the spend ledger would vanish, and a restart
would also reset the spend cap. Off unless LOG_REPO is set (the laptop keeps everything in data/ as before).

    LOG_REPO=AnayChauhan/isl-bridge-testlogs   private dataset (created if missing)
    HF_TOKEN=...                               write access to it (a Space secret)
    LOG_PREFIX=space                           folder inside the dataset
    LOG_SYNC_MIN=10                            minutes between uploads

restore() runs once at start-up (profiles + ledger come back); start() then uploads every LOG_SYNC_MIN minutes
(huggingface_hub.CommitScheduler: a background thread, only changed files, never blocks a request).
"""
import os
import shutil
import tempfile
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
REPO = os.environ.get("LOG_REPO", "")
TOKEN = os.environ.get("HF_TOKEN", "")
PREFIX = os.environ.get("LOG_PREFIX", "space")
EVERY_MIN = float(os.environ.get("LOG_SYNC_MIN", "10"))
PATTERNS = ["users/*.json", "feedback/**", "live_log/**", "recordings/**", "meta_spend.json"]
_scheduler = None


def enabled():
    return bool(REPO and TOKEN)


def restore():
    """Bring back the testers' profiles and the spend ledger written before the last restart."""
    if not enabled():
        return
    try:
        from huggingface_hub import snapshot_download
        tmp = Path(tempfile.mkdtemp(prefix="isl-restore-"))
        snapshot_download(REPO, repo_type="dataset", token=TOKEN, local_dir=tmp,
                          allow_patterns=[f"{PREFIX}/users/*.json", f"{PREFIX}/meta_spend.json"])
        src = tmp / PREFIX
        n = 0
        for f in (src / "users").glob("*.json") if (src / "users").is_dir() else []:
            dst = DATA / "users" / f.name
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                n += 1
        led = src / "meta_spend.json"
        if led.exists() and not (DATA / "meta_spend.json").exists():
            shutil.copy2(led, DATA / "meta_spend.json")
        print(f"[sync] restored {n} profiles + ledger from {REPO}", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:  # first start (empty dataset) or HF down: start fresh, never block the app
        print(f"[sync] restore skipped: {type(e).__name__}: {str(e)[:200]}", flush=True)


def start():
    global _scheduler
    if not enabled() or _scheduler is not None:
        return
    try:
        from huggingface_hub import CommitScheduler
        _scheduler = CommitScheduler(repo_id=REPO, repo_type="dataset", private=True, token=TOKEN,
                                     folder_path=DATA, path_in_repo=PREFIX, every=EVERY_MIN,
                                     allow_patterns=PATTERNS)
        print(f"[sync] uploading tester data to {REPO}/{PREFIX} every {EVERY_MIN:g} min", flush=True)
    except Exception as e:
        print(f"[sync] disabled: {type(e).__name__}: {str(e)[:200]}", flush=True)


def flush():
    """Upload now (e.g. before a planned restart)."""
    if _scheduler is not None:
        try:
            _scheduler.trigger().result(timeout=120)
        except Exception as e:
            print(f"[sync] flush failed: {e!r}"[:300], flush=True)
