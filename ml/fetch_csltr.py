"""Download the ISL-CSLTR corpus videos (Elakkiya & Natarajan, Mendeley Data kcmpdxky7p, CC BY 4.0) out of the
single 8.9 GB remote zip, video files only -> data/csltr/videos/{words,sentences}/<label>/<file>

    python ml/fetch_csltr.py

7 signers (Navajeevan school for the Deaf), 101 everyday sentences: 292 sentence videos + 395 word videos.
Used as an unseen-signer test set (words) and a live-pipeline test (continuous sentences). Resumable.
"""
import json
import re
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download import HttpFile  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "csltr" / "videos"
API = "https://data.mendeley.com/public-api/datasets/kcmpdxky7p/files?folder_id=root&version=1"
VIDEO = (".mp4", ".mov", ".avi", ".mkv")


def signed_url():
    meta = json.loads(subprocess.run(["curl", "-s", "--ssl-no-revoke", API], capture_output=True, text=True).stdout)
    head = subprocess.run(["curl", "-sI", "--ssl-no-revoke", meta[0]["content_details"]["download_url"]],
                          capture_output=True, text=True).stdout
    return next(l.split(" ", 1)[1].strip() for l in head.splitlines() if l.lower().startswith("location:"))


def size_of(url):
    head = subprocess.run(["curl", "-sI", "--ssl-no-revoke", "-r", "0-0", url], capture_output=True, text=True).stdout
    return int(head.split("Content-Range: bytes 0-0/")[1].split()[0])


def dest(name):
    p = PurePosixPath(name)
    folder = p.parent.name
    stem = re.sub(r"\s*\(\d+\)\s*$", "", p.stem).strip()
    kind = "sentences" if stem.lower() == folder.lower() else "words"
    label = folder if kind == "sentences" else stem.lower()
    # keep the sentence folder in the word file name: the same word recorded for different sentences
    return OUT / kind / label / (re.sub(r"[^\w() .-]+", "_", f"{folder}__{p.name}") if kind == "words" else p.name)


def main():
    url = signed_url()
    z = zipfile.ZipFile(HttpFile(url, size_of(url)))
    todo = [i for i in z.infolist() if i.filename.lower().endswith(VIDEO) and not dest(i.filename).exists()]
    print(f"{len(todo)} videos to fetch ({sum(i.compress_size for i in todo) / 1e9:.1f} GB)", flush=True)

    def fetch(i):
        d = dest(i.filename)
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = d.with_suffix(d.suffix + ".part")
        zz = zipfile.ZipFile(HttpFile(url, z.fp.size if hasattr(z.fp, "size") else size_of(url)))
        with zz.open(i) as src, tmp.open("wb") as out:
            shutil.copyfileobj(src, out, 1 << 20)
        tmp.replace(d)
        return d

    with ThreadPoolExecutor(4) as ex:
        for k, d in enumerate(ex.map(fetch, todo), 1):
            if k % 20 == 0 or k == len(todo):
                print(f"[{k}/{len(todo)}] {d.parent.name}/{d.name}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
