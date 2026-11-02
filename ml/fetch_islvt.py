"""Fetch ISLVT (Indian Sign Language Video and Text dataset for sentences, Mendeley Data 10.17632/98mzk82wbb.1,
CC BY 4.0): 76 everyday sentences signed by one professional signer from Maharashtra, two takes each; file names are
the ISL gloss in signing order ("31 food taste good.MOV"). Only clips that contain one of our signs are downloaded
-> data/islvt/videos/<file>. Resumable. (Mendeley's API needs curl: Python's urllib gets HTTP 403.)

    python ml/fetch_islvt.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from fetch_isign import sign_matcher  # noqa: E402

API = "https://data.mendeley.com/public-api/datasets/98mzk82wbb/files?folder_id=root&version=1"
OUT = ROOT / "data" / "islvt" / "videos"


def main():
    files = json.loads(subprocess.run(["curl", "-s", "--ssl-no-revoke", API], capture_output=True, text=True).stdout)
    v = json.loads((ROOT / "ml" / "vocab.json").read_text())
    match = sign_matcher(v["include_signs"] + v["cislr_signs"] + v.get("common_signs", []))
    # "(2)" / "... 2.MOV" takes are filmed from the side (shoulders 5% of the frame wide): useless for a frontal model
    side = lambda n: "(2)" in n or re.search(r" 2 ?\.mov$", n.lower())
    todo = [f for f in files if f["filename"].lower().endswith(".mov") and not side(f["filename"])
            and match(f["filename"].rsplit(".", 1)[0])]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(todo)} of {len(files)} files contain our signs ({sum(f['size'] for f in todo) / 1e9:.2f} GB)",
          flush=True)
    def one(f):
        dst = OUT / f["filename"].strip()
        if dst.exists() and dst.stat().st_size == f["size"]:
            return
        tmp = dst.with_suffix(".part")
        r = subprocess.run(["curl", "-s", "-L", "--ssl-no-revoke", "--retry", "5", "--retry-all-errors", "-o", str(tmp),
                            f["content_details"]["download_url"]])
        if r.returncode or not tmp.exists() or tmp.stat().st_size != f["size"]:
            print(f"failed {f['filename']} (curl {r.returncode}); run again to retry", flush=True)
            return
        if tmp.stat().st_size != f["size"]:
            print(f"size mismatch {f['filename']}: {tmp.stat().st_size} != {f['size']}", flush=True)
            return
        tmp.replace(dst)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as ex:  # one Mendeley connection gives ~1 MB/s
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 10 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}]", flush=True)
    (OUT.parent / "files.json").write_text(json.dumps(files, indent=0))


if __name__ == "__main__":
    main()
