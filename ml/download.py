"""Download INCLUDE videos straight out of the remote Zenodo zips using HTTP range requests.

Only the videos we ask for are fetched (no 56.8 GB of full zips). Resumable: finished files are skipped.

    python ml/download.py            # demo vocabulary from ml/vocab.json
    python ml/download.py --all      # every sign (demo vocabulary first)
"""
import argparse
import json
import re
import ssl
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RECORD = DATA / "zenodo_record.json"
INDEX = DATA / "zip_index.json"
OUT = DATA / "videos"
VIDEO_EXT = (".mov", ".mp4", ".avi", ".mkv")
CTX = ssl.create_default_context()


def http_range(url, start, end):
    """Bytes [start, end] inclusive, retrying on throttling / transient errors."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(10):
        try:
            with urllib.request.urlopen(req, context=CTX, timeout=300) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"giving up on {url} [{start}-{end}]")


class HttpFile:
    """Minimal seekable file over HTTP so zipfile can read the central directory."""

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        data = http_range(self.url, self.pos, min(self.pos + n, self.size) - 1)
        self.pos += len(data)
        return data


def label_of(member):
    """Label = parent folder name with leading 'N. ' stripped."""
    return re.sub(r"^\d+\.\s*", "", member.replace("\\", "/").split("/")[-2]).strip()


def build_index():
    if INDEX.exists():
        return json.loads(INDEX.read_text())
    files = [f for f in json.loads(RECORD.read_text())["files"] if f["key"].endswith(".zip")]
    index = []
    for f in files:
        zf = zipfile.ZipFile(HttpFile(f["links"]["self"], f["size"]))
        for info in zf.infolist():
            if info.filename.lower().endswith(VIDEO_EXT):
                index.append({
                    "zip": f["key"], "url": f["links"]["self"], "member": info.filename,
                    "label": label_of(info.filename), "offset": info.header_offset,
                    "csize": info.compress_size, "size": info.file_size,
                    "method": info.compress_type, "crc": info.CRC,
                })
        print(f"indexed {f['key']}", flush=True)
    INDEX.write_text(json.dumps(index))
    return index


def dest_of(e):
    stem = Path(e["member"].replace("\\", "/")).name
    return OUT / e["label"] / f"{e['zip'][:-4]}__{stem}"


def fetch(e):
    dest = dest_of(e)
    if dest.exists() and dest.stat().st_size == e["size"]:
        return "skip", e
    hdr = http_range(e["url"], e["offset"], e["offset"] + 29)
    name_len, extra_len = struct.unpack("<HH", hdr[26:30])
    start = e["offset"] + 30 + name_len + extra_len
    blob = http_range(e["url"], start, start + e["csize"] - 1) if e["csize"] else b""
    if e["method"] == zipfile.ZIP_DEFLATED:
        blob = zlib.decompress(blob, -15)
    elif e["method"] != zipfile.ZIP_STORED:
        raise RuntimeError(f"unsupported compression {e['method']} for {e['member']}")
    if zlib.crc32(blob) != e["crc"]:
        raise RuntimeError(f"CRC mismatch for {e['member']}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(blob)
    tmp.replace(dest)
    return "ok", e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="download every sign (demo vocab first)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    vocab = set(json.loads((ROOT / "ml" / "vocab.json").read_text())["include_signs"])
    index = build_index()
    known = {e["label"] for e in index}
    missing = vocab - known
    if missing:
        print("WARNING: vocab words not in INCLUDE:", sorted(missing))

    todo = [e for e in index if args.all or e["label"] in vocab]
    todo.sort(key=lambda e: (e["label"] not in vocab, e["zip"], e["member"]))
    total = sum(e["size"] for e in todo)
    print(f"{len(todo)} videos, {total / 1e9:.1f} GB ({len({e['label'] for e in todo})} signs)", flush=True)

    done_bytes, t0, n = 0, time.time(), 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, e) for e in todo]
        for fut in as_completed(futs):
            n += 1
            try:
                status, e = fut.result()
            except Exception as err:  # keep going; rerun picks up failures
                print(f"[{n}/{len(todo)}] FAILED: {err}", flush=True)
                continue
            done_bytes += e["size"]
            rate = done_bytes / 1e6 / max(1e-6, time.time() - t0)
            eta = (total - done_bytes) / 1e6 / max(rate, 1e-6) / 60
            if status == "ok" or n % 50 == 0:
                print(f"[{n}/{len(todo)}] {status} {e['label']}/{Path(e['member']).name} "
                      f"{done_bytes / 1e9:.2f}/{total / 1e9:.1f} GB  {rate:.1f} MB/s  ETA {eta:.0f} min", flush=True)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
