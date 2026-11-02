"""List the sign words inside every INCLUDE zip on Zenodo without downloading them (HTTP range requests)."""
import io, json, re, ssl, sys, time, urllib.error, urllib.request, zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

API = "https://zenodo.org/api/records/4010759"
CTX = ssl.create_default_context()


class HttpFile:
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        for attempt in range(8):
            try:
                data = urllib.request.urlopen(req, context=CTX, timeout=60).read()
                break
            except urllib.error.HTTPError as e:
                if e.code != 429: raise
                time.sleep(5 * (attempt + 1))
        self.pos += len(data)
        return data


def list_zip(f):
    hf = HttpFile(f["links"]["self"], f["size"])
    names = zipfile.ZipFile(hf).namelist()
    return f["key"], [n for n in names if n.lower().endswith((".mov", ".mp4", ".avi", ".mkv"))]


def main():
    files = json.load(open("data/zenodo_record.json"))["files"]
    zips = [f for f in files if f["key"].endswith(".zip")]
    words = defaultdict(lambda: defaultdict(int))
    with ThreadPoolExecutor(1) as ex:
        for key, vids in ex.map(list_zip, zips):
            cat = key.rsplit("_", 1)[0]
            for v in vids:
                parts = v.replace("\\", "/").split("/")
                label = re.sub(r"^\d+\.\s*", "", parts[-2]).strip()
                words[cat][label] += 1
            print(f"{key}: {len(vids)} videos", file=sys.stderr)
    out = {c: dict(sorted(w.items())) for c, w in sorted(words.items())}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for c, w in out.items():
        print(f"\n## {c} ({len(w)} signs, {sum(w.values())} videos)")
        print(", ".join(f"{k}({v})" for k, v in w.items()))


if __name__ == "__main__":
    main()
