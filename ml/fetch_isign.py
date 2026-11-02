"""Fetch the iSign clips we can learn from, without downloading the whole 58 GB.

iSign (Joshi et al., ACL 2024 Findings; IIT Kanpur Exploration Lab): 118k real-world ISL clips from ISH News, ISLRTC
and Deaf Enabled Foundation YouTube channels (scraped with the channels' permission) + ISLTranslate + CISLR, each
with its English text. Licence CC BY-NC-SA 4.0: research / non-commercial only; credit it. Gated on Hugging Face:
the account behind HF_TOKEN (.env) must have accepted the conditions on the dataset page.

The videos are ONE zip split into two HF files, so this script reads the zip's central directory with HTTP range
requests and pulls only the clips whose English text mentions one of our signs (data/isign/videos/<uid>.mp4).

    python ml/fetch_isign.py --plan              # CSV + zip index -> data/isign/plan.json, prints counts and GB
    python ml/fetch_isign.py --per-sign 40       # plan + download (resumable, 6 parallel range requests)
"""
import argparse
import csv
import io
import json
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "isign"
VIDEOS = OUT / "videos"
REPO = "https://huggingface.co/datasets/Exploration-Lab/iSign/resolve/main/"
PARTS = [("iSign-videos_v1.1_part_aa", 32212254720), ("iSign-videos_v1.1_part_ab", 25686272443)]
TOTAL = sum(s for _, s in PARTS)
CSV_NAME = "iSign_v1.1.csv"
# pointing signs: English pronouns are in almost every sentence and "I"/"you" are already over-predicted
SKIP = {"I", "you", "he", "she", "it", "we", "they", "you (plural)", "this", "that"}
PHRASES = {"thank you": "Thank you", "how are you": "How are you", "good morning": "Good Morning",
           "good afternoon": "Good afternoon", "good evening": "Good evening", "good night": "Good night",
           "cell phone": "Cell phone", "mobile phone": "Cell phone", "train station": "Train Station",
           "railway station": "Train Station", "grand mother": "Grandmother", "grand father": "Grandfather",
           "granfather": "Grandfather"}
SPLIT_LABELS = {"Store or Shop": ("store", "shop"), "big large": ("big", "large"), "small little": ("small", "little"),
                "Street or Road": ("street", "road")}


def _token():
    for line in open(ROOT / ".env", encoding="utf-8"):
        if line.strip().startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("HF_TOKEN missing from .env")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


_signed = {}


def signed_url(name):
    """The resolve URL redirects to a signed CDN URL; cache it (refreshed every 20 min or on 401/403)."""
    hit = _signed.get(name)
    if hit and time.time() - hit[1] < 1200:
        return hit[0]
    req = urllib.request.Request(REPO + name, method="HEAD", headers={"Authorization": f"Bearer {_token()}"})
    try:
        r = urllib.request.build_opener(_NoRedirect).open(req, timeout=60)
        loc = REPO + name if r.status == 200 else r.headers.get("Location")
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise SystemExit(f"{name}: HTTP {e.code} (has the HF account accepted the iSign conditions?)")
        loc = e.headers["Location"]
    if loc.startswith("/"):
        loc = "https://huggingface.co" + loc
    _signed[name] = (loc, time.time())
    return loc


def get_range(name, start, end):
    """Bytes [start, end] (inclusive) of one HF file."""
    for attempt in range(7):
        url = signed_url(name)
        hdr = {"Range": f"bytes={start}-{end}"}
        if url.startswith(REPO):
            hdr["Authorization"] = f"Bearer {_token()}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=180) as r:
                if r.status != 206:  # never read a whole 30 GB body by accident
                    raise IOError(f"expected 206, got {r.status}")
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read {len(data)} of {end - start + 1}")
            return data
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                _signed.pop(name, None)
            err = e
        except Exception as e:  # network blip
            err = e
        time.sleep(min(60, 2 ** attempt))
    raise IOError(f"{name} [{start}-{end}]: {err}")


def read_global(off, n):
    """n bytes at offset `off` of the virtual zip = part_aa + part_ab."""
    out = []
    while n > 0:
        i = 0 if off < PARTS[0][1] else 1
        base = 0 if i == 0 else PARTS[0][1]
        take = min(n, PARTS[i][1] - (off - base))
        out.append(get_range(PARTS[i][0], off - base, off - base + take - 1))
        off, n = off + take, n - take
    return b"".join(out)


def zip_index():
    """Central directory of the remote zip -> {member name: [local header offset, compressed size, method]}."""
    path = OUT / "zip_index.json"
    if path.exists():
        return json.loads(path.read_text())
    tail_n = 65536 + 22
    tail = read_global(TOTAL - tail_n, tail_n)
    p = tail.rfind(b"PK\x05\x06")
    if p < 0:
        raise SystemExit("zip end record not found")
    _, _, _, _, n_entries, cd_size, cd_off, _ = struct.unpack("<4s4H2LH", tail[p:p + 22])
    if 0xFFFFFFFF in (cd_size, cd_off) or n_entries == 0xFFFF:  # zip64
        loc = tail[p - 20:p]
        assert loc[:4] == b"PK\x06\x07", "zip64 locator missing"
        e64_off = struct.unpack("<Q", loc[8:16])[0]
        e64 = read_global(e64_off, 56)
        assert e64[:4] == b"PK\x06\x06"
        n_entries, cd_size, cd_off = struct.unpack("<3Q", e64[32:56])
    print(f"zip: {n_entries} entries, central directory {cd_size / 1e6:.1f} MB", flush=True)
    cd = b"".join(read_global(cd_off + s, min(8 << 20, cd_size - s)) for s in range(0, cd_size, 8 << 20))
    idx, q = {}, 0
    while q + 46 <= len(cd) and cd[q:q + 4] == b"PK\x01\x02":
        (_, _, _, _, method, _, _, _, csize, usize, nlen, xlen, clen, _, _, _, lho) = struct.unpack(
            "<4s6H3L5H2L", cd[q:q + 46])
        name = cd[q + 46:q + 46 + nlen].decode("utf-8", "replace")
        extra = cd[q + 46 + nlen:q + 46 + nlen + xlen]
        e = 0
        while e + 4 <= len(extra):  # zip64 extended information: only the fields that overflowed, in order
            hid, hlen = struct.unpack("<2H", extra[e:e + 4])
            if hid == 1:
                vals, v = [], e + 4
                for field in (usize, csize, lho):
                    if field == 0xFFFFFFFF:
                        vals.append(struct.unpack("<Q", extra[v:v + 8])[0])
                        v += 8
                    else:
                        vals.append(field)
                usize, csize, lho = vals
            e += 4 + hlen
        if not name.endswith("/"):
            idx[name] = [lho, csize, method, nlen]
        q += 46 + nlen + xlen + clen
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(idx))
    print(f"indexed {len(idx)} files -> {path}", flush=True)
    return idx


def fetch_member(entry):
    lho, csize, method, nlen = entry
    want = 30 + nlen + 64 + csize  # local header + name + (usually short) extra field + data, one request
    blob = read_global(lho, min(want, TOTAL - lho))
    assert blob[:4] == b"PK\x03\x04", "bad local header"
    ln, lx = struct.unpack("<2H", blob[26:30])
    start = 30 + ln + lx
    data = blob[start:start + csize]
    if len(data) < csize:
        data += read_global(lho + start + len(data), csize - len(data))
    if method == 8:
        return zlib.decompress(data, -15)
    if method != 0:
        raise ValueError(f"zip method {method}")
    return data


def load_csv():
    path = OUT / CSV_NAME
    if not path.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(REPO + CSV_NAME, headers={"Authorization": f"Bearer {_token()}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8", errors="replace"))))
    head, rows = rows[0], rows[1:]
    uid_col = next(i for i, h in enumerate(head) if "uid" in h.lower() or "id" in h.lower())
    txt_col = next((i for i, h in enumerate(head) if h.lower() in ("text", "english", "translation", "sentence")),
                   1 if uid_col == 0 else 0)
    print(f"csv columns {head}: uid={head[uid_col]!r} text={head[txt_col]!r}, {len(rows)} rows", flush=True)
    return [(r[uid_col], r[txt_col]) for r in rows if len(r) > max(uid_col, txt_col)]


def sign_matcher(labels):
    m = {l.lower(): l for l in labels if " " not in l and l not in SKIP}
    for lab, words in SPLIT_LABELS.items():
        if lab in labels:
            m.update({w: lab for w in words})
    sys.path.insert(0, str(ROOT / "ml"))
    from eval_live import ALIAS
    m.update({k: v for k, v in ALIAS.items() if v in labels and v not in SKIP})
    phrases = {p: l for p, l in PHRASES.items() if l in labels}

    def ordered(text):
        """Signs in text order (for gloss-ordered sentences), phrases first."""
        t = " " + " ".join(re.findall(r"[a-z]+", text.lower())) + " "
        for p, l in phrases.items():
            t = t.replace(f" {p} ", f" <{len(l)}:{l}> ") if f" {p} " in t else t
        out = []
        for w in re.findall(r"<\d+:[^>]+>|\S+", t):
            if w.startswith("<"):
                out.append(w.split(":", 1)[1][:-1])
                continue
            lab = m.get(w) or (m.get(w[:-1]) if len(w) > 3 and w.endswith("s") else None)
            if lab:
                out.append(lab)
        return out

    def match(text):
        t = " " + " ".join(re.findall(r"[a-z]+", text.lower())) + " "
        found = set()
        for p, l in phrases.items():  # a phrase sign ("good morning") replaces its words' own signs
            if f" {p} " in t:
                found.add(l)
                t = t.replace(f" {p} ", " | ")
        for w in t.split():
            lab = m.get(w)
            if lab is None and len(w) > 4 and w.endswith("ies"):
                lab = m.get(w[:-3] + "y")
            if lab is None and len(w) > 4 and w.endswith("es"):
                lab = m.get(w[:-2])
            if lab is None and len(w) > 3 and w.endswith("s"):
                lab = m.get(w[:-1])
            if lab:
                found.add(lab)
        return found
    match.ordered = ordered
    return match


def plan(per_sign):
    v = json.loads((ROOT / "ml" / "vocab.json").read_text())
    labels = v["include_signs"] + v["cislr_signs"] + v.get("common_signs", [])
    match = sign_matcher(labels)
    rows = load_csv()
    idx = zip_index()
    by_uid = {Path(n).stem: n for n in idx if n.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))}
    print(f"{len(by_uid)} videos in the zip; csv uids found in zip: "
          f"{sum(u in by_uid for u, _ in rows)}/{len(rows)}", flush=True)
    cand = defaultdict(list)  # sign -> [(n words, video id, uid)]
    texts = {}
    for uid, text in rows:
        if uid not in by_uid:
            continue
        signs = match(text)
        n_words = len(re.findall(r"[a-z]+", text.lower()))
        if not signs or n_words == 0 or n_words > 14:
            continue
        texts[uid] = text
        vid = uid.rsplit("-", 1)[0]
        for s in signs:
            cand[s].append((n_words, vid, uid))
    chosen, why = set(), defaultdict(list)
    for s, items in cand.items():
        items.sort()
        seen_vid, picked = Counter(), []
        for rnd in range(3):  # short texts first, at most 1 (then 2, 3) clips per source video: more signers
            for n, vid, uid in items:
                if len(picked) >= per_sign:
                    break
                if uid not in picked and seen_vid[vid] == rnd:
                    picked.append(uid)
                    seen_vid[vid] += 1
        for uid in picked:
            chosen.add(uid)
            why[uid].append(s)
    plan = {uid: {"text": texts[uid], "signs": sorted(match(texts[uid])), "file": by_uid[uid]} for uid in sorted(chosen)}
    gb = sum(idx[p["file"]][1] for p in plan.values()) / 1e9
    (OUT / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=0))
    per = Counter(s for p in plan.values() for s in p["signs"])
    print(f"plan: {len(plan)} clips, {gb:.2f} GB, {len(per)} signs; candidates per sign (min/median/max) "
          f"{min(map(len, cand.values()))}/{sorted(map(len, cand.values()))[len(cand) // 2]}/"
          f"{max(map(len, cand.values()))}", flush=True)
    print("signs with < 10 clips:", sorted(s for s in labels if per.get(s, 0) < 10 and s not in SKIP), flush=True)
    return plan, idx


def download(plan, idx, workers):
    VIDEOS.mkdir(parents=True, exist_ok=True)
    todo = [(uid, p["file"]) for uid, p in plan.items() if not (VIDEOS / f"{uid}.mp4").exists()]
    total = sum(idx[f][1] for _, f in todo)
    print(f"downloading {len(todo)} clips ({total / 1e9:.2f} GB) with {workers} parallel requests", flush=True)

    def one(uid, f):
        data = fetch_member(idx[f])
        tmp = VIDEOS / f"{uid}.part"
        tmp.write_bytes(data)
        tmp.replace(VIDEOS / f"{uid}.mp4")
        return len(data)

    t0, done, nbytes, failed = time.time(), 0, 0, 0
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one, uid, f): uid for uid, f in todo}
        for fu in as_completed(futs):
            done += 1
            try:
                nbytes += fu.result()
            except Exception as e:
                failed += 1
                print(f"failed {futs[fu]}: {e}", flush=True)
            if done % 100 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"[{done}/{len(todo)}] {nbytes / 1e6:.0f} MB, {nbytes / 1e6 / el:.2f} MB/s, "
                      f"ETA {(len(todo) - done) * el / done / 60:.1f} min, failed {failed}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="only build the plan, no video download")
    ap.add_argument("--per-sign", type=int, default=40, help="max clips per sign (short texts, many videos first)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    p, idx = plan(args.per_sign)
    if not args.plan:
        download(p, idx, args.workers)


if __name__ == "__main__":
    main()
