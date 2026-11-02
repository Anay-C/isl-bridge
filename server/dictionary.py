"""The ISLRTC ISL Dictionary as a searchable, on-demand clip source (13,665 videos; the 2,311 single-sign
entries are used, the rest also contain a signed definition).

Source: Hugging Face mirror of the data.gov.in "Indian Sign Language Dictionary" (ISLRTC); file list cached in
data/islrtc_hf_files.json. ISLRTC terms: research / ISL technology use with acknowledgement, no profiteering.
Clips are downloaded the first time they are needed, centre-cropped (drops the burned-in word label and logo),
scaled to 480p and cached in data/clips_dict/.
"""
import asyncio
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPO = "silentone0725/Indian_Sign_Language_Data.gov_Rencoded"
CACHE = DATA / "clips_dict"
RAW = DATA / "islrtc" / "raw"

STOP = set("""a an the is am are was were be been being i you he she it we they me him her us them my your his
its our their this that these those to of in on at for with from by and or but if then so do does did have has
had will would shall should can could may might must not no yes please okay ok what where when who why how
which there here just very also too much many some any all""".split())


def _norm(s):
    s = re.sub(r"(\w)-(\w)", r"\1\2", s.lower().replace("_", " "))  # X-Ray -> xray
    return re.sub(r"[^a-z0-9 ]+", " ", s).split()


# Most dictionary entries are a sign PLUS a signed definition (10-45 s). Only short files are single signs:
# on the 190 clips we have keypoints for, <= 1.0 MB kept 130 single signs and 0 of the 30 definition videos.
MAX_BYTES = 1_000_000


@lru_cache(maxsize=1)
def entries():
    """[(display name, repo path, normalised tokens, is_variant)] for single-sign entries only."""
    files = json.loads((DATA / "islrtc_hf_files.json").read_text())
    sizes_f = DATA / "islrtc_hf_sizes.json"
    sizes = json.loads(sizes_f.read_text()) if sizes_f.exists() else {}
    out = []
    for f in files:
        if not f.endswith(".mp4") or sizes.get(f, 0) > MAX_BYTES:
            continue
        name = Path(f).stem.replace("_", " ").replace(" - English", "").strip()
        variant = bool(re.search(r"\((sign|explanation|example|academic)", name, re.I))
        out.append((name, f, _norm(re.sub(r"\(.*?\)", " ", name)), variant))
    return out


def search(query, k=6, min_score=-999):
    """Best dictionary entries for a word/phrase: exact > all query words > overlap; main entry before variants."""
    q = [w for w in _norm(query) if w]
    if not q:
        return []
    qs = set(q)
    scored = []
    for name, path, toks, variant in entries():
        if not toks:
            continue
        ts = set(toks)
        if toks == q:
            s = 100
        elif qs <= ts:
            s = 60 - len(ts - qs) * 12  # extra words in the entry = more specific / different meaning
        elif ts <= qs:
            s = 40 - len(qs - ts) * 5
        else:
            inter = len(qs & ts)
            if not inter:
                continue
            s = 10 * inter - len(ts - qs)
        scored.append((s - (3 if variant else 0), name, path))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    return [{"entry": n, "id": p} for sc, n, p in scored[:k] if sc >= min_score]


def candidates_for(text, known):
    """Pre-search the dictionary for content words / word pairs in `text` that have no app clip yet."""
    words = _norm(text)
    known_l = {k.lower() for k in known}
    out = {}
    for i, w in enumerate(words):
        if w in STOP or w in known_l or len(w) < 3:
            continue
        pair = [f"{w} {words[i + 1]}"] if i + 1 < len(words) and words[i + 1] not in STOP else []
        for phrase in pair + [w]:
            hits = search(phrase, k=3, min_score=37)  # strong matches only; Spark can search for the rest
            if hits and phrase not in out:
                out[phrase] = hits
    return out


def _safe(path):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path).stem) + ".mp4"


def _fetch_blocking(path):
    import imageio_ffmpeg
    from huggingface_hub import hf_hub_download
    dst = CACHE / _safe(path)
    if dst.exists():
        return dst
    src = hf_hub_download(REPO, path, repo_type="dataset", local_dir=RAW)
    CACHE.mkdir(parents=True, exist_ok=True)
    import os
    tmp = dst.with_suffix(f".{os.getpid()}.tmp.mp4")  # two servers (LAN + public) may fetch the same clip
    # signer stands in the middle; the word label (left) and ISLRTC logo (bottom right) fall outside this crop
    vf = "crop=iw*0.5:ih:iw*0.25:0,scale=-2:480,fps=25,format=yuv420p"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-vf", vf, "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-movflags", "+faststart", str(tmp)],
                   check=True)
    tmp.replace(dst)
    return dst


async def fetch_clip(path):
    """Repo path -> URL of a cached, browser-ready clip (downloads on first use)."""
    if not any(path == p for _, p, _, _ in entries()):
        raise ValueError(f"not a dictionary entry: {path}")
    lock = _locks.setdefault(path, asyncio.Lock())  # two requests for one clip must not share a temp file
    async with lock:
        dst = await asyncio.to_thread(_fetch_blocking, path)
    return f"/clips_dict/{dst.name}"


_locks: dict[str, asyncio.Lock] = {}


# Words a doctor, nurse or pharmacist often says that have no app clip: their dictionary clips are fetched ahead
# of time (python server/dictionary.py --prefetch), so a reply never waits for a first download (~3 s each).
COMMON_REPLY_WORDS = """take give come go back wait sit stand lie sleep rest eat drink breathe swallow open close show
tell write read call pay bring buy check test measure wash clean apply use stop start finish need want feel walk keep
avoid return visit admit discharge operate inject examine scan weigh cough sneeze bleed understand know think remember
forget worry relax wear remove change repeat follow allow high low normal good bad better worse hot cold warm big small
more less again later now soon early late slowly fast careful important serious safe free ready empty full every daily
twice once after before since until always never sometimes only blood sugar urine pressure report result appointment
bed room ward floor lift stairs money fee card form name address age weight height sample syrup drip bandage cream
drops eye ear nose throat chest leg arm hand foot skin tooth teeth heart lung kidney liver bone infection diabetes
operation surgery scan week month day night evening hour time milk tea rice fruit family husband wife child baby
friend pharmacy prescription insurance number two three four five ten half breakfast lunch dinner meal empty stomach
increase decrease sign yes no okay sorry welcome""".split()


def prefetch(words=COMMON_REPLY_WORDS, clip_labels=()):
    """Download + convert the top dictionary entry for each word (skips words the app has a clip for)."""
    have = {c.lower() for c in clip_labels}
    todo = []
    for w in dict.fromkeys(words):
        if w in have or w in STOP:
            continue
        hits = search(w, k=1, min_score=37)
        if hits and hits[0]["id"] not in todo:
            todo.append(hits[0]["id"])
    done = 0
    for i, path in enumerate(todo):
        try:
            _fetch_blocking(path)
            done += 1
        except Exception as e:
            print(f"  {path}: {e!r}"[:200], flush=True)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(todo)}", flush=True)
    print(f"prefetched {done}/{len(todo)} dictionary clips into {CACHE}", flush=True)


if __name__ == "__main__":
    import sys
    if "--prefetch" in sys.argv:
        clips = [p.stem for p in (DATA / "clips").glob("*.mp4")]
        prefetch(clip_labels=clips)
