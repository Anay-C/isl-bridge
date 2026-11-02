"""FastAPI server: live sign recognition over WebSocket + speech -> ISL clip playlist.

    .venv-ml/Scripts/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

WebSocket /ws/sign  (client -> server JSON)
    {"type":"hello", "user":"anay", "scenario":"hospital"}
    {"type":"frame", "w":640, "h":480, "pose":[132 floats]|null, "lh":[63]|null, "rh":[63]|null}
    {"type":"enroll", "label":"Pain", "reps":5}   next N signs become personal samples
    {"type":"undo"} | {"type":"speak_now"} | {"type":"reset"}
    {"type":"feedback", "sid":3, "signs":[{"i":0,"meant":"Pain"}], "note":"..."}   "that was wrong"
    {"type":"consent", "log":false}   stop saving this person's sign keypoints
server -> client: state / sign / sentence / enroll_progress / feedback_ok / error

Public deployment (ISL_PUBLIC=1, deploy/hf): debug routes off, per-client rate limits, the Meta spend cap
(server/muse.py) and tester data copied to a private HF dataset (server/sync.py).
"""
import asyncio
import json
import os
import re
import secrets
import shutil
import sys
import time
import traceback
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "server"))

import coach  # noqa: E402
import spotting  # noqa: E402
import dictionary  # noqa: E402
from features import clip_frames  # noqa: E402
from recognizer import Recognizer  # noqa: E402
import segmenter  # noqa: E402
from segmenter import Segmenter  # noqa: E402
import muse  # noqa: E402
import pictures  # noqa: E402
import sync  # noqa: E402

DATA = ROOT / "data"
PUBLIC = os.environ.get("ISL_PUBLIC") == "1"  # open to the internet (Hugging Face Space)
VERSION = os.environ.get("ISL_VERSION", "dev")
FEEDBACK = DATA / "feedback"  # <user>.jsonl mistake reports (+ <user>/*.npz keypoints of corrected signs)
# profiles nobody on the public link may open (profiles have no password; the LAN server shares data/users)
PROTECTED = {u.strip() for u in os.environ.get("ISL_PROTECTED_USERS", "anay").split(",") if u.strip()}
# ISL_MODEL: checkpoint(s) to serve, comma-separated = ensemble (ml/recognizer.py); ISL_DEVICE: cpu | cuda
MODEL = os.environ.get("ISL_MODEL") or str(DATA / "model.pt")
DEVICE = os.environ.get("ISL_DEVICE", "cpu")
USERS = DATA / "users"
LIVE_LOG = DATA / "live_log"  # <user>/<time>.npz: every live segment (keypoints only) + what was predicted
RECORDINGS = DATA / "recordings"  # <user>/<label>/*.npz raw segments (training data for missing signs + eval)
CLIPS = DATA / "clips"  # <label>.mp4 plain sign clips (ISLRTC / INCLUDE / CISLR)
CUTOUTS = DATA / "clips_sam"  # <label>.mp4 SAM 3.1 stacked cut-outs (colour over mask) for the AR interpreter
VOCAB = json.loads((ROOT / "ml" / "vocab.json").read_text())
SCENARIOS = {
    "hospital": "A Deaf patient at a hospital / clinic reception or with a doctor in India.",
    "pharmacy": "A Deaf customer buying medicine at a pharmacy in India.",
    "general": "An everyday conversation in India.",
}
SENTENCE_PAUSE_S = 0.5  # hands down this long = the sentence is finished -> speak it (Anay: 0.5 s)
SPEC_MAX = 6  # speculative Spark calls per stretch of signing (~$0.001 each)
INTERIM_EVERY_S = 0.5  # live guesses while signing: at most one decode in flight, at most this often
INTERIM_MAX_FRAMES = 150  # decode only the latest ~5 s for live guesses (the final decode uses everything)
MAX_PERSONAL = 20  # newest personal samples kept per sign (teaching + corrections)
# Meta calls per signer per minute. A normal conversation makes ~5-20 (one per sentence + a few speculative ones);
# a phone left facing a TV must not drain the shared credit. Over SPEC_PER_MIN no more speculative calls; over
# SPARK_PER_MIN sentences are word by word until the minute has passed.
SPARK_PER_MIN = int(os.environ.get("SPARK_PER_MIN", 40))
SPEC_PER_MIN = int(os.environ.get("SPEC_PER_MIN", 25))
SESSIONS: set = set()
STARTED = time.time()


class Rate:
    """At most n events per `per` seconds for each key (a session, or a client IP for the HTTP routes)."""

    def __init__(self, n, per=60.0):
        self.n, self.per, self.t = n, per, {}

    def count(self, key):
        q = self.t.get(key)
        now = time.monotonic()
        while q and now - q[0] > self.per:
            q.popleft()
        return len(q) if q else 0

    def ok(self, key="", limit=None):
        if self.count(key) >= (self.n if limit is None else limit):
            return False
        if len(self.t) > 10000:  # forget idle clients
            self.t = {k: q for k, q in self.t.items() if q and time.monotonic() - q[-1] < self.per}
        self.t.setdefault(key, deque()).append(time.monotonic())
        return True


# typed / spoken replies per client IP (generous: testers on one Wi-Fi or a mobile carrier share an IP)
RATE_HTTP = Rate(int(os.environ.get("HTTP_PER_MIN", 40)))
RATE_LENS = Rate(int(os.environ.get("LENS_PER_MIN", 8)))
RATE_SPEECH = Rate(int(os.environ.get("SPEECH_PER_MIN", 40)))


def client_ip(r):
    """Request or WebSocket -> the visitor's IP, for the per-device limits. Behind the Cloudflare tunnel:
    Cf-Connecting-Ip (Cloudflare sets it; a visitor cannot forge it through the tunnel). Otherwise the LAST
    X-Forwarded-For hop (added by the nearest proxy; the first one is whatever the client claimed)."""
    cf = r.headers.get("cf-connecting-ip", "").strip()
    if cf:
        return cf
    hops = [x.strip() for x in r.headers.get("x-forwarded-for", "").split(",") if x.strip()]
    return hops[-1] if hops else (r.client.host if r.client else "?")


@asynccontextmanager
async def lifespan(_app):
    sync.restore()  # public deployment: testers' profiles + the spend ledger survive a restart
    sync.start()
    recognizer()  # load the model before the first visitor, not during their first sign
    yield
    sync.flush()


app = FastAPI(title="ISL conversation", lifespan=lifespan)
_recognizer: Recognizer | None = None


class _SelectiveGZip:
    """gzip the app bundle, the MediaPipe wasm (10 MB -> ~3 MB on a phone's data plan) and JSON; never videos
    (Safari plays them with byte-range requests, which compression would break)."""

    def __init__(self, inner):
        self.inner = inner
        self.gz = GZipMiddleware(inner, minimum_size=1024)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith(("/assets/", "/wasm/", "/api/")):
            return await self.gz(scope, receive, send)
        return await self.inner(scope, receive, send)


app.add_middleware(_SelectiveGZip)


@app.middleware("http")
async def _cache_headers(request: Request, call_next):
    """Repeat visits load instantly: hashed bundles are immutable, models / wasm / clips change rarely."""
    resp = await call_next(request)
    p = request.url.path
    if resp.status_code == 200 and "cache-control" not in resp.headers:
        if p.startswith("/assets/"):
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif p.startswith(("/wasm/", "/models/")):
            resp.headers["Cache-Control"] = "public, max-age=604800"
        elif p.startswith(("/clips", "/pictures/")):
            resp.headers["Cache-Control"] = "public, max-age=86400"
        elif not p.startswith(("/api/", "/ws/")):
            resp.headers["Cache-Control"] = "no-cache"  # index.html: always pick up a new deployment
    return resp


def recognizer():
    global _recognizer
    if _recognizer is None and all(Path(p).exists() for p in MODEL.split(",")):
        if os.environ.get("TORCH_THREADS"):  # e.g. 2 on a 2-vCPU Space
            import torch
            torch.set_num_threads(int(os.environ["TORCH_THREADS"]))
        _recognizer = Recognizer(MODEL, device=DEVICE)
        print(f"[server] recogniser: {MODEL} on {DEVICE}", flush=True)
        _recognizer.active = scenario_vocab("general")
    return _recognizer


# Hospital / pharmacy: the hospital vocabulary chosen for the demo (INCLUDE + CISLR) plus people, days and "it".
# Off-topic everyday signs (Dress, Market, Train, School, colours...) are only guessed in "general": in Anay's
# live logs (31 Oct) "me" kept coming out as "Dress" and Muse Spark then wrote about dressing.
HOSPITAL_EXTRA = {"it", "you (plural)", "Man", "Woman", "Boy", "Girl", "Son", "Daughter", "Parent", "Grandfather",
                  "Grandmother", "Year", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}


def scenario_vocab(scenario):
    core = set(VOCAB["include_signs"]) | set(VOCAB["cislr_signs"])
    return core | HOSPITAL_EXTRA if scenario in ("hospital", "pharmacy") else core | set(VOCAB.get("common_signs", []))


# words people type when teaching -> the app's sign label ("help" must improve "Help", not become a new sign)
LABEL_ALIAS = {"me": "I", "my": "I", "mine": "I", "myself": "I", "hi": "Hello", "thanks": "Thank you",
               "thank": "Thank you", "medicines": "Medicine", "tablets": "Tablet", "shop": "Store or Shop"}


def canonical_label(label, known):
    low = {k.lower(): k for k in known}
    l = label.strip()
    return LABEL_ALIAS.get(l.lower()) or low.get(l.lower()) or l


def _arr(v, shape, w, h):
    """Landmark list from the phone -> pixel units. Anything malformed (wrong length, NaN, strings) counts as
    "not detected" instead of raising: one bad frame must never end a signer's session."""
    if v is None:
        return None
    try:
        a = np.asarray(v, np.float32)
        if a.size != shape[0] * shape[1] or not np.isfinite(a).all():
            return None
        a = a.reshape(shape).copy()
    except (TypeError, ValueError):
        return None
    a[:, 0] *= w
    a[:, 1] *= h
    a[:, 2] *= w  # same as features._to_pixels
    return a


def _dim(v, default):
    try:
        v = float(v)
        return v if np.isfinite(v) and 0 < v < 100000 else default
    except (TypeError, ValueError):
        return default


class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.user = "guest"
        self.scenario = "hospital"
        self.seg = Segmenter(hold=None, max_len=360)  # whole utterance until hands drop; spotting splits it
        self.pending: list[list[dict]] = []  # top-3 per sign, waiting for a sentence
        self.pending_emb: list[np.ndarray] = []  # embedding of each pending sign (for tap-to-correct)
        self.history: list[str] = []
        self.enroll: dict | None = None
        self.practice: str | None = None  # Learn page: next sign is scored against this word
        self.timer: asyncio.Task | None = None
        self.speaking = False  # Spark is writing a sentence right now
        self.learn = False  # Learn page: only practice attempts are scored
        self.signing = False
        self.frame_times: list[float] = []
        self.rec = None
        self.version = 0  # bumps whenever `pending` changes (invalidates a speculative sentence)
        # speculative Spark calls, keyed by the signs they were written for: started from the live guesses while
        # signing and at hands-down, so the sentence is usually ready when the pause ends
        self.specs: dict[tuple, asyncio.Task] = {}
        self.spec_count = 0  # speculative calls started during the current stretch of signing
        self.interim_task: asyncio.Task | None = None
        self.interim_at = 0.0
        self.utterance = 0  # id of the current stretch of signing (drops stale live guesses)
        self.quiet_until = 0.0  # just finished teaching: the end of the last rep is not a conversation sign
        self.down_t = 0.0  # when the hands last went down (the sentence pause is counted from here)
        self.errors = 0  # messages that raised (logged; the session carries on)
        self.log_ok = True  # "help improve ISL Bridge": keep this person's sign keypoints (they can switch it off)
        self.spark = Rate(SPARK_PER_MIN)  # this signer's Meta calls in the last minute
        self.summaries = Rate(4)
        self.feedbacks = Rate(20)
        self.client_logs = 0
        # "that was wrong" reports: the keypoints of the last stretches of signing, and for each pending sign
        # which stretch (and frames) it came from; spoken sentences keep their signs for a while
        self.segs: deque = deque(maxlen=12)  # (seg_id, {"pose", "lh", "rh", "lh_p", "rh_p", "fps"})
        self.seg_id = 0
        self.pending_src: list[tuple] = []  # (seg_id, start, end) per pending sign
        self.sentences: dict[int, dict] = {}  # sid -> {"cands", "embs", "src", "english"}
        self.sid = 0

    async def send(self, **msg):
        await self.ws.send_text(json.dumps(msg, ensure_ascii=False))

    def spark_ok(self, speculative=False):
        """A Meta call for this signer is within the per-minute limit (and the credit is not used up)."""
        if muse.budget_left() <= 0:
            return False
        return self.spark.ok("s", SPEC_PER_MIN if speculative else None)

    def _log_live(self, pose, lh, rh, lh_p, rh_p, top):
        """Keep every live segment so live accuracy problems can be diagnosed (data/ is git-ignored).
        Only while the person leaves "help improve ISL Bridge" on (the welcome screen says so)."""
        if not self.log_ok:
            return
        try:
            out = LIVE_LOG / self.user / f"{time.strftime('%Y%m%d-%H%M%S')}-{len(self.pending)}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, pose=pose, lh=lh, rh=rh, lh_p=lh_p, rh_p=rh_p, fps=self.fps(), width=1, height=1,
                                top=json.dumps(top))
        except Exception:
            pass

    def fps(self):
        """Camera frame rate over the last 30 frames. Uses the phone's capture timestamps ("t", ms) when sent: on
        patchy Wi-Fi frames arrive in bursts, and arrival-time fps would spike and mis-scale the spotting windows
        (spotting.spot scales its window lengths by fps / 25). Clamped to what phones really deliver."""
        ts = self.frame_times[-30:]
        fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 30.0
        return float(min(60.0, max(5.0, fps)))

    def user_file(self):
        return USERS / f"{self.user}.json"

    async def on_frame(self, m):
        t = m.get("t")
        now = time.perf_counter()
        if isinstance(t, (int, float)) and np.isfinite(t):
            t = float(t) / 1000.0
            # client clock (seconds); a restarted client clock (smaller t) starts a fresh window
            if self.frame_times and (t <= self.frame_times[-1] or t - self.frame_times[-1] > 5.0):
                self.frame_times = []
            self.frame_times = (self.frame_times + [t])[-60:]
        else:
            self.frame_times = (self.frame_times + [now])[-60:]
        # conversation: one segment = a whole utterance (spotting splits it). Teach / practice: one sign per
        # attempt, so a hold-still also ends it (people often keep their hands up after the last rep)
        self.seg.hold = segmenter.HOLD if (self.enroll or self.practice) else None
        w, h = _dim(m.get("w"), 640.0), _dim(m.get("h"), 480.0)
        frame = {"pose": _arr(m.get("pose"), (33, 4), w, h), "lh": _arr(m.get("lh"), (21, 3), w, h),
                 "rh": _arr(m.get("rh"), (21, 3), w, h), "w": w, "h": h}
        seg = self.seg.push(frame, fps=self.fps())
        if self.seg.signing != self.signing:
            self.signing = self.seg.signing
            await self.send(type="state", signing=self.signing)
            if self.signing:
                self.cancel_timer()
                self.spec_count = 0
            else:
                self.utterance += 1
                if seg is None and self.pending and not self.speaking:
                    # the hands came up but it was no sign (too short / no hand seen): the signs made before it must
                    # still be spoken (the pause timer was cancelled when the hands came up)
                    self.down_t = time.perf_counter()
                    self.schedule_sentence()
        if seg:
            self.utterance += 1
            # the segmenter ends a stretch END_GAP frames after the hands drop
            self.down_t = time.perf_counter() - segmenter.END_GAP / self.fps()
            await self.on_segment(seg)
        elif self.signing:
            self.maybe_interim()

    def maybe_interim(self):
        """Live guesses while the person is still signing, so the screen reacts before the hands drop."""
        if (self.rec is None or self.enroll or self.practice or self.learn or time.perf_counter() < self.quiet_until
                or (self.interim_task and not self.interim_task.done())
                or time.perf_counter() - self.interim_at < INTERIM_EVERY_S):
            return
        buf = self.seg.current()[-INTERIM_MAX_FRAMES:]
        if len(buf) < 12:
            return
        self.interim_at = time.perf_counter()
        uid, fps, rec = self.utterance, self.fps(), self.rec

        async def run():
            try:
                spots = await asyncio.to_thread(spotting.spot, rec, list(buf), fps)
                if uid == self.utterance and self.signing:
                    await self.send(type="interim", signs=[top[0]["label"] for _, _, top in spots])
                    cands = self.pending + [top for _, _, top in spots]
                    if spots and self.spec_count < SPEC_MAX and self.spec_key(cands) not in self.specs:
                        self.spec_count += 1
                        self.start_spec(cands)
            except Exception:
                pass
        self.interim_task = asyncio.create_task(run())

    async def on_segment(self, seg):
        rec = self.rec
        t0 = time.perf_counter()
        pose = np.stack([f["pose"] if f["pose"] is not None else np.zeros((33, 4), np.float32) for f in seg])
        lh = np.stack([f["lh"] if f["lh"] is not None else np.zeros((21, 3), np.float32) for f in seg])
        rh = np.stack([f["rh"] if f["rh"] is not None else np.zeros((21, 3), np.float32) for f in seg])
        lh_p = np.array([f["lh"] is not None for f in seg])
        rh_p = np.array([f["rh"] is not None for f in seg])
        if rec is None and not (self.enroll and self.enroll.get("save")):
            await self.send(type="error", message="model not trained yet")
            return
        if self.practice and rec:
            # model inference runs in a worker thread so one signer never freezes the event loop for everyone
            res = await asyncio.to_thread(coach.score_attempt, rec, self.practice, pose, lh, rh, lh_p, rh_p)
            await self.send(type="practice_result", **res)
            return
        if self.learn and not self.enroll:  # Learn page, not practising: no sentences, no Spark spend
            return
        if not self.enroll and time.perf_counter() < self.quiet_until:
            return
        # frames already in pixel units -> pass width = height = 1
        emb = (await asyncio.to_thread(lambda: rec.embed_frames([clip_frames(pose, lh, rh, lh_p, rh_p, 1, 1)])[0])
               if rec else None)
        ms = round((time.perf_counter() - t0) * 1000)

        if self.enroll and self.enroll.get("save") and (self.log_ok or not PUBLIC):
            out = RECORDINGS / self.user / self.enroll["label"] / f"{time.strftime('%Y%m%d-%H%M%S')}-{self.enroll['n']}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            # pixel units already -> width = height = 1 (same layout as ml/extract.py output)
            np.savez_compressed(out, pose=pose, lh=lh, rh=rh, lh_p=lh_p, rh_p=rh_p, fps=30.0, width=1, height=1)
        if self.enroll:
            e = self.enroll
            if e.get("personalise", True) and rec:
                rec.add_sample(e["label"], emb)
            e["n"] += 1
            await self.send(type="enroll_progress", label=e["label"], n=e["n"], reps=e["reps"])
            if e["n"] >= e["reps"]:
                if e.get("personalise", True) and rec:
                    # personal samples live only in data/users/<user>.json; the shared model never sees them
                    await self.send(type="enroll_done", **rec.enroll_report(e["label"]))
                    rec.save(self.user_file())
                self.enroll = None
                self.quiet_until = time.perf_counter() + 1.5
            return

        # one stretch of signing can hold several signs: spot them (see ml/spotting.py)
        fps = self.fps()

        def decode():
            spots = spotting.spot(rec, seg, fps=fps) or [(0, len(seg), rec.predict(emb, k=3))]
            embs = [emb] if len(spots) == 1 else [spotting.embed_segment(rec, seg[s:e]) for s, e, _ in spots]
            return spots, embs

        spots, embs = await asyncio.to_thread(decode)
        ms = round((time.perf_counter() - t0) * 1000)
        self._log_live(pose, lh, rh, lh_p, rh_p, [{"start": s, "end": e, "top": top} for s, e, top in spots])
        self.seg_id += 1
        self.segs.append((self.seg_id, {"pose": pose, "lh": lh, "rh": rh, "lh_p": lh_p, "rh_p": rh_p, "fps": fps}))
        for (s, e, top), em in zip(spots, embs):
            self.pending.append(top)
            self.pending_emb.append(em)
            self.pending_src.append((self.seg_id, int(s), int(e)))
            await self.send(type="sign", top=top, index=len(self.pending) - 1, latency_ms=ms)
        self.changed()
        self.schedule_sentence()

    def changed(self):
        self.version += 1

    def spec_key(self, cands):
        return len(self.history), self.scenario, tuple(c[0]["label"] for c in cands if c)

    def start_spec(self, cands):
        k = self.spec_key(cands)
        if not cands or k in self.specs or not self.spark_ok(speculative=True):
            return
        self.specs[k] = asyncio.create_task(muse.build_sentence(
            [list(c) for c in cands], list(self.history), SCENARIOS.get(self.scenario, self.scenario)))
        while len(self.specs) > 8:  # oldest guesses are no longer useful
            self.specs.pop(next(iter(self.specs))).cancel()

    def drop_specs(self):
        for t in self.specs.values():
            t.cancel()
        self.specs = {}

    def clear_pending(self):
        self.pending, self.pending_emb, self.pending_src = [], [], []
        self.changed()

    def schedule_sentence(self):
        self.cancel_timer()
        # speculative: start Spark now (unless a live guess already started it for exactly these signs)
        self.start_spec(self.pending)
        self.timer = asyncio.create_task(self._sentence_after_pause())

    def cancel_timer(self):
        """Cancel only the waiting pause. A sentence already being written by Spark is never cancelled:
        its signs have left `pending`, so cancelling it would silently lose what the person signed."""
        if self.timer and not self.speaking:
            self.timer.cancel()

    async def _sentence_after_pause(self):
        # counted from the moment the hands went down, so recognition time is not added on top of the pause
        await asyncio.sleep(max(0.0, self.down_t + SENTENCE_PAUSE_S - time.perf_counter()))
        await self.speak()

    async def speak(self):
        if not self.pending:
            return
        spec = self.specs.pop(self.spec_key(self.pending), None)
        cands, embs, srcs = self.pending, self.pending_emb, self.pending_src
        self.pending, self.pending_emb, self.pending_src = [], [], []
        self.version += 1
        self.speaking = True
        try:
            if spec is None or spec.cancelled():
                spec = None
            if not (spec and spec.done()):
                await self.send(type="composing")  # pause is over, Spark is still writing: tell the screen
            out = None
            if spec:
                try:
                    out = await spec
                except asyncio.CancelledError:
                    out = None
            if out is None or out.get("_error"):
                if self.spark_ok():
                    out = await muse.build_sentence(cands, self.history, SCENARIOS.get(self.scenario, self.scenario))
                else:  # this signer's per-minute limit, or the credit is used up: still say what was signed
                    out = muse.plain_sentence(cands, "Meta AI limit reached: word by word")
        finally:
            self.speaking = False
        self.drop_specs()  # written for the old conversation history
        self.history.append(f"Deaf person: {out['english']}")
        self.sid += 1
        self.sentences[self.sid] = {"cands": cands, "embs": embs, "src": srcs, "english": out.get("english", "")}
        for k in sorted(self.sentences)[:-20]:  # "that was wrong" works on the last 20 sentences
            del self.sentences[k]
        # n = how many of the client's pending signs this sentence used (more may have arrived meanwhile);
        # cands = exactly what Muse Spark chose from, so the chat can show where context changed a sign
        await self.send(type="sentence", n=len(cands), sid=self.sid,
                        cands=[[{"label": c["label"], "p": round(float(c["p"]), 3)} for c in top] for top in cands],
                        **out)

    def delete_my_data(self):
        """"Delete my data": this profile's taught signs, keypoint logs, reports and recordings. Returns the
        number of files removed. Logging stops for the rest of the session."""
        n = 0
        for f in (self.user_file(), FEEDBACK / f"{self.user}.jsonl"):
            if f.exists():
                f.unlink()
                n += 1
        for d in (LIVE_LOG / self.user, FEEDBACK / self.user, RECORDINGS / self.user):
            if d.is_dir():
                n += sum(1 for x in d.rglob("*") if x.is_file())
                shutil.rmtree(d, ignore_errors=True)
        if self.rec is not None:
            self.rec.samples, self.rec._trust_cache = {}, {}
        self.log_ok = False
        self.segs.clear()
        self.sentences = {}
        return n

    async def on_feedback(self, m):
        """"That was wrong": what the person really signed, per sign of a spoken sentence, and/or a note.
        Always saved to data/feedback/<user>.jsonl (they chose to send it); the keypoints of the corrected signs
        only while "help improve" is on. A corrected sign is also learned privately, like tap-to-correct."""
        if not self.feedbacks.ok():
            await self.send(type="error", message="Too many reports at once - wait a minute.")
            return
        sid = m.get("sid")
        sent = self.sentences.get(sid) if isinstance(sid, int) else None
        fixes, learned, files = [], [], []
        known = self.rec.sign_labels() if self.rec is not None else []
        for f in (m.get("signs") or [])[:12] if sent else []:
            try:
                i, meant = int(f.get("i")), str(f.get("meant", "")).strip()[:60]
            except (TypeError, ValueError, AttributeError):
                continue
            if not meant or not 0 <= i < len(sent["cands"]):
                continue
            top = [c["label"] for c in sent["cands"][i]]
            label = canonical_label(meant, known) if known else meant
            fixes.append({"i": i, "said": top[0] if top else "", "top": top, "meant": label})
            if label != (top[0] if top else None) and label in known and sent["embs"][i] is not None:
                self.rec.add_sample(label, sent["embs"][i])
                self.rec.samples[label] = self.rec.samples[label][-MAX_PERSONAL:]
                learned.append(label)
            src = sent["src"][i] if i < len(sent["src"]) else None
            seg = next((d for k, d in self.segs if src and k == src[0]), None)
            if self.log_ok and seg is not None:
                try:
                    out = FEEDBACK / self.user / f"{time.strftime('%Y%m%d-%H%M%S')}-{sid}-{i}.npz"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out, **{k: v for k, v in seg.items() if k != "fps"}, fps=seg["fps"],
                                        width=1, height=1, span=np.array(src[1:]), meant=label,
                                        top=json.dumps(sent["cands"][i]))
                    files.append(out.name)
                except Exception as e:
                    print(f"[feedback {self.user}] keypoints not saved: {e!r}", flush=True)
        note = str(m.get("note") or "").strip()[:1000]
        if not fixes and not note:
            return
        if learned:
            self.rec.save(self.user_file())
            await self.send(type="personal", personal={k: len(v) for k, v in self.rec.samples.items()},
                            vocab=self.rec.sign_labels())
        line = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "user": self.user, "scenario": self.scenario,
                "version": VERSION, "sid": sid if sent else None, "sentence": sent["english"] if sent else None,
                "signs": [c[0]["label"] if c else "" for c in sent["cands"]] if sent else None, "fixes": fixes,
                "note": note, "keypoints": files, "device": str(m.get("device") or "")[:200],
                "history": self.history[-6:]}
        FEEDBACK.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK / f"{self.user}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        await self.send(type="feedback_ok", learned=learned, saved_keypoints=len(files))


@app.websocket("/ws/sign")
async def ws_sign(ws: WebSocket):
    await ws.accept()
    s = Session(ws)
    SESSIONS.add(s)
    rec = recognizer()
    if rec is not None:  # per-connection copy so personal samples don't leak between users
        s.rec = Recognizer.__new__(Recognizer)
        s.rec.__dict__.update(rec.__dict__)
        s.rec.samples = {}
        s.rec._trust_cache = {}
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            raw = msg.get("text") or (msg.get("bytes") or b"").decode("utf-8", "replace")
            # one message at a time, each in its own try: a malformed message or a bug in one sign's processing
            # is logged and reported, but never drops the connection (that would lose the whole conversation)
            try:
                m = json.loads(raw)
                if isinstance(m, dict):
                    await _dispatch(s, m)
            except WebSocketDisconnect:
                raise
            except json.JSONDecodeError:  # not from our client: ignore quietly
                continue
            except Exception as e:
                s.errors += 1
                if s.errors <= 20 or s.errors % 100 == 0:  # never flood the log
                    print(f"[ws_sign {s.user}] {type(e).__name__}: {e} (message type {str(raw[:40])!r})\n"
                          f"{traceback.format_exc(limit=4)}", flush=True)
                try:
                    await s.send(type="error", message="Something went wrong with that sign - please sign it again.")
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        SESSIONS.discard(s)
        if s.timer:
            s.timer.cancel()
        s.drop_specs()


async def _dispatch(s, m):
    """One client message of /ws/sign."""
    t = m.get("type")
    if t == "frame":
        await s.on_frame(m)
    elif t == "hello":
        s.user = ("".join(ch for ch in str(m.get("user", "guest")) if ch.isalnum() or ch in "-_") or "guest")[:40]
        if PUBLIC and s.user in PROTECTED:  # e.g. ?user=anay on the public link: a fresh tester profile instead
            s.user = f"t-{secrets.token_hex(3)}"
        s.scenario = m.get("scenario", s.scenario) if m.get("scenario") in SCENARIOS else s.scenario
        s.learn = m.get("mode") == "learn"
        s.log_ok = m.get("log", True) is not False
        if s.rec is not None:
            s.rec.active = scenario_vocab(s.scenario)
            s.rec.load(s.user_file())
            # older profiles: "help" taught as its own sign -> merge into the app's "Help"
            for k in [k for k in s.rec.samples if canonical_label(k, s.rec.classes) != k]:
                c = canonical_label(k, s.rec.classes)
                s.rec.samples[c] = s.rec.samples.get(c, []) + s.rec.samples.pop(k)
                s.rec.save(s.user_file())
        await s.send(type="ready", model=s.rec is not None,
                     personal={k: len(v) for k, v in (s.rec.samples if s.rec else {}).items()},
                     vocab=s.rec.sign_labels() if s.rec else [], meta_ai=muse.KEY != "" and muse.budget_left() > 0,
                     public=PUBLIC, version=VERSION, logging=s.log_ok, user=s.user)
    elif t == "scenario":  # switch scenario without losing the conversation
        if m.get("scenario") in SCENARIOS:
            s.scenario = m["scenario"]
            if s.rec is not None:
                s.rec.active = scenario_vocab(s.scenario)
                s.changed()
                await s.send(type="personal", personal={k: len(v) for k, v in s.rec.samples.items()},
                             vocab=s.rec.sign_labels())
    elif t == "enroll":
        label = str(m.get("label", "")).strip()[:40]
        if not label:
            return
        if s.rec is not None:
            label = canonical_label(label, s.rec.classes + list(s.rec.samples))
        if s.rec is not None and m.get("restart"):
            s.rec.clear_samples(label)
        s.enroll = {"label": label, "reps": max(1, min(30, int(m.get("reps", 5)))), "n": 0,
                    "save": bool(m.get("save")), "personalise": m.get("personalise", True)}
        s.cancel_timer()
        s.clear_pending()
        await s.send(type="enroll_progress", label=s.enroll["label"], n=0, reps=s.enroll["reps"])
    elif t == "practice":
        s.practice = m.get("label") or None
        s.cancel_timer()
        s.clear_pending()
    elif t == "cancel_enroll":
        s.enroll = None
        s.quiet_until = time.perf_counter() + 1.5
    elif t == "forget":
        if s.rec is not None:
            s.rec.clear_samples(m.get("label"))
            s.rec.save(s.user_file())
            await s.send(type="personal", personal={k: len(v) for k, v in s.rec.samples.items()},
                         vocab=s.rec.sign_labels())
    elif t == "undo":
        if s.pending:
            s.pending.pop()
            s.pending_emb.pop()
            if s.pending_src:
                s.pending_src.pop()
            s.changed()
            if s.pending:
                s.schedule_sentence()
    elif t == "correct":  # user tapped the right word: fix it and learn it privately
        try:
            i, label = int(m.get("index", -1)), str(m.get("label", ""))
        except (TypeError, ValueError):
            return
        if s.rec is not None and 0 <= i < len(s.pending) and label in s.rec.sign_labels():
            s.pending[i] = [{"label": label, "p": 1.0, "cos": 1.0}]
            s.changed()
            s.rec.add_sample(label, s.pending_emb[i])
            s.rec.samples[label] = s.rec.samples[label][-MAX_PERSONAL:]
            s.rec.save(s.user_file())
            await s.send(type="personal", personal={k: len(v) for k, v in s.rec.samples.items()},
                         vocab=s.rec.sign_labels())
            s.schedule_sentence()
    elif t == "speak_now":
        s.cancel_timer()
        await s.speak()
    elif t == "reply":  # hearing person's words, so Spark has both sides of the conversation
        if m.get("text"):
            s.history.append(f"Hearing person: {str(m['text'])[:500]}")
            s.changed()  # a speculative sentence was written without this reply
    elif t == "client_log":  # e.g. the phone's text-to-speech failed: visible in data/server.log
        s.client_logs += 1
        if s.client_logs <= 50:
            print(f"[client {s.user}] {str(m.get('msg'))[:300]}", flush=True)
    elif t == "consent":  # "help improve ISL Bridge" switched on / off
        s.log_ok = bool(m.get("log"))
    elif t == "feedback":
        await s.on_feedback(m)
    elif t == "delete_me":
        n = s.delete_my_data()
        print(f"[privacy] {s.user}: deleted {n} files on request", flush=True)
        await s.send(type="deleted", files=n)
    elif t == "summary":
        if not s.summaries.ok():
            await s.send(type="summary", **muse.plain_summary(s.history, "too many summaries - wait a minute"))
            return
        await s.send(type="summary", **await muse.summarise(s.history, SCENARIOS.get(s.scenario, s.scenario)))
    elif t == "reset":
        s.cancel_timer()
        s.clear_pending()
        s.history = []


@app.get("/api/vocab")
def vocab():
    rec = recognizer()
    labs = rec.sign_labels() if rec else VOCAB["include_signs"]
    return {"signs": labs, "clips": sorted(p.stem for p in CLIPS.glob("*.mp4")), "scenarios": SCENARIOS,
            "include_signs": VOCAB["include_signs"], "cislr_signs": VOCAB["cislr_signs"]}


def asr_keywords():
    """Keyword biasing for Voice Transcribe (max 100): the scenario's words first, not the alphabet's first 100."""
    clips = {p.stem for p in CLIPS.glob("*.mp4")}
    ordered = VOCAB.get("cislr_signs", []) + VOCAB.get("common_signs", []) + VOCAB.get("include_signs", []) + sorted(clips)
    out = []
    for w in ordered:
        if w in clips and w not in out:
            out.append(w)
    return out[:100]


def _plain_gloss(text, clip_labels):
    """Fallback when Muse Spark is unreachable: app signs whose name appears in the text, else fingerspell."""
    lower = {k.lower(): k for k in clip_labels}
    words = dictionary._norm(text)
    gloss, i = [], 0
    while i < len(words) and len(gloss) < 8:
        two = f"{words[i]} {words[i + 1]}" if i + 1 < len(words) else None
        if two and two in lower:
            gloss.append({"sign": lower[two], "dictionary": "", "fingerspell": ""})
            i += 2
            continue
        w = words[i]
        if w in lower:
            gloss.append({"sign": lower[w], "dictionary": "", "fingerspell": ""})
        elif w not in dictionary.STOP and len(w) > 2:
            # an ISLRTC dictionary sign named exactly this word (no "(Maths)"-style qualifier: without Spark
            # nothing checks the meaning) beats fingerspelling English at a Deaf reader
            hit = next((h for h in dictionary.search(w, k=3, min_score=97) if h["entry"].lower() == w), None)
            gloss.append({"sign": "", "dictionary": hit["id"] if hit else "", "fingerspell": "" if hit else w.upper()})
        i += 1
    return {"caption": text, "gloss": gloss, "_latency_ms": 0, "_error": "Muse Spark unavailable, word-by-word"}


_boxes = {"mtime": None, "boxes": {}}


def cutout_boxes():
    """data/clips_sam/boxes.json (signer bounding box per cut-out), re-read when the file changes."""
    f = CUTOUTS / "boxes.json"
    try:
        m = f.stat().st_mtime
        if m != _boxes["mtime"]:
            _boxes.update(mtime=m, boxes=json.loads(f.read_text()))
    except (OSError, ValueError):
        pass
    return _boxes["boxes"]


async def text_to_sign(text, scenario):
    """English/Hindi text -> ISL gloss (Muse Spark, app vocab + ISLRTC dictionary via tool calls) -> playlist."""
    clip_labels = {p.stem for p in CLIPS.glob("*.mp4")}
    try:
        cands = dictionary.candidates_for(text, clip_labels)
        g = await muse.to_gloss(text, clip_labels, SCENARIOS.get(scenario, scenario), cands, dictionary.search)
    except Exception as e:  # the Deaf person must still get the hearing person's words
        print(f"to_gloss failed: {e!r}", flush=True)
        g = _plain_gloss(text, clip_labels)

    def picture(it, word):  # Muse Image picture card for things the Deaf person may not know as a sign
        what = str(it.get("picture") or "").strip()
        if not what or not word:
            return {}
        key = pictures.common_key(word) or re.sub(r"\s*\(.*?\)", "", word).strip()
        return {"picture": pictures.ensure(key, pictures.COMMON.get(key, what))}

    async def item(it):
        if it.get("sign") in clip_labels:
            out = {"sign": it["sign"], "clip": f"/clips/{it['sign']}.mp4", **picture(it, it["sign"])}
            if (CUTOUTS / f"{it['sign']}.mp4").exists():
                out["cutout"] = f"/clips_sam/{it['sign']}.mp4"
                if it["sign"] in cutout_boxes():  # draw just the signer (ml/cutout_boxes.py)
                    out["cutout_box"] = cutout_boxes()[it["sign"]]
            return out
        if it.get("dictionary"):
            try:
                name = next((n for n, p, _, _ in dictionary.entries() if p == it["dictionary"]), it["dictionary"])
                return {"sign": name, "clip": await dictionary.fetch_clip(it["dictionary"]), "dictionary": True,
                        **picture(it, name)}
            except Exception:
                pass
        word = it.get("fingerspell") or it.get("sign") or ""
        return {"fingerspell": word, **picture(it, word)}

    t0 = time.perf_counter()
    playlist = [x for x in await asyncio.gather(*(item(i) for i in g["gloss"])) if x.get("clip") or x.get("fingerspell")]
    return g, playlist, round((time.perf_counter() - t0) * 1000)


def _too_many(rate, r):
    if rate.ok(client_ip(r)):
        return None
    return JSONResponse({"error": "too many requests from this device - wait a minute and try again"},
                        status_code=429)


@app.post("/api/speech")
async def speech(request: Request, audio: UploadFile = File(...), scenario: str = Form("hospital")):
    """Hearing person's speech (16/24 kHz mono WAV) -> transcript -> ISL gloss -> clip playlist."""
    if (busy := _too_many(RATE_HTTP, request)) is not None:
        return busy
    wav = await audio.read(8_000_000)  # ~4 min of 16 kHz audio; hold-to-talk replies are seconds
    t0 = time.perf_counter()
    try:
        asr, asr_ms = await muse.transcribe_wav(wav, keywords=asr_keywords())
    except Exception as e:
        return JSONResponse({"error": f"speech recognition failed: {e}"}, status_code=502)
    text = asr.get("transcript") or asr.get("text") or " ".join(t.get("transcript", "") for t in asr.get("turns", []))
    if not text.strip():
        return JSONResponse({"error": "no speech heard"}, status_code=422)
    g, playlist, fetch_ms = await text_to_sign(text, scenario)
    return {"transcript": text, "caption": g["caption"], "playlist": playlist,
            "latency_ms": {"asr": asr_ms, "gloss": g["_latency_ms"], "clips": fetch_ms,
                           "total": round((time.perf_counter() - t0) * 1000)}}


@app.websocket("/ws/speech")
async def ws_speech(ws: WebSocket):
    """Hold-to-talk streaming: {"type":"start","scenario"} -> binary 16 kHz PCM16 -> {"type":"end"}.
    Server -> {"type":"partial","text"} live captions, then {"type":"result", ...same as /api/speech}."""
    await ws.accept()
    asr, scenario = None, "hospital"
    try:
        while True:
            m = await ws.receive()
            if m.get("type") == "websocket.disconnect":
                break
            if m.get("bytes") is not None:
                if asr:
                    await asr.send(m["bytes"])
                continue
            msg = json.loads(m.get("text") or "{}")
            if msg.get("type") == "start":
                scenario = msg.get("scenario", scenario)
                if not RATE_SPEECH.ok(client_ip(ws)):
                    await ws.send_text(json.dumps({"type": "error", "message": "too many voice replies from this "
                                                   "device - wait a minute, or type the reply"}))
                    continue

                async def partial(text):
                    await ws.send_text(json.dumps({"type": "partial", "text": text}, ensure_ascii=False))
                asr = muse.RealtimeASR(on_partial=partial)
                try:
                    await asr.start(keywords=asr_keywords())
                    await ws.send_text(json.dumps({"type": "ready"}))
                except Exception as e:
                    asr = None
                    await ws.send_text(json.dumps({"type": "error", "message": f"realtime ASR: {e}"}))
            elif msg.get("type") == "end" and asr:
                t0 = time.perf_counter()
                text, asr_ms = await asr.finish()
                asr = None
                if not text:
                    await ws.send_text(json.dumps({"type": "error", "message": "no speech heard"}))
                    continue
                # captions first: the Deaf person can read what was said while Spark puts it into ISL (4-10 s)
                await ws.send_text(json.dumps({"type": "transcript", "text": text}, ensure_ascii=False))
                try:
                    g, playlist, fetch_ms = await text_to_sign(text, scenario)
                except Exception as e:
                    await ws.send_text(json.dumps({"type": "error", "message": f"translation failed: {e}"}))
                    continue
                await ws.send_text(json.dumps({
                    "type": "result", "transcript": text, "caption": g["caption"], "playlist": playlist,
                    "latency_ms": {"asr_after_release": asr_ms, "gloss": g["_latency_ms"], "clips": fetch_ms,
                                   "total": round((time.perf_counter() - t0) * 1000)}}, ensure_ascii=False))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if asr is not None:  # client vanished mid-utterance: close the Meta stream (and log its seconds)
            try:
                await asr.finish(timeout=1)
            except Exception:
                pass


@app.post("/api/lens")
async def lens(request: Request, body: dict):
    """ISL Lens: {"image": data URL of a camera frame} -> object name (Muse Spark vision) -> its ISL sign."""
    if (busy := _too_many(RATE_LENS, request)) is not None:
        return busy
    if not isinstance(body.get("image"), str) or len(body["image"]) > 3_000_000:
        return JSONResponse({"error": "no image"}, status_code=422)
    clip_labels = {p.stem for p in CLIPS.glob("*.mp4")}
    try:
        o = await muse.identify_object(body["image"], clip_labels)
    except Exception as e:
        return JSONResponse({"error": f"lens failed: {e}"}, status_code=502)
    if o.get("app_sign") in clip_labels:  # Spark matched one of the app's own signs: playable + learnable
        l = o["app_sign"]
        return {**o, "word": l, "sign": l, "clip": f"/clips/{l}.mp4", "learnable": True}
    lower = {k.lower(): k for k in clip_labels}
    for word in [o["object"]] + o["alternatives"]:
        if not word:
            continue
        if word.lower() in lower:  # a sign the app knows: playable + learnable with Ghost Coach
            l = lower[word.lower()]
            return {**o, "word": word, "sign": l, "clip": f"/clips/{l}.mp4", "learnable": True}
        hits = dictionary.search(word, k=1, min_score=37)
        if hits:
            try:
                return {**o, "word": word, "sign": hits[0]["entry"], "learnable": False,
                        "clip": await dictionary.fetch_clip(hits[0]["id"])}
            except Exception:
                continue
    return {**o, "word": o["object"], "sign": None, "clip": None, "learnable": False}


@app.post("/api/text")
async def text_endpoint(request: Request, body: dict):
    """Typed text -> ISL clips (doctor can type in a noisy ward; also used for testing). {"text", "scenario"}"""
    if (busy := _too_many(RATE_HTTP, request)) is not None:
        return busy
    t0 = time.perf_counter()
    text = str(body.get("text", "")).strip()[:500]
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=422)
    g, playlist, fetch_ms = await text_to_sign(text, body.get("scenario", "hospital"))
    return {"transcript": text, "caption": g["caption"], "playlist": playlist,
            "latency_ms": {"gloss": g["_latency_ms"], "clips": fetch_ms,
                           "total": round((time.perf_counter() - t0) * 1000)}}


@app.get("/api/coach/{label}")
def coach_ref(label: str):
    """Ghost trajectory (body space, synced to /clips/<label>.mp4) for the Learn page."""
    if not _safe_name(label):
        return JSONResponse({"error": "bad label"}, status_code=400)
    r = coach.reference(label)
    if not r:
        return JSONResponse({"error": "no reference"}, status_code=404)
    desc = CLIPS / "descriptions.json"  # Muse Spark (vision) how-to, see ml/describe_signs.py
    if desc.exists():
        r["howto"] = json.loads(desc.read_text(encoding="utf-8")).get(label)
    return r


def _safe_name(s):
    return bool(s) and ".." not in s and "/" not in s and "\\" not in s


@app.get("/api/ref/{label}")
def ref_video(label: str, i: int = 0):
    """INCLUDE reference video for a sign (used by Record mode so the signer can learn it)."""
    if not _safe_name(label):
        return JSONResponse({"error": "bad label"}, status_code=400)
    vids = sorted((DATA / "videos" / label).glob("*")) if (DATA / "videos" / label).is_dir() else []
    vids = [v for v in vids if v.suffix.lower() in (".mov", ".mp4")]
    if not vids:
        return JSONResponse({"error": "no reference"}, status_code=404)
    return FileResponse(vids[i % len(vids)], media_type="video/mp4")


@app.get("/api/recordings/{user}")
def recordings(user: str):
    if not _safe_name(user):
        return {}
    d = RECORDINGS / user
    return {p.name: len(list(p.glob("*.npz"))) for p in sorted(d.iterdir())} if d.is_dir() else {}


@app.get("/api/health")
def health():
    """For monitoring and the end-to-end tests: is the model loaded, which build, how much credit is left."""
    return {"ok": recognizer() is not None, "version": VERSION, "public": PUBLIC, "sessions": len(SESSIONS),
            "uptime_s": round(time.time() - STARTED), "meta_ai": bool(muse.KEY) and muse.budget_left() > 0,
            "log_sync": sync.enabled()}


@app.post("/api/compose")
async def compose(body: dict):
    """Debug: {"candidates": [[{"label","p"}...]...], "history": [], "scenario": "hospital"}"""
    if PUBLIC:  # an open proxy to Muse Spark on the internet would only spend the credit
        return JSONResponse({"error": "not available"}, status_code=404)
    return await muse.build_sentence(body["candidates"], body.get("history", []),
                                     SCENARIOS.get(body.get("scenario", "hospital"), "hospital"))


CLIPS.mkdir(parents=True, exist_ok=True)
app.mount("/clips", StaticFiles(directory=CLIPS), name="clips")
CUTOUTS.mkdir(parents=True, exist_ok=True)
app.mount("/clips_sam", StaticFiles(directory=CUTOUTS), name="clips_sam")
dictionary.CACHE.mkdir(parents=True, exist_ok=True)
app.mount("/clips_dict", StaticFiles(directory=dictionary.CACHE), name="clips_dict")
pictures.CACHE.mkdir(parents=True, exist_ok=True)
app.mount("/pictures", StaticFiles(directory=pictures.CACHE), name="pictures")
DIST = ROOT / "web" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
