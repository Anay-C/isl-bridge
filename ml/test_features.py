"""Feature regression of the live app over its real WebSocket: every /ws/sign message type, the "help improve"
consent, "✏️ Wrong?" reports and the public-mode limits. Uses real Muse Spark (a few cents per run).

    .venv-ml/Scripts/python ml/test_features.py                      # local server, port 8000
    .venv-ml/Scripts/python ml/test_features.py https://<space>.hf.space
    .venv-ml/Scripts/python ml/test_features.py http://127.0.0.1:8020 data/space_build/data   # staged bundle

Each check prints PASS/FAIL with what came back; exit code 1 if any check failed. File checks (logs written or
not) only run against a server on this machine (same data/ folder). Test users are named wstest_* and their
files are removed at the end.
"""
import asyncio
import glob
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ws import frames_msgs  # noqa: E402

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
WS = BASE.replace("https://", "wss://").replace("http://", "ws://") + "/ws/sign"
LOCAL = "127.0.0.1" in BASE or "localhost" in BASE or len(sys.argv) > 2  # data folder given: same machine
KP = Path(__file__).resolve().parent.parent / "data"  # test clips (INCLUDE keypoints)
# the server's data folder, for the file checks (a staged Space bundle has its own: data/space_build/data)
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else KP
DOCTOR = sorted(glob.glob(str(KP / "keypoints" / "Doctor" / "*.npz")))
MEDICINE = sorted(glob.glob(str(KP / "keypoints" / "Medicine" / "*.npz")))
RUN = time.strftime("%H%M%S")
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def http(path, body=None, timeout=60, headers=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


class Client:
    def __init__(self, ws):
        self.ws, self.msgs, self.clock = ws, [], 0.0
        self.task = asyncio.create_task(self._read())

    async def _read(self):
        while True:
            self.msgs.append(json.loads(await self.ws.recv()))

    async def send(self, **m):
        await self.ws.send(json.dumps(m))

    async def sign(self, path, speed=2.0):
        sign, rest, fps = frames_msgs(path)
        for f in sign + rest:
            self.clock += 1000 / fps
            await self.ws.send(json.dumps(dict(f, t=round(self.clock))))
            await asyncio.sleep(1 / fps / speed)

    async def wait(self, typ, timeout=15.0, since=0):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            for m in self.msgs[since:]:
                if m.get("type") == typ:
                    return m
            await asyncio.sleep(0.1)
        return None


async def main():
    st, h = http("/api/health")
    check("GET /api/health", st == 200 and h.get("ok"), f"{h}")
    public = bool(h and h.get("public"))

    user = f"wstest_feat_{RUN}"
    async with websockets.connect(WS, max_size=None, open_timeout=30) as ws:
        c = Client(ws)
        # start with "help improve" OFF: nothing may be logged for this user
        await c.send(type="hello", user=user, scenario="hospital", log=False)
        r = await c.wait("ready", timeout=30)
        check("hello -> ready (model, limits, consent echoed)",
              r is not None and r.get("model") is True and r.get("logging") is False and "meta_ai" in r,
              f"vocab {len(r.get('vocab', [])) if r else 0}, meta_ai {r and r.get('meta_ai')}, public {r and r.get('public')}")

        # conversation: sign -> sign msg -> sentence (with a sentence id)
        n0 = len(c.msgs)
        await c.sign(DOCTOR[0])
        s = await c.wait("sign", since=n0)
        sent = await c.wait("sentence", since=n0, timeout=25)
        check("sign -> sign + sentence (+ sid)", s is not None and sent is not None and isinstance(sent.get("sid"), int),
              f"{s and s['top'][0]['label']} -> {sent and sent.get('english')!r} (spark error: {sent and sent.get('_error')})")
        if LOCAL:
            check("consent off -> no keypoint log written", not (DATA / "live_log" / user).exists())

        # "✏️ Wrong?" with consent off: learned, but no keypoints kept
        n0 = len(c.msgs)
        await c.send(type="feedback", sid=sent["sid"] if sent else -1, signs=[{"i": 0, "meant": "Medicine"}],
                     note="regression test", device="test")
        fb = await c.wait("feedback_ok", since=n0)
        p = await c.wait("personal", since=n0)
        check("feedback (consent off) -> learned, 0 keypoint files",
              fb is not None and fb.get("learned") == ["Medicine"] and fb.get("saved_keypoints") == 0
              and p is not None and p["personal"].get("Medicine", 0) >= 1, f"{fb}")

        # switch "help improve" ON: the next sign is logged, a report keeps its keypoints
        await c.send(type="consent", log=True)
        await asyncio.sleep(0.5)
        n0 = len(c.msgs)
        await c.sign(DOCTOR[1])
        sent2 = await c.wait("sentence", since=n0, timeout=25)
        if LOCAL:
            check("consent on -> keypoint log written", (DATA / "live_log" / user).is_dir()
                  and any((DATA / "live_log" / user).glob("*.npz")))
        n0 = len(c.msgs)
        await c.send(type="feedback", sid=sent2["sid"] if sent2 else -1, signs=[{"i": 0, "meant": "Pain"}])
        fb = await c.wait("feedback_ok", since=n0)
        check("feedback (consent on) -> keypoints of the sign saved", fb is not None and fb.get("saved_keypoints") == 1, f"{fb}")
        n0 = len(c.msgs)
        await c.send(type="feedback", note="general note from the regression test")
        fb = await c.wait("feedback_ok", since=n0)
        check("note-only feedback -> saved", fb is not None, f"{fb}")
        if LOCAL:
            lines = (DATA / "feedback" / f"{user}.jsonl").read_text(encoding="utf-8").splitlines()
            rows = [json.loads(x) for x in lines]
            check("feedback file: 3 reports, fixes + note", len(rows) == 3 and rows[0]["fixes"][0]["meant"] == "Medicine"
                  and rows[2]["note"].startswith("general"), f"{len(rows)} rows")
        # bad reports must not break anything
        for bad in ({"sid": "x"}, {"sid": 999, "signs": [{"i": "a"}]}, {"signs": "nope"}, {"note": 5}):
            await c.send(type="feedback", **bad)

        # undo: sign, then undo before the pause ends -> no sentence for it
        # (over a slow link the 0.5 s pause can end before the undo arrives: then the sentence is already out and
        # "✏️ Wrong?" is the fix; only a sentence arriving AFTER the undo means undo failed)
        n0 = len(c.msgs)
        await c.sign(MEDICINE[0], speed=4.0)
        await c.wait("sign", since=n0)
        n_undo = len(c.msgs)
        await c.send(type="undo")
        await asyncio.sleep(3)
        late = any(m.get("type") == "sentence" for m in c.msgs[n0:n_undo])
        after = any(m.get("type") == "sentence" for m in c.msgs[n_undo:])
        check("undo removes the pending sign" + (" (sentence was out before the undo: link slower than the pause)" if late else ""),
              not after, f"messages {[m.get('type') for m in c.msgs[n0:]]}")

        # correct: sign Doctor, tap 'Medicine' -> sentence built from Medicine
        n0 = len(c.msgs)
        await c.sign(DOCTOR[2], speed=4.0)
        s = await c.wait("sign", since=n0)
        await c.send(type="correct", index=s["index"] if s else 0, label="Medicine")
        sent = await c.wait("sentence", since=n0, timeout=25)
        check("correct -> sentence uses it", sent is not None and "Medicine" in [x[0]["label"] for x in sent.get("cands", [])],
              f"{sent and sent.get('english')!r}")

        # forget
        n0 = len(c.msgs)
        await c.send(type="forget", label="Medicine")
        p = await c.wait("personal", since=n0)
        check("forget -> sample removed", p is not None and "Medicine" not in p["personal"], f"{p and p['personal']}")

        # reply + speak_now with nothing pending + summary
        await c.send(type="reply", text="Please take this medicine after food.")
        await c.send(type="speak_now")
        n0 = len(c.msgs)
        await c.send(type="summary")
        sm = await c.wait("summary", since=n0, timeout=45)
        check("summary (Spark)", sm is not None and bool(sm.get("summary_english")),
              f"{(sm or {}).get('summary_english', '')[:80]!r}")

        # scenario switch keeps the session
        n0 = len(c.msgs)
        await c.send(type="scenario", scenario="general")
        p = await c.wait("personal", since=n0)
        check("scenario -> general vocabulary", p is not None and len(p.get("vocab", [])) > 150, f"vocab {len(p.get('vocab', [])) if p else 0}")

        # teach a brand-new sign -> progress x3 + enroll_done, then forget it
        n0 = len(c.msgs)
        await c.send(type="enroll", label="Insulin", reps=3)
        for f in MEDICINE[1:4]:
            await c.sign(f, speed=4.0)
            await asyncio.sleep(0.3)
        d = await c.wait("enroll_done", since=n0, timeout=12)
        prog = [m["n"] for m in c.msgs[n0:] if m.get("type") == "enroll_progress"]
        check("teach a new sign -> done", d is not None and prog[-1:] == [3], f"progress {prog} self-test {d and d.get('self_test')}/{d and d.get('kept')}")
        await c.send(type="forget", label="Insulin")

        # cancel_enroll, then conversation still works
        await c.send(type="enroll", label="Pain", reps=5)
        await c.send(type="cancel_enroll")
        await asyncio.sleep(2)
        n0 = len(c.msgs)
        await c.sign(DOCTOR[3])
        s = await c.wait("sign", since=n0)
        check("cancel_enroll -> back to conversation", s is not None, f"{s and s['top'][0]['label']}")

        # practice (Learn page)
        await asyncio.sleep(2)
        n0 = len(c.msgs)
        await c.send(type="practice", label="Doctor")
        await c.sign(DOCTOR[4 % len(DOCTOR)])
        pr = await c.wait("practice_result", since=n0, timeout=12)
        check("practice -> practice_result", pr is not None, f"stars {pr and pr.get('stars')}")
        await c.send(type="practice", label=None)

        await c.send(type="client_log", msg="regression test")
        await c.send(type="reset")
        await asyncio.sleep(0.5)
        errs = [m for m in c.msgs if m.get("type") == "error"]
        check("no error messages from the server", not errs, f"{errs[:2]}")

        # "Delete my data": profile, keypoint logs and reports of this profile are gone
        n0 = len(c.msgs)
        await c.send(type="delete_me")
        dm = await c.wait("deleted", since=n0)
        gone = (not LOCAL) or not any(x.exists() for x in (DATA / "live_log" / user, DATA / "feedback" / user,
                                                           DATA / "feedback" / f"{user}.jsonl", DATA / "users" / f"{user}.json"))
        check("delete my data -> files removed", dm is not None and dm.get("files", 0) > 0 and gone, f"{dm}")
        c.task.cancel()

    if public:  # the LAN owner's profile can't be opened (or wiped) from the public link
        async with websockets.connect(WS, max_size=None, open_timeout=30) as ws:
            c = Client(ws)
            await c.send(type="hello", user="anay", scenario="hospital", log=False)
            r = await c.wait("ready", timeout=30)
            check("public: protected profile 'anay' -> a fresh tester id", r is not None and r.get("user", "").startswith("t-"),
                  f"got {r and r.get('user')}")
            c.task.cancel()

    # HTTP: typed reply -> ISL playlist; then the per-device limit (empty texts are refused before any Meta call)
    st, j = http("/api/text", {"text": "Take one tablet after food", "scenario": "hospital"})
    check("typed reply -> ISL playlist", st == 200 and j.get("playlist"), f"{st} {[x.get('sign') or x.get('fingerspell') for x in (j or {}).get('playlist', [])]}")
    if public:  # (the burst uses ISL Lens, 8/min, so the typed-reply limit of the next run is not used up)
        codes = [http("/api/lens", {})[0] for _ in range(12)]
        check("public: Meta-backed routes are rate-limited per device", 429 in codes, f"codes {sorted(set(codes))}")
        st, _ = http("/api/lens", {}, headers={"X-Forwarded-For": "1.2.3.4"})
        check("public: a forged X-Forwarded-For does not reset the limit", st == 429, f"{st}")
        st, _ = http("/api/compose", {"candidates": []})
        check("public: debug route /api/compose is off", st == 404, f"{st}")

    if LOCAL:  # remove the test user's files
        for d in (DATA / "live_log" / user, DATA / "feedback" / user):
            shutil.rmtree(d, ignore_errors=True)
        for f in (DATA / "feedback" / f"{user}.jsonl", DATA / "users" / f"{user}.json"):
            f.unlink(missing_ok=True)
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


asyncio.run(main())
