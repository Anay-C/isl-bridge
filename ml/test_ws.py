"""End-to-end test of the live server: every sign streamed over the real WebSocket, like a phone.

For each app sign, one clip (held-out by train.py when there is one, else a training clip, flagged) is sent frame by
frame at its own frame rate as {"type":"frame", w, h, t, pose, lh, rh} (normalised landmarks, exactly what
web/src/vision.ts sends), followed by hands-down frames. Several clients run at once (each its own session, as
several phones would). Recorded per clip: the "sign" messages, the "sentence", latency from hands-down to the
first sign / the sentence, "error" messages, dropped connections.
--fuzz also sends malformed messages before every clip (bad JSON, wrong array sizes, NaN, missing fields, binary,
unknown types, bad indices): the session must survive all of them.

    MODEL_API_KEY= python -m uvicorn server.app:app --port 8765     # no Meta key: tests the fallback sentence path
    python ml/test_ws.py --url ws://127.0.0.1:8765/ws/sign --clients 8 --fuzz
"""
import argparse
import asyncio
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import websockets

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "ml"))
from test_signs import ROOTS, scenario_vocab  # noqa: E402

FUZZ = [
    "not json at all", "[1, 2, 3]", "null", '{"type": 5}', '{"no_type": true}', '{"type": "frame"}',
    '{"type": "frame", "w": "abc", "h": null, "pose": [1, 2, 3], "lh": "x", "rh": {}}',
    json.dumps({"type": "frame", "w": 640, "h": 480, "pose": [float("nan")] * 132, "lh": [0.5] * 62, "rh": None}),
    json.dumps({"type": "frame", "w": -1, "h": 0, "pose": [[0.5] * 4] * 33, "lh": [[0.5] * 3] * 21, "rh": None}),
    '{"type": "correct", "index": "x", "label": 7}', '{"type": "correct"}', '{"type": "undo"}',
    '{"type": "enroll", "label": "", "reps": "many"}', '{"type": "scenario", "scenario": "moon"}',
    '{"type": "practice", "label": null}', '{"type": "forget"}', '{"type": "unknown_kind"}',
]


def pick_clips(model, labels):
    """One clip per sign: a held-out one if train.py kept one, else any recording (flagged as seen)."""
    ck = torch.load(model, map_location="cpu", weights_only=False)
    held = {}
    for key in ck.get("test_include", []) + ck.get("test_cislr", []):
        src, rel = key.split("/", 1)
        held.setdefault(key.split("/")[1], ROOTS[src] / rel)
    out = []
    for lab in labels:
        if lab in held:
            out.append((lab, held[lab], True))
            continue
        for src in ("cislr", "include", "islrtc"):
            fs = sorted((ROOTS[src] / lab).glob("*.npz")) if (ROOTS[src] / lab).exists() else []
            if fs:
                out.append((lab, fs[0], False))
                break
    return out


def frames_msgs(path):
    d = np.load(path)
    w, h = int(d["width"]), int(d["height"])
    if w <= 1 or h <= 1:  # mined clips are stored in pixel units with w = h = 1: not a phone stream
        raise ValueError("pixel-unit clip")
    fps = float(d["fps"]) or 25.0
    msgs = []
    for t in range(len(d["pose"])):
        msgs.append({"type": "frame", "w": w, "h": h, "pose": [round(float(x), 4) for x in d["pose"][t].ravel()],
                     "lh": [round(float(x), 4) for x in d["lh"][t].ravel()] if d["lh_p"][t] else None,
                     "rh": [round(float(x), 4) for x in d["rh"][t].ravel()] if d["rh_p"][t] else None})
    rest = dict(msgs[-1], lh=None, rh=None)
    return msgs, [dict(rest) for _ in range(30)], fps


async def client(k, url, queue, results, scenario, fuzz, speed):
    rng = random.Random(k)
    clock = 1000.0 * k  # client capture clock (ms), like performance.now()
    drops = 0
    while queue:
        try:
            async with websockets.connect(url, max_size=None, open_timeout=20) as ws:
                await ws.send(json.dumps({"type": "hello", "user": f"wstest{k}", "scenario": scenario}))
                while True:  # skip until "ready"
                    m = json.loads(await asyncio.wait_for(ws.recv(), 30))
                    if m.get("type") == "ready":
                        break
                while queue:
                    lab, path, held = queue.pop()
                    try:
                        sign_frames, rest, fps = frames_msgs(path)
                    except ValueError:
                        continue
                    if fuzz:
                        for f in rng.sample(FUZZ, 6):
                            await ws.send(f)
                        await ws.send(b"\x00\x01binary")
                    got, errors, t_down, t_sign, t_sent, sentence = [], [], None, None, None, None

                    async def reader():
                        nonlocal t_sign, t_sent, sentence
                        while True:
                            m = json.loads(await ws.recv())
                            ty = m.get("type")
                            if ty == "sign":
                                got.append([c["label"] for c in m["top"]])
                                t_sign = t_sign or time.perf_counter()
                            elif ty == "error":
                                errors.append(m.get("message"))
                            elif ty == "sentence":
                                sentence, t_sent = m, time.perf_counter()
                                return
                    rd = asyncio.create_task(reader())
                    for i, f in enumerate(sign_frames + rest):
                        clock += 1000.0 / fps
                        f["t"] = round(clock)
                        await ws.send(json.dumps(f))
                        if i == len(sign_frames):
                            t_down = time.perf_counter()
                        await asyncio.sleep(1.0 / fps / speed)
                    try:
                        await asyncio.wait_for(rd, 20)
                    except asyncio.TimeoutError:
                        rd.cancel()
                    results.append({"label": lab, "held_out": held, "file": str(path.relative_to(DATA)),
                                    "signs": got, "sentence": (sentence or {}).get("english"),
                                    "errors": errors,
                                    "first_sign_s": round(t_sign - t_down, 2) if t_sign and t_down else None,
                                    "sentence_s": round(t_sent - t_down, 2) if t_sent and t_down else None})
                    clock += 3000.0  # a pause between people's turns
        except (websockets.ConnectionClosed, OSError, asyncio.TimeoutError) as e:
            drops += 1
            print(f"client {k}: connection lost ({type(e).__name__}: {e}); reconnecting", flush=True)
            if drops > 5:
                return drops
            await asyncio.sleep(1)
    return drops


async def main_async(args):
    labels = sorted(scenario_vocab(args.scenario))
    clips = pick_clips(args.model, labels)
    if args.limit:
        clips = clips[:args.limit]
    queue = list(reversed(clips))
    results = []
    t0 = time.time()
    drops = await asyncio.gather(*[client(k, args.url, queue, results, args.scenario, args.fuzz, args.speed)
                                   for k in range(args.clients)])
    n = len(results)
    top1 = [r for r in results if r["signs"] and r["signs"][0][0] == r["label"]]
    any1 = [r for r in results if any(s[0] == r["label"] for s in r["signs"])]
    exact = [r for r in results if len(r["signs"]) == 1 and r["signs"][0][0] == r["label"]]
    silent = [r for r in results if not r["signs"]]
    nosent = [r for r in results if r["signs"] and r["sentence"] is None]
    errs = Counter(e for r in results for e in r["errors"])
    lat = [r["first_sign_s"] for r in results if r["first_sign_s"] is not None]
    lat_s = [r["sentence_s"] for r in results if r["sentence_s"] is not None]
    held = [r for r in results if r["held_out"]]
    print(f"\n{n} signs streamed in {time.time() - t0:.0f}s by {args.clients} clients | dropped connections: {sum(drops)}")
    print(f"first sign right {len(top1)}/{n} | sign among detected {len(any1)}/{n} | exactly that one sign {len(exact)}/{n}"
          f" | held-out clips: first sign right {sum(1 for r in held if r in top1)}/{len(held)}")
    print(f"no sign at all: {len(silent)} {[r['label'] for r in silent]}")
    print(f"signs but no sentence: {len(nosent)} {[r['label'] for r in nosent][:20]}")
    print(f"error messages from the server: {dict(errs)}")
    if lat:
        print(f"latency hands-down -> first sign: median {np.median(lat):.2f}s p90 {np.percentile(lat, 90):.2f}s | "
              f"-> sentence: median {np.median(lat_s):.2f}s p90 {np.percentile(lat_s, 90):.2f}s")
    out = Path(args.out) if args.out else DATA / "sign_test" / f"ws_{args.scenario}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"n": n, "dropped": sum(drops), "results": sorted(results, key=lambda r: r["label"])}, indent=1))
    print("->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8765/ws/sign")
    ap.add_argument("--model", default=str(DATA / "model.pt"), help="checkpoint whose held-out clips to prefer")
    ap.add_argument("--scenario", default="general")
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--fuzz", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0, help="frame pacing speed-up (1 = real time)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
