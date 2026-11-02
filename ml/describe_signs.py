"""Muse Spark (vision) writes beginner instructions for every sign clip -> data/clips/descriptions.json

    python ml/describe_signs.py [--only Pain,Doctor] [--force]

Grounded, because a plain video prompt hallucinated (it described PAIN as a fist at the chest; it is a claw hand):
Spark gets 6 key frames of the clip AND facts measured from our pose tracking (which hands move, at what height,
how much), and is told to trust the measurements and leave out anything it cannot see.
"""
import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "ml"))
os.environ.setdefault("META_SPEND_TAG", "describe_signs")
import coach  # noqa: E402
import muse  # noqa: E402
from features import npz_frames, resample  # noqa: E402

CLIPS = ROOT / "data" / "clips"
OUT = CLIPS / "descriptions.json"

INSTRUCTIONS = """You are an Indian Sign Language teacher. From the frames (in time order, NOT mirrored: the
signer's right hand appears on the viewer's left) and the measured pose-tracking facts (trust these for which
hand moves and where), tell a beginner how to make this sign. Only describe what you can actually see; if a
detail is unclear, leave it out rather than guess."""
SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": {"type": "string"},
                  "description": "2-4 short steps, second person ('your right hand ...')"},
        "tip": {"type": "string", "description": "one short memory hook, or empty"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["steps", "tip", "confidence"], "additionalProperties": False,
}


def facts(label):
    f = resample(npz_frames(coach._source(label)), 48)
    nose = float(np.median(f[:, coach.NOSE_Y]))
    out = []
    for side in ("right", "left"):
        b, _, use = coach._hand(f, side)
        if use.mean() < 0.2:
            out.append(f"signer's {side} hand: not used (stays down)")
            continue
        y = float(np.median(b[use, 0, 1]))
        where = "face height" if y < nose + 0.5 else "chest height" if y < 1.0 else "stomach height"
        path = float(np.linalg.norm(np.diff(b[use, 0, :2], axis=0), axis=1).sum())
        move = "large movement" if path > 2 else "small movement" if path > 0.6 else "held almost still"
        out.append(f"signer's {side} hand: used {use.mean():.0%} of the sign, mostly at {where}, {move}")
    return out


def key_frames(label, n=6):
    cap = cv2.VideoCapture(str(CLIPS / f"{label}.mp4"))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(total * 0.15, total * 0.85, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            fr = cv2.resize(fr, (int(240 * fr.shape[1] / fr.shape[0]), 240))
            out.append(base64.b64encode(cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode())
    return out


async def describe(label):
    content = [{"type": "input_text", "text": f"Sign meaning: {label}\nMeasured: " + "; ".join(facts(label))}]
    content += [{"type": "input_image", "image_url": f"data:image/jpeg;base64,{b}"} for b in key_frames(label)]
    r = await muse._client.responses.create(
        model=muse.SPARK, instructions=INSTRUCTIONS, input=[{"role": "user", "content": content}],
        reasoning={"effort": "low"}, max_output_tokens=2000,
        text={"format": {"type": "json_schema", "name": "sign_howto", "schema": SCHEMA, "strict": True}})
    u = r.usage
    muse._log_spend("spark", u.input_tokens * muse.PRICE_IN + u.output_tokens * muse.PRICE_OUT,
                    input_tokens=u.input_tokens, output_tokens=u.output_tokens)
    return json.loads(r.output_text)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    labels = args.only.split(",") if args.only else sorted(p.stem for p in CLIPS.glob("*.mp4"))
    todo = [l for l in labels if args.force or l not in done]
    sem = asyncio.Semaphore(4)

    async def one(l):
        async with sem:
            try:
                return l, await describe(l)
            except Exception as e:
                return l, {"error": repr(e)[:200]}

    for i, fut in enumerate(asyncio.as_completed([one(l) for l in todo]), 1):
        l, d = await fut
        if "error" not in d:
            done[l] = d
            OUT.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[{i}/{len(todo)}] {l}: {d.get('confidence', d.get('error'))} | {' / '.join(d.get('steps', []))[:160]}",
              flush=True)
    print(f"{len(done)} descriptions, spent ${muse.spent():.2f}")


if __name__ == "__main__":
    asyncio.run(main())
