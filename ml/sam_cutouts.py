"""SAM 3.1 video segmentation of every sign clip -> signer cut-outs for the AR interpreter.

    python ml/sam_cutouts.py [--only Pain,Doctor] [--workers 4]

data/clips/<label>.mp4 -> Meta Model API (sam-3.1, prompt "person") -> per-frame masks ->
data/clips_sam/<label>.mp4: a "stacked" H.264 video, colour on top, mask (white = signer) below. Browsers on
every phone play H.264; the web app recombines the halves into a transparent signer on a canvas (alpha video
codecs are not portable: iOS has no VP9 alpha). Resumable; spend is logged to data/meta_spend.json.
"""
import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
os.environ.setdefault("META_SPEND_TAG", "sam_cutouts")
import meta_sam_parser as msp  # noqa: E402
import muse  # noqa: E402

SRC = ROOT / "data" / "clips"
OUT = ROOT / "data" / "clips_sam"
PRICE_FRAME = 0.20 / 1000
BUDGET_USD = 10.0


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames, fps


def masks_from_output(text, n, h, w):
    """SAM output text -> (n, h, w) uint8 masks of the single most confident person per frame."""
    p = msp.video_segmentation_format().create_parser()
    p.push(text)
    res = p.finish(msp.CompletedOutcome()).result
    best = {}  # frame -> (confidence, record)
    for r in res.records:
        if isinstance(r, msp.SegmentationMaskRecord) and r.frame is not None:
            i, c = r.frame.frame_index, r.confidence or 0.0
            if i not in best or c > best[i][0]:
                best[i] = (c, r)
    out = np.zeros((n, h, w), np.uint8)
    for i, (_, r) in best.items():
        if i >= n:
            continue
        m = np.frombuffer(msp.decode_mask_to_raster(r.mask), np.uint8).reshape(r.mask.height, r.mask.width)
        x0, y0 = int(r.bounds.left), int(r.bounds.top)
        x1, y1 = min(w, x0 + m.shape[1]), min(h, y0 + m.shape[0])
        out[i, y0:y1, x0:x1] = m[: y1 - y0, : x1 - x0] * 255
    for i in range(n):  # frames SAM skipped: hold the previous mask
        if i not in best and i > 0:
            out[i] = out[i - 1]
    return out, len(best)


def write_stacked(dst, frames, masks, fps):
    h, w = frames[0].shape[:2]
    tmp = dst.with_suffix(".tmp.mp4")
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                             "-pix_fmt", "bgr24", "-s", f"{w}x{h * 2}", "-r", f"{fps}", "-i", "-",
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart", str(tmp)], stdin=subprocess.PIPE)
    k = np.ones((3, 3), np.uint8)
    for f, m in zip(frames, masks):
        soft = cv2.GaussianBlur(cv2.erode(m, k), (5, 5), 0)  # clean, slightly feathered edge
        proc.stdin.write(np.vstack([f, cv2.cvtColor(soft, cv2.COLOR_GRAY2BGR)]).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    tmp.replace(dst)


async def cutout(label, sem):
    src, dst = SRC / f"{label}.mp4", OUT / f"{label}.mp4"
    if dst.exists():
        return "exists"
    async with sem:
        if muse.spent() > BUDGET_USD + 1.0:
            return "budget"
        frames, fps = await asyncio.to_thread(read_frames, src)
        h, w = frames[0].shape[:2]
        client = muse._client.with_options(timeout=180, max_retries=1)  # never hang for the SDK's 10 min default
        t0 = time.perf_counter()
        with src.open("rb") as fh:
            f = await client.files.create(file=fh, purpose="user_data")
        t1 = time.perf_counter()
        r = await client.responses.create(model="sam-3.1", input=[{"role": "user", "content": [
            {"type": "input_text", "text": "person"}, {"type": "input_video", "file_id": f.id}]}],
            metadata={"mask_encoding": "one_bit", "include_confidence": "true"})
        billed = max(50, getattr(r.usage, "video_frames_processed", None) or len(frames))
        muse._log_spend("sam", billed * PRICE_FRAME, frames=billed)
        masks, got = masks_from_output(r.output_text, len(frames), h, w)
        if got == 0:
            return "no person found"
        await asyncio.to_thread(write_stacked, dst, frames, masks, fps)
        return f"ok {got}/{len(frames)} frames (upload {t1 - t0:.1f}s, sam {time.perf_counter() - t1:.1f}s)"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    labels = args.only.split(",") if args.only else sorted(p.stem for p in SRC.glob("*.mp4"))
    sem = asyncio.Semaphore(args.workers)

    async def run(l):
        try:
            return l, await cutout(l, sem)
        except Exception as e:
            return l, f"FAILED {e!r}"[:200]

    for i, fut in enumerate(asyncio.as_completed([run(l) for l in labels]), 1):
        l, msg = await fut
        print(f"[{i}/{len(labels)}] {l}: {msg} (spent ${muse.spent():.2f})", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
