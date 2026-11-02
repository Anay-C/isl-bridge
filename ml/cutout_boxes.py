"""Bounding box of the signer in every SAM 3.1 cut-out clip, so the AR interpreter can draw just the person (bigger)
instead of the whole video frame around them.

    python ml/cutout_boxes.py        # -> data/clips_sam/boxes.json  {label: [x0, y0, x1, y1]} in 0..1 of the colour half

A stacked clip is colour on top, mask below (ml/sam_cutouts.py); the box is the union of the mask over all frames
(so moving hands never leave it) plus a small margin. Re-run after new cut-outs are made (it skips known clips).
"""
import json
from pathlib import Path

import cv2
import numpy as np

SAM = Path(__file__).resolve().parent.parent / "data" / "clips_sam"
OUT = SAM / "boxes.json"


def box(path, margin=0.03):
    cap = cv2.VideoCapture(str(path))
    union = None
    w = h = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[0] // 2, f.shape[1]
        m = f[h:, :, 0] > 127
        union = m if union is None else (union | m)
    if union is None or not union.any():
        return None
    ys, xs = np.nonzero(union)
    x0, x1 = xs.min() / w - margin, (xs.max() + 1) / w + margin
    y0, y1 = ys.min() / h - margin, (ys.max() + 1) / h + margin
    return [round(max(0.0, x0), 4), round(max(0.0, y0), 4), round(min(1.0, x1), 4), round(min(1.0, y1), 4)]


def main():
    boxes = json.loads(OUT.read_text()) if OUT.exists() else {}
    for p in sorted(SAM.glob("*.mp4")):
        if p.stem not in boxes and not p.name.endswith(".tmp.mp4"):
            b = box(p)
            if b:
                boxes[p.stem] = b
    OUT.write_text(json.dumps(boxes, indent=0))
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes.values()]
    print(f"{len(boxes)} boxes -> {OUT} | median box covers {np.median(areas):.0%} of the frame")


if __name__ == "__main__":
    main()
