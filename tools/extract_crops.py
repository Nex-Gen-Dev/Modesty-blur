"""Step 1 of building your own training set: cut people out of photos and videos.

    python tools/extract_crops.py my_videos_and_photos/ --out dataset/unlabeled --every 15

Walks a folder of videos (mp4, mov, ...) and images, finds every person with YOLO, and saves each
person as a tall crop. Near-duplicate crops (same person in neighbouring frames) are skipped so you
do not have to label hundreds of almost identical pictures.

Keep the footage and crops PRIVATE: they show real people. Do not upload or share them.
Next step: python tools/label.py dataset/unlabeled --out dataset
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ahash(crop, size=8):
    g = cv2.cvtColor(cv2.resize(crop, (size, size * 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return (g > g.mean()).flatten()


def frames(path: Path, every: int):
    if path.suffix.lower() in IMAGE:
        im = cv2.imread(str(path))
        if im is not None:
            yield im
        return
    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i % every == 0:
            yield im
        i += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="folder with videos and/or photos")
    ap.add_argument("--out", default="dataset/unlabeled")
    ap.add_argument("--every", type=int, default=15, help="use every Nth video frame")
    ap.add_argument("--min-height", type=int, default=96, help="skip people smaller than this many pixels tall")
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO weights (plain detector is enough)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--dup-distance", type=int, default=10, help="bit difference below which two crops count as duplicates")
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seen = []
    saved = 0
    files = sorted(p for p in Path(a.source).rglob("*") if p.suffix.lower() in VIDEO | IMAGE)
    if not files:
        sys.exit(f"No videos or photos found in {a.source}")
    for f in files:
        for im in frames(f, a.every):
            r = model.predict(im, classes=[0], conf=a.conf, verbose=False)[0]
            if r.boxes is None:
                continue
            for box in r.boxes.xyxy.cpu().numpy().astype(int):
                x1, y1, x2, y2 = box
                x1, y1 = max(x1, 0), max(y1, 0)
                if y2 - y1 < a.min_height or x2 - x1 < 24:
                    continue
                crop = im[y1:y2, x1:x2]
                h = ahash(crop)
                if any(np.count_nonzero(h != o) < a.dup_distance for o in seen[-400:]):
                    continue
                seen.append(h)
                cv2.imwrite(str(out / f"{f.stem}_{saved:06d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved += 1
        print(f"{f.name}: {saved} crops so far")
    print(f"Saved {saved} crops to {out}")


if __name__ == "__main__":
    main()
