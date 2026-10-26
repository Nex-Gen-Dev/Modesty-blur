"""Step 2: sort the crops into classes with single key presses.

    python tools/label.py dataset/unlabeled --out dataset

Shows one crop at a time. Keys:
    w = woman    m = man    g = girl    b = boy
    x = junk / not a person / cannot tell     s = skip for now
    Backspace = undo the last label           q = quit (progress is kept)

Files are MOVED into dataset/<class>/ so you can stop and continue any time.
Only label what you can see: if you honestly cannot tell, press x.
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

KEYS = {ord("w"): "woman", ord("m"): "man", ord("g"): "girl", ord("b"): "boy", ord("x"): "junk"}
IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def show(img, counts, name, remaining):
    h, w = img.shape[:2]
    scale = 640 / max(h, w)
    view = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    pad = np.full((view.shape[0] + 70, max(view.shape[1], 640), 3), 30, np.uint8)
    pad[:view.shape[0], :view.shape[1]] = view
    info = "  ".join(f"{k}:{v}" for k, v in counts.items())
    cv2.putText(pad, "w woman  m man  g girl  b boy  x junk  s skip  <- undo  q quit",
                (8, view.shape[0] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(pad, f"{info}   left: {remaining}", (8, view.shape[0] + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 220, 120), 1)
    cv2.imshow("label", pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="folder of unlabeled crops")
    ap.add_argument("--out", default="dataset")
    a = ap.parse_args()

    out = Path(a.out)
    files = sorted(p for p in Path(a.source).iterdir() if p.suffix.lower() in IMAGE)
    if not files:
        sys.exit("Nothing to label.")
    counts = {c: len(list((out / c).glob("*"))) if (out / c).exists() else 0 for c in KEYS.values()}
    history = []
    i = 0
    while i < len(files):
        f = files[i]
        img = cv2.imread(str(f))
        if img is None:
            i += 1
            continue
        show(img, counts, f.name, len(files) - i)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            i += 1
        elif k in KEYS:
            cls = KEYS[k]
            (out / cls).mkdir(parents=True, exist_ok=True)
            dst = out / cls / f.name
            shutil.move(str(f), str(dst))
            counts[cls] += 1
            history.append((f, dst, cls))
            i += 1
        elif k in (8, 127) and history:                      # backspace / delete = undo
            src, dst, cls = history.pop()
            shutil.move(str(dst), str(src))
            counts[cls] -= 1
            i = files.index(src)
    cv2.destroyAllWindows()
    print("Labeled so far:", counts)


if __name__ == "__main__":
    main()
