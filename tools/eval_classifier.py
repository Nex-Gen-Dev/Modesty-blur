"""Measure how often the classifier would let a woman through (the dangerous error).

Folder layout (crops of single people, any size):
    eval_set/woman/*.jpg
    eval_set/man/*.jpg

Usage:  python tools/eval_classifier.py eval_set --threshold 0.30
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modesty_blur.classifier import ClipPersonClassifier  # noqa: E402
from modesty_blur.config import Config  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load(folder):
    imgs = []
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() in EXT:
            im = cv2.imread(str(p))
            if im is not None:
                imgs.append(im)
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--threshold", type=float, default=Config.blur_threshold)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    clf = ClipPersonClassifier(Config.clip_model, Config.clip_pretrained, a.device)
    women = load(Path(a.root) / "woman")
    men = load(Path(a.root) / "man")
    pw = np.concatenate([clf.predict_female(women[i:i + 32]) for i in range(0, len(women), 32)]) if women else np.zeros(0)
    pm = np.concatenate([clf.predict_female(men[i:i + 32]) for i in range(0, len(men), 32)]) if men else np.zeros(0)

    print(f"threshold = {a.threshold}")
    if len(pw):
        miss = float((pw < a.threshold).mean())
        print(f"women: {len(pw)} images | MISS rate (woman not blurred): {miss:.1%}   <- keep this near 0")
    if len(pm):
        fb = float((pm >= a.threshold).mean())
        print(f"men:   {len(pm)} images | false-blur rate (man blurred):   {fb:.1%}")
    print("Sweep threshold lower to cut misses, at the cost of more false blurs.")


if __name__ == "__main__":
    main()
