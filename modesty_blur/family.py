"""Family whitelist via face recognition (InsightFace / ArcFace).

Privacy design:
  * `enroll` turns your family photos into face *embeddings* (numeric vectors) saved
    in one local .npz file. You can delete the photos afterwards.
  * Only people who gave consent should be enrolled. Nothing leaves your machine.
  * Matching is conservative: several consecutive matches are required to un-blur.
"""

from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _make_app(device: str):
    from insightface.app import FaceAnalysis
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device not in ("cpu", "mps") else ["CPUExecutionProvider"])
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0 if providers[0].startswith("CUDA") else -1, det_size=(320, 320))
    return app


def _largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def enroll(folder: str, out_path: str, device: str = "cpu") -> int:
    """folder/<person_name>/*.jpg  ->  out_path (.npz). Returns number of embeddings saved."""
    app = _make_app(device)
    names, embs = [], []
    for person_dir in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
        n_before = len(embs)
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXT:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            face = _largest_face(app.get(img))
            if face is None:
                print(f"  no face found: {img_path}")
                continue
            names.append(person_dir.name)
            embs.append(face.normed_embedding.astype(np.float32))
        print(f"  {person_dir.name}: {len(embs) - n_before} face(s) enrolled")
    if not embs:
        raise SystemExit("No faces enrolled. Check the folder layout: family/<name>/*.jpg")
    np.savez(out_path, names=np.array(names), embs=np.stack(embs))
    return len(embs)


class FamilyMatcher:
    def __init__(self, gallery_path: str, threshold: float, device: str = "cpu"):
        data = np.load(gallery_path, allow_pickle=False)
        self.names = list(data["names"])
        self.embs = data["embs"].astype(np.float32)       # N x 512, L2-normalized
        self.threshold = threshold
        self.app = _make_app(device)

    def check(self, person_crop_bgr):
        """Returns (matched: bool, name: str, similarity: float)."""
        face = _largest_face(self.app.get(person_crop_bgr))
        if face is None:
            return False, "", 0.0                         # no usable face -> stay blurred
        sims = self.embs @ face.normed_embedding.astype(np.float32)
        i = int(np.argmax(sims))
        return bool(sims[i] >= self.threshold), self.names[i], float(sims[i])
