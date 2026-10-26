"""Frame pipeline: segment -> track -> classify -> (family check) -> blur."""

import os
import sys
import time

import cv2
import numpy as np

from .blur import blur_masked, polygon_to_mask
from .classifier import ClipPersonClassifier, resolve_device
from .decisions import DecisionEngine


def resolve_asset(name: str) -> str:
    """Use the model bundled inside a PyInstaller build if present, else the plain name."""
    base = getattr(sys, "_MEIPASS", None)
    if base and os.path.exists(os.path.join(base, name)):
        return os.path.join(base, name)
    return name


class Pipeline:
    def __init__(self, cfg):
        from ultralytics import YOLO
        self.cfg = cfg
        dev = resolve_device(cfg.device)
        self.yolo_device = 0 if dev == "cuda" else dev
        print(f"[init] device={dev}")
        self.seg = YOLO(resolve_asset(cfg.seg_model))
        self.clf = ClipPersonClassifier(cfg.clip_model, cfg.clip_pretrained, dev)
        self.engine = DecisionEngine(cfg)
        self.family = None
        if os.path.exists(cfg.family_gallery):
            try:
                from .family import FamilyMatcher
                self.family = FamilyMatcher(cfg.family_gallery, cfg.family_threshold, dev)
                print(f"[init] family whitelist loaded ({len(self.family.names)} embeddings)")
            except Exception as e:  # insightface missing, etc.
                print(f"[init] family whitelist disabled: {e}")
        else:
            print("[init] no family gallery found -> everyone classified female is blurred")
        self.ghosts = {}          # tid -> (polygon, last_seen_frame) for blurred tracks
        self.frame_idx = 0
        self.fps = 0.0
        self._t = time.time()

    # ------------------------------------------------------------------
    def _detect(self, frame):
        c = self.cfg
        r = self.seg.track(frame, persist=True, tracker=c.tracker, classes=[0],
                           conf=c.person_conf, imgsz=c.imgsz, device=self.yolo_device,
                           verbose=False)[0]
        people = []
        if r.boxes is None or len(r.boxes) == 0:
            return people
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = (r.boxes.id.int().cpu().tolist() if r.boxes.id is not None
               else [-(i + 1) for i in range(len(boxes))])     # untracked -> temp negative ids
        polys = r.masks.xy if r.masks is not None else [None] * len(boxes)
        for tid, box, poly in zip(ids, boxes, polys):
            if poly is None or len(poly) < 3:                  # no mask -> fall back to the box
                x1, y1, x2, y2 = box
                poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
            people.append((int(tid), box.astype(int), poly))
        return people

    @staticmethod
    def _crop(frame, box, pad=0.05):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
        x1, y1 = max(x1 - px, 0), max(y1 - py, 0)
        x2, y2 = min(x2 + px, w), min(y2 + py, h)
        return frame[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    def process(self, frame):
        """Takes a BGR frame, returns the blurred BGR frame."""
        c, eng, f = self.cfg, self.engine, self.frame_idx
        original = frame.copy()                  # classify on clean pixels, not blurred ones
        out = frame
        people = self._detect(original)

        # 1) classification (batched)
        todo = [(tid, box) for tid, box, _ in people if eng.needs_classification(tid, f)]
        crops = [self._crop(original, box) for _, box in todo]
        valid = [(t, cr) for (t, _), cr in zip(todo, crops) if cr.size > 0]
        if valid:
            probs = self.clf.predict_female([cr for _, cr in valid])
            for (tid, _), p in zip(valid, probs):
                eng.update_classification(tid, float(p), f)

        # 2) family whitelist check, only for tracks that would otherwise be blurred
        if self.family is not None:
            for tid, box, _ in people:
                if eng.should_blur(tid) and eng.needs_family_check(tid, f):
                    crop = self._crop(original, box)
                    if crop.size == 0:
                        continue
                    matched, name, _sim = self.family.check(crop)
                    eng.update_family(tid, matched, name, f)

        # 3) blur
        seen = set()
        for tid, box, poly in people:
            seen.add(tid)
            if eng.should_blur(tid):
                self.ghosts[tid] = (poly, f)
                blur_masked(out, polygon_to_mask(poly, out.shape[:2], c.mask_dilate_px),
                            c.blur_strength)
            else:
                self.ghosts.pop(tid, None)
            if c.debug:
                s = eng.state(tid)
                tag = "FAMILY" if s.is_family else ("blur" if eng.should_blur(tid) else "ok")
                x1, y1, x2, y2 = box
                cv2.putText(out, f"#{tid} {tag} pF={s.ema_female:.2f}", (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 4) keep blurring recently-lost blurred tracks (detector dropouts must not leak)
        for tid in list(self.ghosts):
            poly, last = self.ghosts[tid]
            if tid in seen:
                continue
            if f - last <= c.hold_frames:
                blur_masked(out, polygon_to_mask(poly, out.shape[:2], c.mask_dilate_px + 6),
                            c.blur_strength)
            else:
                del self.ghosts[tid]

        eng.prune(seen, keep=self.ghosts.keys())
        self.frame_idx += 1
        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self._t, 1e-6) if self.fps else 0.0
        self._t = now
        if c.debug:
            cv2.putText(out, f"{self.fps:.1f} fps", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        return out
