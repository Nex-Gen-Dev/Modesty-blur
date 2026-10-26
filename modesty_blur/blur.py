"""Full-silhouette blur.

The blur is computed with "normalized convolution": colors are averaged only from
pixels *inside* the silhouette, so the background never bleeds into the person and
the person never bleeds out. The result is a smooth, unrecognizable color blob in the
exact shape of the body (not a box).
"""

import cv2
import numpy as np


def polygon_to_mask(polygon, shape_hw, dilate_px: int = 0) -> np.ndarray:
    """Rasterize a polygon (Nx2 float pixel coords) into a uint8 mask (0/255)."""
    h, w = shape_hw
    mask = np.zeros((h, w), np.uint8)
    if polygon is None or len(polygon) < 3:
        return mask
    pts = np.round(np.asarray(polygon)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, k)
    return mask


def blur_masked(frame: np.ndarray, mask: np.ndarray, strength: float = 0.14,
                feather_px: float = 3.0) -> np.ndarray:
    """Blur `frame` inside `mask` (in place) and return it.

    strength: blur sigma as a fraction of the silhouette's smaller side.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return frame
    pad = 6
    H, W = mask.shape
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + 1 + pad, W)
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + 1 + pad, H)
    roi = frame[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1]
    rh, rw = m.shape

    # Work at reduced resolution: faster, and a stronger blur for the same sigma.
    scale = 0.25 if min(rh, rw) > 80 else 1.0
    sw, sh = max(int(rw * scale), 2), max(int(rh * scale), 2)
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    ms = (cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)

    sigma = max(3.0, strength * min(sw, sh))
    num = cv2.GaussianBlur(small * ms[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(ms, (0, 0), sigma)[..., None]
    blob = num / np.maximum(den, 1e-3)
    blob = cv2.resize(blob, (rw, rh), interpolation=cv2.INTER_CUBIC)

    alpha = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), feather_px)[..., None]
    out = roi.astype(np.float32) * (1.0 - alpha) + blob * alpha
    frame[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
    return frame
