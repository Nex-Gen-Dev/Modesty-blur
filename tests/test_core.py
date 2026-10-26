"""Core logic tests (no ML models needed). Run:  python -m pytest -q   or   python tests/test_core.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modesty_blur.blur import blur_masked, polygon_to_mask  # noqa: E402
from modesty_blur.config import Config  # noqa: E402
from modesty_blur.decisions import DecisionEngine  # noqa: E402


def test_new_track_is_blurred_before_classification():
    e = DecisionEngine(Config())
    assert e.should_blur(1) is True


def test_female_stays_blurred():
    e = DecisionEngine(Config())
    for f in range(10):
        e.update_classification(1, 0.9, f)
    assert e.should_blur(1)


def test_man_cleared_only_after_enough_confident_observations():
    c = Config()
    e = DecisionEngine(c)
    for f in range(c.min_obs_to_clear - 1):
        e.update_classification(2, 0.01, f)
        assert e.should_blur(2), "must not clear too early"
    e.update_classification(2, 0.01, 99)
    assert not e.should_blur(2)


def test_uncertain_never_cleared():
    c = Config()
    e = DecisionEngine(c)
    for f in range(30):
        e.update_classification(3, 0.2, f)          # between clear (0.12) and blur (0.30)
    assert e.should_blur(3)


def test_cleared_track_rearms_on_doubt():
    c = Config()
    e = DecisionEngine(c)
    for f in range(8):
        e.update_classification(4, 0.0, f)
    assert not e.should_blur(4)
    for f in range(8, 14):
        e.update_classification(4, 0.95, f)
    assert e.should_blur(4)


def test_family_needs_consecutive_matches():
    c = Config()
    e = DecisionEngine(c)
    e.update_classification(5, 0.99, 0)
    e.update_family(5, True, "dad", 1)
    e.update_family(5, False, "", 2)                # streak broken
    e.update_family(5, True, "dad", 3)
    e.update_family(5, True, "dad", 4)
    assert e.should_blur(5)
    e.update_family(5, True, "dad", 5)
    assert not e.should_blur(5)


def test_blur_hides_detail_inside_mask_and_leaves_outside_untouched():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)   # noisy = max detail
    orig = frame.copy()
    poly = np.array([[100, 40], [200, 40], [220, 230], [80, 230]], np.float32)
    mask = polygon_to_mask(poly, frame.shape[:2], dilate_px=0)
    blur_masked(frame, mask, strength=0.14)
    inside = mask > 0
    # detail (local variation) inside the silhouette must collapse
    var_before = orig[inside].astype(float).std()
    var_after = frame[inside].astype(float).std()
    assert var_after < 0.25 * var_before
    # pixels well outside the silhouette are unchanged
    far = np.zeros_like(inside)
    far[:, :40] = True
    assert np.array_equal(frame[far], orig[far])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
