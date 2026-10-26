from dataclasses import dataclass


@dataclass
class Config:
    # ---- input / output -------------------------------------------------
    source: str = "0"                 # webcam index ("0") or a video file path
    save_path: str = ""               # write the blurred output to this file if set
    width: int = 1280                 # requested camera width (ignored for files)
    height: int = 720

    # ---- detection / segmentation / tracking ----------------------------
    seg_model: str = "yolo11n-seg.pt"  # n=fastest; try yolo11s-seg.pt / yolo11m-seg.pt
    imgsz: int = 640
    person_conf: float = 0.30          # low on purpose: a missed person is worse than a false one
    device: str = "auto"               # "auto" | "cpu" | "mps" | "0" (first CUDA GPU)
    tracker: str = "bytetrack.yaml"

    # ---- person classification (CLIP zero-shot) -------------------------
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    classify_every: int = 6            # re-classify each track every N frames
    min_obs_to_clear: int = 4          # observations needed before a track may be un-blurred
    blur_threshold: float = 0.30       # blur when smoothed P(female) >= this  (low = safer)
    clear_threshold: float = 0.12      # un-blur as "not female" only when P(female) <= this
    ema_alpha: float = 0.5             # smoothing of the per-track probability

    # ---- family whitelist (face recognition) ----------------------------
    family_gallery: str = "family_gallery.npz"   # built by the `enroll` command
    family_threshold: float = 0.50     # cosine similarity needed to count as a match
    family_hits_required: int = 3      # consecutive matches before un-blurring (anti-false-match)
    family_check_every: int = 4        # frames between face checks for a blurred track

    # ---- blur look / safety ---------------------------------------------
    mask_dilate_px: int = 9            # grow the silhouette so edges never leak
    blur_strength: float = 0.14        # blur sigma as a fraction of the person's size
    hold_frames: int = 10              # keep blurring a person this many frames after tracking loss

    # ---- misc ------------------------------------------------------------
    debug: bool = False                # draw boxes + probabilities (for testing only)
