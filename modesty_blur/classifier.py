"""Zero-shot person classifier built on CLIP.

Returns P(female) for each person crop. No training required to get started;
fine-tune later (see README) once you have labeled footage.
"""

import numpy as np

# Index 0-1 count as "female" for blurring purposes; 2-3 as "male".
PROMPTS = [
    "a photo of a woman",
    "a photo of a girl",
    "a photo of a man",
    "a photo of a boy",
]
FEMALE_IDX = [0, 1]


def resolve_device(pref: str) -> str:
    import torch
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pad_square(bgr):
    """Pad a (usually tall) person crop to a square so CLIP sees head and feet, not a center crop."""
    from PIL import Image
    h, w = bgr.shape[:2]
    s = max(h, w)
    canvas = np.empty((s, s, 3), np.uint8)
    canvas[:] = (104, 116, 124)  # BGR of the CLIP mean color
    y, x = (s - h) // 2, (s - w) // 2
    canvas[y:y + h, x:x + w] = bgr
    return Image.fromarray(canvas[..., ::-1])


class ClipPersonClassifier:
    def __init__(self, model_name: str, pretrained: str, device: str = "auto"):
        import torch
        import open_clip
        from PIL import Image  # noqa: F401  (ensures Pillow is present)

        self.torch = torch
        self.device = resolve_device(device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device)
        self.model = model.eval()
        self.preprocess = preprocess
        tok = open_clip.get_tokenizer(model_name)
        with torch.no_grad():
            t = self.model.encode_text(tok(PROMPTS).to(self.device))
            self.text = t / t.norm(dim=-1, keepdim=True)

    def predict_female(self, crops_bgr) -> np.ndarray:
        """crops_bgr: list of HxWx3 uint8 BGR arrays -> array of P(female)."""
        from PIL import Image
        if not crops_bgr:
            return np.zeros(0, np.float32)
        torch = self.torch
        batch = torch.stack([
            self.preprocess(_pad_square(c)) for c in crops_bgr
        ]).to(self.device)
        with torch.no_grad():
            f = self.model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
            probs = (100.0 * f @ self.text.T).softmax(dim=-1).cpu().numpy()
        return probs[:, FEMALE_IDX].sum(axis=1).astype(np.float32)
