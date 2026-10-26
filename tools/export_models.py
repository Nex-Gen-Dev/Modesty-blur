"""Export the two models the Android app needs to ONNX (runs in CI).

    python tools/export_models.py android/app/src/main/assets

Produces:
  yolo11n-seg.onnx   person segmentation (input 1x3x640x640)
  clip_visual.onnx   CLIP ViT-B/32 image encoder, int8-quantized (input 1x3x224x224)
  clip_text.json     normalized text embeddings for the 4 prompts in modesty_blur.classifier
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modesty_blur.classifier import PROMPTS  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)


def export_yolo():
    """Four input sizes (height, width) for the speed/quality presets."""
    from ultralytics import YOLO

    # sizes are [height, width] in 16:9 portrait proportions (the app shows frames upright)
    for name, size in [("fast", [320, 192]), ("balanced", [384, 224]),
                       ("quality", [512, 288]), ("ultra", [640, 352])]:
        p = YOLO("yolo11n-seg.pt").export(format="onnx", imgsz=size, opset=12,
                                          dynamic=False, simplify=False)
        shutil.copy(p, out / f"yolo_{name}.onnx")
        print("yolo", name, size, "->", out / f"yolo_{name}.onnx")


def export_face_models():
    """SCRFD face detector + MobileFaceNet embedder from the InsightFace 'buffalo_s' pack."""
    import time
    import urllib.request
    import zipfile

    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
    tmp = out / "_buffalo_s.zip"
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(url, tmp)
            break
        except Exception as e:  # network hiccup in CI
            print("download failed, retrying:", e)
            time.sleep(5 * (attempt + 1))
    else:
        raise SystemExit("could not download buffalo_s.zip")
    with zipfile.ZipFile(tmp) as z:
        for name in ("det_500m.onnx", "w600k_mbf.onnx", "genderage.onnx"):
            if name not in z.namelist():
                print("warning: missing", name, "in buffalo_s.zip")
                continue
            with z.open(name) as src, open(out / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    tmp.unlink()
    print("face models ->", out)


def export_clip():
    import numpy as np
    import onnxruntime as ort
    import open_clip
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval()

    class Visual(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.encode_image(x)

    dummy = torch.randn(1, 3, 224, 224)
    fp32 = out / "clip_visual_fp32.onnx"
    kw = dict(input_names=["image"], output_names=["embed"], opset_version=17)
    try:
        torch.onnx.export(Visual(model), dummy, str(fp32), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(Visual(model), dummy, str(fp32), **kw)

    quant = out / "clip_visual.onnx"
    quantize_dynamic(str(fp32), str(quant), weight_type=QuantType.QUInt8)

    # sanity check: quantized output should stay close to the original
    with torch.no_grad():
        ref = model.encode_image(dummy).numpy()[0]
    got = ort.InferenceSession(str(quant)).run(None, {"image": dummy.numpy()})[0][0]
    cos = float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got)))
    print(f"clip int8 vs fp32 cosine similarity: {cos:.4f}")
    fp32.unlink()
    for extra in out.glob("clip_visual_fp32*"):
        extra.unlink()

    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        t = model.encode_text(tok(PROMPTS))
        t = (t / t.norm(dim=-1, keepdim=True)).numpy()
    (out / "clip_text.json").write_text(json.dumps({"prompts": PROMPTS, "text": t.tolist()}))
    print("clip ->", quant)


if __name__ == "__main__":
    export_yolo()
    export_face_models()
    export_clip()
