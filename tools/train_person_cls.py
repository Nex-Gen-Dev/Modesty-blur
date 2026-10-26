"""Step 3: train the small, fast person classifier on your labeled crops and export it for the app.

    python tools/train_person_cls.py dataset --out models --epochs 15

Dataset layout (made by tools/label.py):   dataset/woman  dataset/man  dataset/girl  dataset/boy
Female classes (blurred): woman, girl.  Anything in dataset/junk is ignored.

What you get in --out:
    person_cls.onnx   MobileNetV3-Small, input 1x3x192x96 (RGB 0..1), output = class probabilities
    person_cls.json   class names, which are "female", and the input size

Commit both files in the models/ folder and push: CI bundles them into the next APK and the app then uses
this model (a few milliseconds per person) instead of the slow CLIP model.

The number that matters is printed at the end: the MISS RATE = share of women/girls the model would let
through unblurred, at several blur thresholds. Choose the threshold in the app so it is as close to 0 as
you can get. Train with --female-weight 3 (or higher) to push the model toward blurring when unsure.

Self-test with fake data (used by CI):  python tools/train_person_cls.py --smoke
"""

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

W, H = 96, 192                       # model input (width, height): tall, like a standing person
MEAN_RGB = (124, 116, 104)           # padding color (same in the Android app)
CLASSES = ["woman", "man", "girl", "boy"]
FEMALE = [0, 2]
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pad_to_aspect(im: Image.Image) -> Image.Image:
    """Pad (do not stretch) to W:H with the mean color, then resize. Mirrors PersonClassifier.kt."""
    w, h = im.size
    aspect = W / H
    if w / h > aspect:
        cw, ch = w, round(w / aspect)
    else:
        ch, cw = h, round(h * aspect)
    canvas = Image.new("RGB", (cw, ch), MEAN_RGB)
    canvas.paste(im, ((cw - w) // 2, (ch - h) // 2))
    return canvas.resize((W, H), Image.BILINEAR)


class RandomDownUp:
    """Simulates a far-away, low-resolution person."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im):
        if random.random() > self.p:
            return im
        f = random.uniform(0.25, 0.7)
        small = im.resize((max(8, int(im.width * f)), max(16, int(im.height * f))), Image.BILINEAR)
        return small.resize(im.size, Image.BILINEAR)


def build_transforms():
    train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=5, translate=(0.08, 0.05), scale=(0.85, 1.1), fill=MEAN_RGB),
        transforms.ColorJitter(0.35, 0.35, 0.35, 0.03),
        RandomDownUp(0.5),
        transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.5))], p=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),   # partial occlusion
    ])
    plain = transforms.ToTensor()
    return train, plain


class Crops(torch.utils.data.Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        im = pad_to_aspect(Image.open(path).convert("RGB"))
        return self.tf(im), y


def collect(root: Path):
    items = []
    for ci, c in enumerate(CLASSES):
        d = root / c
        if d.exists():
            items += [(p, ci) for p in sorted(d.iterdir()) if p.suffix.lower() in IMG]
    return items


def split(items, val_frac=0.15, seed=0):
    rng = random.Random(seed)
    tr, va = [], []
    for ci in range(len(CLASSES)):
        cls = [it for it in items if it[1] == ci]
        rng.shuffle(cls)
        n = max(1, int(len(cls) * val_frac)) if len(cls) >= 5 else 0
        va += cls[:n]
        tr += cls[n:]
    return tr, va


class Exportable(nn.Module):
    """Raw RGB 0..1 in, class probabilities out (normalization lives inside the model)."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return F.softmax(self.net((x - self.mean) / self.std), dim=1)


def make_net(pretrained: bool):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    net = models.mobilenet_v3_small(weights=weights)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(CLASSES))
    return net


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    probs, ys = [], []
    for x, y in loader:
        p = F.softmax(net(((x.to(device) - MEAN) / STD)), dim=1).cpu()
        probs.append(p)
        ys.append(y)
    return torch.cat(probs), torch.cat(ys)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def report(probs, ys):
    pred = probs.argmax(1)
    print("\nConfusion matrix (rows = truth, columns = prediction):")
    print("          " + "".join(f"{c:>8}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        row = [(pred[ys == i] == j).sum().item() for j in range(len(CLASSES))]
        print(f"{c:>10}" + "".join(f"{v:8d}" for v in row))
    pf = probs[:, FEMALE].sum(1)
    is_f = torch.tensor([int(y) in FEMALE for y in ys])
    print("\nBlur threshold on P(female) | MISS rate (female not blurred) | false-blur rate (male blurred)")
    for thr in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        miss = (pf[is_f] < thr).float().mean().item() if is_f.any() else float("nan")
        fb = (pf[~is_f] >= thr).float().mean().item() if (~is_f).any() else float("nan")
        print(f"        {thr:4.2f}              |            {miss:7.1%}              |        {fb:7.1%}")


def export(net, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    net = net.cpu().eval()
    wrapper = Exportable(net).eval()
    dummy = torch.rand(1, 3, H, W)
    path = out / "person_cls.onnx"
    kw = dict(input_names=["image"], output_names=["probs"], opset_version=17)
    try:
        torch.onnx.export(wrapper, dummy, str(path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(wrapper, dummy, str(path), **kw)
    (out / "person_cls.json").write_text(json.dumps(
        {"classes": CLASSES, "female": FEMALE, "width": W, "height": H}))
    # sanity check against onnxruntime
    import onnxruntime as ort
    got = ort.InferenceSession(str(path)).run(None, {"image": dummy.numpy()})[0]
    ref = wrapper(dummy).detach().numpy()
    assert got.shape == (1, len(CLASSES)), got.shape
    assert np.allclose(got, ref, atol=1e-3), "ONNX output differs from PyTorch"
    print(f"Exported {path} and person_cls.json  ({path.stat().st_size / 1e6:.1f} MB)")


def make_smoke_dataset(root: Path):
    rng = np.random.default_rng(0)
    for ci, c in enumerate(CLASSES):
        (root / c).mkdir(parents=True, exist_ok=True)
        for i in range(14):
            h, w = int(rng.integers(80, 220)), int(rng.integers(40, 120))
            base = np.zeros((h, w, 3), np.uint8)
            base[:] = (ci * 60, 255 - ci * 60, 128)
            noise = rng.integers(0, 40, (h, w, 3), dtype=np.uint8)
            Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8)).save(root / c / f"{i}.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="dataset")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--female-weight", type=float, default=2.0,
                    help="how much more a missed woman/girl costs than a falsely blurred man/boy")
    ap.add_argument("--smoke", action="store_true", help="tiny fake-data run to check the code works")
    a = ap.parse_args()

    tmp = None
    if a.smoke:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        make_smoke_dataset(root)
        a.epochs, a.batch = 1, 8
        pretrained = False
        out = Path(tmp.name) / "out"
    else:
        root, out = Path(a.dataset), Path(a.out)
        pretrained = True

    items = collect(root)
    if len(items) < 8:
        sys.exit(f"Found only {len(items)} labeled crops in {root}. Label more with tools/label.py first.")
    counts = {c: sum(1 for _, y in items if y == i) for i, c in enumerate(CLASSES)}
    print("labeled crops:", counts)
    tr_items, va_items = split(items)
    train_tf, plain_tf = build_transforms()
    tr = torch.utils.data.DataLoader(Crops(tr_items, train_tf), batch_size=a.batch, shuffle=True, num_workers=0)
    va = torch.utils.data.DataLoader(Crops(va_items, plain_tf), batch_size=a.batch) if va_items else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = make_net(pretrained).to(device)
    w = torch.ones(len(CLASSES))
    for i in FEMALE:
        w[i] = a.female_weight
    loss_fn = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.05)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(1, a.epochs * len(tr)))

    best, best_state = None, None
    for ep in range(a.epochs):
        net.train()
        total = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(net((x - MEAN.to(device)) / STD.to(device)), y)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y)
        msg = f"epoch {ep + 1}/{a.epochs}  train loss {total / len(tr_items):.3f}"
        if va is not None:
            probs, ys = evaluate(net, va, device)
            pf = probs[:, FEMALE].sum(1)
            is_f = torch.tensor([int(t) in FEMALE for t in ys])
            miss = (pf[is_f] < 0.3).float().mean().item() if is_f.any() else 0.0
            acc = (probs.argmax(1) == ys).float().mean().item()
            msg += f"  val acc {acc:.1%}  miss@0.30 {miss:.1%}"
            score = (miss, -acc)
            if best is None or score < best:
                best, best_state = score, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(msg)

    if best_state is not None:
        net.load_state_dict(best_state)
    if va is not None:
        probs, ys = evaluate(net, va, device)
        report(probs, ys)
    export(net, out)
    if tmp is not None:
        print("SMOKE OK")
        tmp.cleanup()


if __name__ == "__main__":
    main()
