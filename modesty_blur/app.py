"""Command line entry point.

  python -m modesty_blur run --source 0                 # live webcam
  python -m modesty_blur run --source street.mp4 --save out.mp4
  python -m modesty_blur enroll family/ --out family_gallery.npz
"""

import argparse

import cv2

from .config import Config


def cmd_run(args):
    from .pipeline import Pipeline

    cfg = Config(source=args.source, save_path=args.save, seg_model=args.model,
                 device=args.device, debug=args.debug, imgsz=args.imgsz,
                 family_gallery=args.gallery, blur_threshold=args.blur_threshold)
    pipe = Pipeline(cfg)

    src = int(cfg.source) if cfg.source.isdigit() else cfg.source
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {cfg.source}")

    writer = None
    print("Running. Keys: q = quit, d = toggle debug overlay")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = pipe.process(frame)
        if cfg.save_path:
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(cfg.save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         cap.get(cv2.CAP_PROP_FPS) or 30, (w, h))
            writer.write(out)
        if not args.headless:
            cv2.imshow("modesty-blur (blurred view)", out)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("d"):
                cfg.debug = not cfg.debug
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def cmd_enroll(args):
    from .family import enroll
    n = enroll(args.folder, args.out, device=args.device)
    print(f"Saved {n} embeddings to {args.out}. You can now delete the source photos.")


def cmd_selftest(args):
    """Load every model and push blank frames through the pipeline (used by CI on each OS)."""
    import numpy as np
    from .pipeline import Pipeline

    pipe = Pipeline(Config(device="cpu", family_gallery="__none__.npz"))
    frame = np.zeros((480, 640, 3), np.uint8)
    for _ in range(2):
        out = pipe.process(frame.copy())
        assert out.shape == frame.shape
    print("SELFTEST OK")


def main():
    p = argparse.ArgumentParser(prog="modesty_blur")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="load models and run blank frames (smoke test)")
    s.set_defaults(fn=cmd_selftest)

    r = sub.add_parser("run", help="run the live blur pipeline")
    r.add_argument("--source", default="0", help="webcam index or video file")
    r.add_argument("--save", default="", help="save blurred output to this mp4")
    r.add_argument("--model", default="yolo11n-seg.pt")
    r.add_argument("--device", default="auto")
    r.add_argument("--imgsz", type=int, default=640)
    r.add_argument("--gallery", default="family_gallery.npz")
    r.add_argument("--blur-threshold", type=float, default=0.30)
    r.add_argument("--debug", action="store_true", help="show boxes and probabilities")
    r.add_argument("--headless", action="store_true", help="no window (use with --save)")
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("enroll", help="build the family gallery from photos")
    e.add_argument("folder", help="folder containing one subfolder per person")
    e.add_argument("--out", default="family_gallery.npz")
    e.add_argument("--device", default="cpu")
    e.set_defaults(fn=cmd_enroll)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
