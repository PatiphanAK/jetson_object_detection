"""
to_onnx.py
==========
Convert a PyTorch .pt checkpoint to ONNX, tuned for Jetson Nano (JetPack 4.x)
+ TensorRT 8.2 + DeepStream-Yolo.

Key defaults (DO NOT change unless you know what you're doing):
  opset    = 11        ← JetPack 4.x ships TRT 8.2 which is happiest with
                         opset 11. opset >= 13 enables ops (Mod, etc.) that
                         TRT 8.2 cannot import.
  dynamic  = True      ← batch dim becomes -1 so transform.py can build a
                         batch-1..N optimization profile (5 cams = batch=5).
  nms      = False     ← do NOT bake NMS into the graph. DeepStream-Yolo
                         custom parser handles NMS in CUDA (faster). Baked
                         NMS adds TopK + Mod + Gather which TRT 8.2 rejects.
  simplify = True      ← fold constants via onnx-simplifier.

Pipeline:
    yolo26n.pt
       │  python3 to_onnx.py --pt yolo26n.pt
       ▼
    yolo26n.onnx       (opset 11, dynamic batch, no NMS)
       │  scp to Nano
       ▼
    Nano: python3 transform.py --onnx yolo26n.onnx
       ▼
    yolo26n_b5.engine  (TRT engine, batch 1..5)
       │
       ▼
    ds_count_v2.py uses it for 5-cam inference

Where to run this script
------------------------
**Best on laptop / desktop** with PyTorch installed. The Jetson Nano normally
does NOT have torch — installing it requires the NVIDIA-built wheel and
~3 GB of deps. Export on laptop, copy ONNX to Nano.

Install requirements (on laptop):
    pip install "ultralytics>=8.0.0"          # pulls torch as a dep
    # OR for a minimal install if you already have torch:
    pip install "ultralytics>=8.0.0" --no-deps && pip install onnx onnxsim

Usage:
    python3 to_onnx.py                                # defaults: yolov8n.pt
    python3 to_onnx.py --pt yolo26n.pt
    python3 to_onnx.py --pt yolo26n.pt --imgsz 320    # smaller = faster
    python3 to_onnx.py --pt my.pt --opset 12 --no-dynamic
"""
import argparse
import os
import sys
import time
from pathlib import Path


def banner(s):
    bar = "─" * 60
    print(f"\n{bar}\n  {s}\n{bar}")


def main():
    p = argparse.ArgumentParser(
        description="Convert YOLO .pt to ONNX (Jetson Nano / TRT 8.2 friendly)")
    p.add_argument("--pt",        default="yolov8n.pt",
                   help="Input PyTorch checkpoint (default: yolov8n.pt)")
    p.add_argument("--onnx",      default=None,
                   help="Output ONNX path (default: <stem>.onnx)")
    p.add_argument("--imgsz",     type=int, default=640,
                   help="Square input size (default: 640; try 320 for speed)")
    p.add_argument("--opset",     type=int, default=11,
                   help="ONNX opset (default: 11 for JetPack 4.x / TRT 8.2)")
    p.add_argument("--no-dynamic", action="store_true",
                   help="Disable dynamic batch (engine will be fixed batch=1)")
    p.add_argument("--with-nms",   action="store_true",
                   help="Bake NMS into ONNX (BREAKS on TRT 8.2 — adds Mod op)")
    p.add_argument("--no-simplify", action="store_true",
                   help="Skip onnxsim folding")
    p.add_argument("--half",       action="store_true",
                   help="Export FP16 ONNX (smaller file; TRT can still FP16-build)")
    p.add_argument("--device",     default="cpu",
                   help="Torch device: 'cpu' or '0' for GPU 0 (default: cpu)")
    args = p.parse_args()

    pt_path = Path(args.pt)
    if not pt_path.exists():
        print(f"ERROR: {pt_path} not found", file=sys.stderr)
        sys.exit(1)

    # ── Sanity warnings ────────────────────────────────────────
    if args.opset >= 13 and not args.with_nms:
        print(f"⚠ WARN  opset={args.opset} may pull in ops that TRT 8.2 cannot import.")
        print(f"        Stick with opset 11 unless you've verified TRT support.")
    if args.with_nms:
        print(f"⚠ WARN  --with-nms enabled. TRT 8.2 on JetPack 4.x cannot")
        print(f"        deserialize Mod op produced by YOLO's NMS head.")
        print(f"        DeepStream-Yolo parser handles NMS in CUDA anyway — leave it off.")

    # ── Import ultralytics (lazy so --help works without torch) ─────
    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"ERROR: ultralytics not installed.\n"
              f"       Run on a machine with PyTorch:\n"
              f"           pip install 'ultralytics>=8.0.0'\n"
              f"       (Jetson Nano normally does NOT have torch — export on laptop.)\n"
              f"       Original error: {e}",
              file=sys.stderr)
        sys.exit(2)

    banner("Export plan")
    print(f"  pt input        : {pt_path}  ({pt_path.stat().st_size/1e6:.1f} MB)")
    print(f"  opset           : {args.opset}")
    print(f"  imgsz           : {args.imgsz}")
    print(f"  dynamic batch   : {not args.no_dynamic}")
    print(f"  NMS in graph    : {args.with_nms}  (recommend: False)")
    print(f"  simplify (sim)  : {not args.no_simplify}")
    print(f"  half (FP16)     : {args.half}")
    print(f"  device          : {args.device}")

    banner("Loading model")
    t0 = time.monotonic()
    model = YOLO(str(pt_path))
    print(f"  model class     : {model.__class__.__name__}")
    print(f"  task            : {getattr(model, 'task', '?')}")
    print(f"  load time       : {time.monotonic()-t0:.1f}s")

    banner(f"Exporting to ONNX (opset {args.opset})")
    t1 = time.monotonic()
    # Ultralytics export writes to <stem>.onnx in the same dir as the .pt by default.
    exported = model.export(
        format="onnx",
        opset=args.opset,
        imgsz=args.imgsz,
        dynamic=not args.no_dynamic,
        nms=args.with_nms,
        simplify=not args.no_simplify,
        half=args.half,
        device=args.device,
    )
    dt = time.monotonic() - t1
    print(f"\n  export time     : {dt:.1f}s")
    print(f"  ultralytics returned: {exported}")

    # Resolve actual output path
    default_out = pt_path.with_suffix(".onnx")
    if args.onnx:
        out_path = Path(args.onnx)
        if default_out.exists() and default_out != out_path:
            os.replace(default_out, out_path)
    else:
        out_path = default_out

    if not out_path.exists():
        # ultralytics may have written somewhere else; trust its return value
        if isinstance(exported, (str, Path)) and Path(exported).exists():
            out_path = Path(exported)
        else:
            print(f"ERROR: cannot locate exported ONNX file", file=sys.stderr)
            sys.exit(3)

    banner("Result")
    print(f"  ONNX output     : {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    print(f"  Next step       : scp this file to Nano, then:")
    print(f"                      python3 transform.py --onnx {out_path.name}")
    print()


if __name__ == "__main__":
    main()
