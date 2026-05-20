"""
transform.py
============
One-shot converter: YOLOv8 ONNX  ->  TensorRT engine for batched inference.

Defaults to **batch=5** so it matches the 5-camera DeepStream pipeline
(`ds_count_v2.py`). Engine is built with a dynamic-batch optimization
profile [1..max_batch] so the same engine can also be used at smaller
batch sizes without rebuilding.

Run once after exporting ONNX. The output `.engine` is consumed by:
  * `ds_count_v2.py` (via config_infer_yolov8.txt -> model-engine-file)
  * `main_batch.py`   (via --model)

Usage:
    # Default: yolov8n.onnx -> yolov8n_b5.engine, FP16, batch 1..5, input 640
    python3 transform.py

    # Custom batch / input / output
    python3 transform.py --max-batch 8 --input-size 320
    python3 transform.py --onnx my.onnx --engine my_b5.engine

    # Force FP32 (slower, more accurate, larger engine)
    python3 transform.py --no-fp16

Notes:
  * Build time on Jetson Nano: ~5-10 minutes (CPU + GPU at 100%)
  * Engine file size: ~20-30 MB for yolov8n at batch=5
  * If the ONNX has hard-coded batch=1 internally (rare for ultralytics),
    re-export with:
        yolo export model=yolov8n.pt format=onnx dynamic=True opset=12
"""

import argparse
import os
import sys
import time
from pathlib import Path

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def fmt_duration(s):
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(int(s), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def banner(title):
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def main():
    p = argparse.ArgumentParser(
        description="Transform YOLOv8 ONNX -> TensorRT engine (dynamic batch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in (__doc__ or "") else "",
    )
    p.add_argument(
        "--onnx", default="yolov8n.onnx", help="Input ONNX file (default: yolov8n.onnx)"
    )
    p.add_argument(
        "--engine",
        default=None,
        help="Output engine path (default: <onnx-stem>_b<N>.engine)",
    )
    p.add_argument(
        "--max-batch",
        type=int,
        default=5,
        help="Max batch size for optimization profile (default: 5)",
    )
    p.add_argument(
        "--input-size", type=int, default=640, help="Square input HxW (default: 640)"
    )
    p.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 (use FP32 — slower, larger)",
    )
    p.add_argument(
        "--workspace-mb",
        type=int,
        default=1024,
        help="TensorRT workspace size in MB (default: 1024)",
    )
    p.add_argument(
        "--input-name",
        default="images",
        help="ONNX input tensor name (default: images)",
    )
    args = p.parse_args()

    fp16 = not args.no_fp16
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"ERROR: ONNX file not found: {onnx_path}", file=sys.stderr)
        print(f"       Export from PyTorch on a laptop with:", file=sys.stderr)
        print(
            f"         yolo export model=yolov8n.pt format=onnx dynamic=True opset=12",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.engine is None:
        engine_path = onnx_path.with_name(f"{onnx_path.stem}_b{args.max_batch}.engine")
    else:
        engine_path = Path(args.engine)

    # ─── Plan summary ────────────────────────────────────────────
    banner("Transform plan")
    print(f"  TensorRT version : {trt.__version__}")
    print(f"  ONNX input       : {onnx_path}  ({fmt_bytes(onnx_path.stat().st_size)})")
    print(f"  Engine output    : {engine_path}")
    print(f"  Input shape      : (-1, 3, {args.input_size}, {args.input_size})")
    print(f"  Batch range      : 1 .. {args.max_batch}")
    print(f"  Precision        : {'FP16' if fp16 else 'FP32'}")
    print(f"  Workspace        : {args.workspace_mb} MB")
    print(f"  Input tensor     : '{args.input_name}'")

    if engine_path.exists():
        prev_size = fmt_bytes(engine_path.stat().st_size)
        print(f"\n  WARNING: {engine_path.name} already exists ({prev_size})")
        print(f"           It will be overwritten.")

    # ─── Build network from ONNX ─────────────────────────────────
    banner("Step 1/4: Parse ONNX")
    t0 = time.monotonic()
    builder = trt.Builder(TRT_LOGGER)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("  ONNX parse errors:")
            for i in range(parser.num_errors):
                print(f"    {parser.get_error(i)}")
            sys.exit(2)
    print(f"  ✓ parsed in {fmt_duration(time.monotonic() - t0)}")
    print(
        f"  network: {network.num_layers} layers, {network.num_inputs} inputs, "
        f"{network.num_outputs} outputs"
    )

    # ─── Set dynamic input + optimization profile ────────────────
    banner("Step 2/4: Configure dynamic batch")
    inp = network.get_input(0)
    original = tuple(inp.shape)
    new_shape = (-1, 3, args.input_size, args.input_size)
    print(f"  input '{inp.name}': {original} -> {new_shape}")
    inp.shape = new_shape

    config = builder.create_builder_config()
    config.max_workspace_size = args.workspace_mb * (1 << 20)

    profile = builder.create_optimization_profile()
    min_shape = (1, 3, args.input_size, args.input_size)
    opt_shape = (args.max_batch, 3, args.input_size, args.input_size)
    max_shape = (args.max_batch, 3, args.input_size, args.input_size)
    profile.set_shape(args.input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    print(f"  optimization profile: min={min_shape}  opt={opt_shape}  max={max_shape}")

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print(f"  ✓ FP16 enabled")
        else:
            print(
                f"  WARNING: platform doesn't support fast FP16 — falling back to FP32"
            )
            fp16 = False

    # ─── Build engine ────────────────────────────────────────────
    banner("Step 3/4: Build engine (this takes 5-10 min on Jetson Nano)")
    print(f"  Started at: {time.strftime('%H:%M:%S')}")
    print(f"  Be patient — TRT is fusing kernels, choosing tactics, etc.")
    print(f"  CPU + GPU will be at 100%. Watch with: tegrastats")
    print()

    t_build0 = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    t_build = time.monotonic() - t_build0

    if serialized is None:
        print(f"\n  ✗ Engine build FAILED.")
        print(
            f"  Most likely cause: ONNX has hard-coded batch=1 in internal shape ops."
        )
        print(f"  Re-export with dynamic batch:")
        print(f"      yolo export model=yolov8n.pt format=onnx dynamic=True opset=12")
        sys.exit(3)

    print(f"  ✓ build complete in {fmt_duration(t_build)}")
    print(f"  serialized size: {fmt_bytes(serialized.nbytes)}")

    # ─── Save + verify ───────────────────────────────────────────
    banner("Step 4/4: Save + verify")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"  ✓ saved to {engine_path}")
    print(f"  on-disk size: {fmt_bytes(engine_path.stat().st_size)}")

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        print(f"  ✗ deserialize check failed (engine file may be corrupt)")
        sys.exit(4)
    print(f"  ✓ deserialize check passed")
    print(f"  bindings:")
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = tuple(engine.get_binding_shape(i))
        is_in = engine.binding_is_input(i)
        kind = "IN " if is_in else "OUT"
        print(f"    [{i}] {kind} {name:<12}: shape={shape}")

    total = time.monotonic() - t0
    banner("Done")
    print(f"  Total time: {fmt_duration(total)}")
    print(f"  Output:     {engine_path}")
    print(
        f"  Next step:  point config_infer_yolov8.txt to this engine, then run ds_count_v2.py"
    )
    print()


if __name__ == "__main__":
    main()
