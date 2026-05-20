"""
build_batch.py
==============
Rebuild a YOLOv8 TensorRT engine that supports dynamic batch [1..max_batch].

The original build.py produces an engine with FIXED batch=1 (because the
ONNX exported by Ultralytics default has batch=1 baked in). To run a
batched inference (e.g. 5 cameras into one infer call) we need a fresh
engine with an explicit optimization profile.

Usage:
    python build_batch.py                       # defaults: yolov8n.onnx -> yolov8n_b5.engine, max batch 5
    python build_batch.py --max-batch 8
    python build_batch.py --onnx my.onnx --engine my_b5.engine --max-batch 5

Important:
    The source ONNX must declare batch dim as dynamic OR fixed=1. We
    rewrite the input shape to (-1, 3, H, W) before building. If the
    model has internal shape ops that hard-code batch=1, the build will
    error out — re-export the ONNX with dynamic batch:
        yolo export model=yolov8n.pt format=onnx dynamic=True opset=12
"""
import argparse
import os
import sys
import time
import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path: str, engine_path: str,
                 max_batch: int, input_size: int,
                 fp16: bool, workspace_mb: int) -> None:
    print(f"TensorRT {trt.__version__}")
    print(f"ONNX:        {onnx_path}")
    print(f"Engine out:  {engine_path}")
    print(f"Max batch:   {max_batch}")
    print(f"Input size:  {input_size}x{input_size}")
    print(f"FP16:        {fp16}")
    print(f"Workspace:   {workspace_mb} MB")

    builder = trt.Builder(TRT_LOGGER)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("ONNX parse errors:")
            for i in range(parser.num_errors):
                print(f"  {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed.")

    # Rewrite input to dynamic batch
    inp = network.get_input(0)
    print(f"Original input '{inp.name}': {tuple(inp.shape)}")
    inp.shape = (-1, 3, input_size, input_size)
    print(f"Rewritten to: {tuple(inp.shape)}")

    config = builder.create_builder_config()
    config.max_workspace_size = workspace_mb * (1 << 20)

    # Optimization profile: min=1, opt=max, max=max
    profile = builder.create_optimization_profile()
    profile.set_shape(
        inp.name,
        (1, 3, input_size, input_size),
        (max_batch, 3, input_size, input_size),
        (max_batch, 3, input_size, input_size),
    )
    config.add_optimization_profile(profile)

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 enabled.")
        else:
            print("Platform doesn't support fast FP16 — falling back to FP32.")

    print("Building engine (this can take 5-10 minutes on Jetson Nano)...")
    t0 = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(
            "Engine build returned None. Likely cause: ONNX has hard-coded "
            "batch=1 in internal shape ops. Re-export with dynamic=True:\n"
            "    yolo export model=yolov8n.pt format=onnx dynamic=True opset=12"
        )
    dt = time.monotonic() - t0
    print(f"Engine built in {dt:.1f}s, serialized size: {len(serialized)/1e6:.1f} MB")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"Engine saved -> {engine_path}")

    # Verify
    print("\nVerifying engine bindings:")
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized)
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        is_in = engine.binding_is_input(i)
        print(f"  [{i}] {'IN ' if is_in else 'OUT'} {name}: shape={tuple(shape)}")


def main():
    p = argparse.ArgumentParser(description="Rebuild YOLOv8 TRT engine with dynamic batch")
    p.add_argument("--onnx",        default="yolov8n.onnx")
    p.add_argument("--engine",      default=None,
                   help="Output path (default: <model>_b<N>.engine)")
    p.add_argument("--max-batch",   type=int, default=5)
    p.add_argument("--input-size",  type=int, default=640)
    p.add_argument("--no-fp16",     action="store_true")
    p.add_argument("--workspace-mb", type=int, default=1024)
    args = p.parse_args()

    if not os.path.exists(args.onnx):
        print(f"ERROR: ONNX not found: {args.onnx}", file=sys.stderr)
        sys.exit(1)

    if args.engine is None:
        base = os.path.splitext(args.onnx)[0]
        args.engine = f"{base}_b{args.max_batch}.engine"

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        max_batch=args.max_batch,
        input_size=args.input_size,
        fp16=not args.no_fp16,
        workspace_mb=args.workspace_mb,
    )


if __name__ == "__main__":
    main()
