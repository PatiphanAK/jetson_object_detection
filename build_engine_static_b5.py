"""
build_engine_static_b5.py
=========================
TRT 8.2 engine builder for the STATIC batch=5 yolo26n_wpost_b5.onnx.

No optimization profile (input shape is already fixed). FP16, 1024MB workspace.
"""
import time
import tensorrt as trt

ONNX = "yolo26n_wpost_b5.onnx"
ENGINE = "yolo26n_b5.engine"
WORKSPACE_MB = 512

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flag)
parser = trt.OnnxParser(network, logger)

print(f"[1/4] Parse ONNX: {ONNX}")
with open(ONNX, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print("  ERR:", parser.get_error(i))
        raise SystemExit("parse failed")
print(f"  inputs={network.num_inputs}, outputs={network.num_outputs}")
inp = network.get_input(0)
print(f"  input '{inp.name}' shape: {tuple(inp.shape)}")
out = network.get_output(0)
print(f"  output '{out.name}' shape: {tuple(out.shape)}")

print(f"\n[2/4] Configure builder (static batch — no profile)")
config = builder.create_builder_config()
config.max_workspace_size = WORKSPACE_MB * (1 << 20)
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
    print(f"  FP16 enabled")
else:
    print(f"  FP32 only")

print(f"\n[3/4] Build engine (5-10 min on Nano)")
t0 = time.monotonic()
serialized = builder.build_serialized_network(network, config)
dt = time.monotonic() - t0
if serialized is None:
    raise SystemExit("build_serialized_network returned None")
print(f"  built in {dt:.1f}s ({dt/60:.1f} min)")

print(f"\n[4/4] Save engine: {ENGINE}")
with open(ENGINE, "wb") as f:
    f.write(serialized)
import os
print(f"  saved {os.path.getsize(ENGINE)/1e6:.2f} MB")
print("done.")
