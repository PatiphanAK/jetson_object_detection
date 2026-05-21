"""
build_shufflenet_engine.py
===========================
TRT 8.2 engine builder for shufflenet_v2_x0_5_prune_30_fp16.onnx.

ONNX has FP16 weights + dynamic batch. Builds an FP16 engine with
optimization profile batch 1..MAX (default 16).
"""
import time
import tensorrt as trt

ONNX = "shufflenet_v2_x0_5_prune_30_fp16.onnx"
ENGINE = "shufflenet_brand.engine"
INPUT_NAME = "input"
IMG = 224
MAX_BATCH = 16
OPT_BATCH = 8
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
for i in range(network.num_inputs):
    t = network.get_input(i); print(f"  input  '{t.name}' shape={tuple(t.shape)} dtype={t.dtype}")
for i in range(network.num_outputs):
    t = network.get_output(i); print(f"  output '{t.name}' shape={tuple(t.shape)} dtype={t.dtype}")

print(f"\n[2/4] Configure builder (dynamic batch 1..{MAX_BATCH})")
config = builder.create_builder_config()
config.max_workspace_size = WORKSPACE_MB * (1 << 20)
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
    print("  FP16 enabled")

profile = builder.create_optimization_profile()
profile.set_shape(INPUT_NAME,
                  (1, 3, IMG, IMG),
                  (OPT_BATCH, 3, IMG, IMG),
                  (MAX_BATCH, 3, IMG, IMG))
config.add_optimization_profile(profile)
print(f"  profile min=(1,3,{IMG},{IMG}) opt=({OPT_BATCH},...) max=({MAX_BATCH},...)")

print(f"\n[3/4] Build engine")
t0 = time.monotonic()
serialized = builder.build_serialized_network(network, config)
dt = time.monotonic() - t0
if serialized is None: raise SystemExit("build returned None")
print(f"  built in {dt:.1f}s")

print(f"\n[4/4] Save: {ENGINE}")
with open(ENGINE, "wb") as f: f.write(serialized)
import os
print(f"  saved {os.path.getsize(ENGINE)/1e6:.2f} MB")
