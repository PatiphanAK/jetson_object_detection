"""
ONNX_Exporter_YOLO26n_BATCH5.py
================================
Modified version of the Colab notebook that exports YOLO26n with
DYNAMIC BATCH (1..5+) instead of fixed batch=1.

Same end2end approach as the original — keeps NMS baked in (so the
Nano gets final detections directly, no custom parser needed) and
surgically rewrites the Mod op so TRT 8.2 can build it.

The only structural change vs the original notebook:
    cell [2]:   dynamic=False   →   dynamic=True

Plus a few additions:
    - test inference with batch=5 (not just batch=1)
    - verify per-image outputs match between batched and single inference
    - print the dynamic dimensions in the final ONNX

Run on Colab (not the Nano).

Produces:
    yolo26n_wpost_b5.onnx
    -- input  'images_raw' shape (-1, 3, 640, 640) float32 in [0,255]
    -- output 'output0'    shape (-1, 300, 6)
    -- TRT 8.2 friendly (Mod-free) + end-to-end NMS baked in
"""

# ============================================================
#  Cell 1 — install + imports
# ============================================================
# %pip install ultralytics onnxruntime onnxslim

import shutil
import urllib.request
from pathlib import Path

from ultralytics import YOLO, checks
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

checks()


# ============================================================
#  Cell 2 — Export with DYNAMIC batch
# ============================================================
print("[2] Exporting yolo26n.pt -> ONNX  (opset=12, end2end=True, DYNAMIC)")

YOLO("yolo26n.pt").export(
    format="onnx",
    opset=12,           # max for TRT 8.2
    imgsz=640,
    end2end=True,       # NMS baked in, output (B, 300, 6)
    simplify=True,
    dynamic=True,       # ★★★ CHANGED — batch dim is now -1 instead of 1
    half=False,
)
shutil.move("yolo26n.onnx", "yolo26n_std.onnx")
print("    saved: yolo26n_std.onnx")


# ============================================================
#  Cell 3 — Surgery (identical to original — works on any batch)
# ============================================================
print("\n[3] Surgery on yolo26n_std.onnx")

m = onnx.load("yolo26n_std.onnx")
m = onnx.shape_inference.infer_shapes(m)

dtype_of = {}
for vi in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
    dtype_of[vi.name] = vi.type.tensor_type.elem_type
for init in m.graph.initializer:
    dtype_of[init.name] = init.data_type

INT_TYPES = {TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64,
             TensorProto.UINT8, TensorProto.UINT16, TensorProto.UINT32, TensorProto.UINT64}

new_nodes = []
n_mod_int = n_mod_float = 0
for node in m.graph.node:
    if node.op_type != "Mod":
        new_nodes.append(node)
        continue
    a, b = node.input[0], node.input[1]
    out = node.output[0]
    base = node.name or f"mod_{n_mod_int + n_mod_float}"
    a_dtype = dtype_of.get(a, dtype_of.get(b, TensorProto.FLOAT))
    if a_dtype in INT_TYPES:
        new_nodes.append(helper.make_node("Div", [a, b],            [f"{base}_q"], name=f"{base}_div"))
        new_nodes.append(helper.make_node("Mul", [f"{base}_q", b],  [f"{base}_p"], name=f"{base}_mul"))
        new_nodes.append(helper.make_node("Sub", [a, f"{base}_p"],  [out],         name=f"{base}_sub"))
        n_mod_int += 1
    else:
        new_nodes.append(helper.make_node("Div",   [a, b],            [f"{base}_q"], name=f"{base}_div"))
        new_nodes.append(helper.make_node("Floor", [f"{base}_q"],     [f"{base}_f"], name=f"{base}_floor"))
        new_nodes.append(helper.make_node("Mul",   [f"{base}_f", b],  [f"{base}_p"], name=f"{base}_mul"))
        new_nodes.append(helper.make_node("Sub",   [a, f"{base}_p"],  [out],         name=f"{base}_sub"))
        n_mod_float += 1

m.graph.ClearField("node")
m.graph.node.extend(new_nodes)
print(f"    replaced Mod ops: {n_mod_int} int + {n_mod_float} float")


# ── Prepend Div(255). Handle dynamic batch (dim_value=0 means dynamic). ────
g = m.graph
orig = g.input[0]
orig_name = orig.name

def dim_to_param(d):
    """Return dim_value if static, else dim_param string (e.g. 'batch')."""
    return d.dim_value if d.dim_value > 0 else d.dim_param

orig_dims = [dim_to_param(d) for d in orig.type.tensor_type.shape.dim]
print(f"    original input '{orig_name}' dims: {orig_dims}")

# Build a TensorValueInfo that PRESERVES the dynamic dim
# Using -1 marker for dynamic positions
new_in = helper.make_tensor_value_info(
    "images_raw",
    TensorProto.FLOAT,
    # helper.make_tensor_value_info accepts strings for dynamic dims
    [d if isinstance(d, str) and d else (d if d else "batch") for d in orig_dims],
)
g.initializer.append(
    helper.make_tensor("div255", TensorProto.FLOAT, [], [255.0])
)
div_node = helper.make_node(
    "Div", inputs=["images_raw", "div255"], outputs=[orig_name], name="pp_normalize"
)
g.input.remove(orig)
g.input.insert(0, new_in)
g.node.insert(0, div_node)
print(f"    prepended Div(255) — input now 'images_raw' shape {orig_dims}")

onnx.checker.check_model(m)
onnx.save(m, "yolo26n_wpost_b5.onnx")
print("    saved: yolo26n_wpost_b5.onnx  (Mod-free + /255 + dynamic batch)")


# ============================================================
#  Cell 4 — Inspect final ONNX dims
# ============================================================
print("\n[4] Final ONNX shapes (should have -1 / dim_param on batch dim)")

mf = onnx.load("yolo26n_wpost_b5.onnx")
for tag, vil in (("input", mf.graph.input), ("output", mf.graph.output)):
    for v in vil:
        dims = [(d.dim_param or d.dim_value) for d in v.type.tensor_type.shape.dim]
        print(f"    {tag:6} '{v.name}': {dims}")


# ============================================================
#  Cell 5 — Test image + letterbox (same as original)
# ============================================================
if not Path("bus.jpg").exists():
    urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", "bus.jpg")
image = cv2.imread("bus.jpg")

def letterbox(img, target=640, pad_value=114):
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((target, target, 3), pad_value, dtype=np.uint8)
    pl, pt = (target - nw) // 2, (target - nh) // 2
    out[pt:pt + nh, pl:pl + nw] = resized
    return out, scale, (pl, pt)

lb, scale, pad = letterbox(image, 640)
rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
single_tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32)  # (1, 3, 640, 640)


# ============================================================
#  Cell 6 — Inference: batch=1 (sanity)
# ============================================================
print("\n[6] ORT inference with batch=1")
sess = ort.InferenceSession("yolo26n_wpost_b5.onnx",
                            providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
out_b1 = sess.run(None, {in_name: single_tensor})[0]
print(f"    out shape: {out_b1.shape}  (expect (1, 300, 6))")


# ============================================================
#  Cell 7 — Inference: batch=5 (the actual change)
# ============================================================
print("\n[7] ORT inference with batch=5")

# Stack 5 copies of the same image so we have ground truth to compare
batch5 = np.repeat(single_tensor, 5, axis=0)   # (5, 3, 640, 640)
print(f"    input shape: {batch5.shape}")

out_b5 = sess.run(None, {in_name: batch5})[0]
print(f"    output shape: {out_b5.shape}  (expect (5, 300, 6))")

# Each of the 5 outputs should equal the batch=1 result
max_diff_per_item = [
    float(np.abs(out_b5[i] - out_b1[0]).max()) for i in range(5)
]
print(f"    per-item max abs diff vs batch=1: {max_diff_per_item}")
ok = all(d < 1e-4 for d in max_diff_per_item)
print(f"    {'✅ batched outputs match single inference' if ok else '❌ MISMATCH'}")


# ============================================================
#  Cell 8 — Bench: throughput batch=1 vs batch=5
# ============================================================
import time

def bench(tensor, name, n=10):
    # warm-up
    sess.run(None, {in_name: tensor})
    t0 = time.monotonic()
    for _ in range(n):
        sess.run(None, {in_name: tensor})
    dt = time.monotonic() - t0
    per_call = dt / n * 1000
    per_image = per_call / tensor.shape[0]
    print(f"    {name:<15} {n} calls in {dt*1000:.1f} ms  "
          f"-> {per_call:.1f} ms/call  ({per_image:.1f} ms/image)")

print("\n[8] Throughput bench (CPU ORT — only indicative, real TRT will be different)")
bench(single_tensor, "batch=1", 10)
bench(batch5,         "batch=5", 10)


# ============================================================
#  Cell 9 — Decode + draw (same logic as original; works per-image)
# ============================================================
COCO = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]

print("\n[9] Decode batch=5 — should give the same 5 dets for every camera")
pl, pt = pad
for cam_idx in range(5):
    d = out_b5[cam_idx].copy()
    d = d[d[:, 4] > 0.25]
    d[:, [0, 2]] -= pl
    d[:, [1, 3]] -= pt
    d[:, :4] /= scale
    print(f"  cam{cam_idx}: {len(d)} detections")
    for x1, y1, x2, y2, sc, c in d:
        nm = COCO[int(c)] if int(c) < len(COCO) else f"cls_{int(c)}"
        print(f"    {nm:10} {sc:.2f}")
