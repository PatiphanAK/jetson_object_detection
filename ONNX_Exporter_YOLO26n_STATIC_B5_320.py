"""
ONNX_Exporter_YOLO26n_STATIC_B5.py
===================================
STATIC batch=5 version — avoids TRT 8.2 dynamic-Range INT64 issue.

Same surgery as the dynamic version, but the ONNX has batch hard-coded to 5.
Engine ends up fixed at batch=5, which is exactly what the 5-cam pipeline
needs. nvstreammux mux=5 -> nvinfer batch=5 -> 1 call/frame, no padding.

Produces:
    yolo26n_wpost_b5.onnx
    -- input  'images_raw' shape (5, 3, 640, 640) float32 in [0,255]
    -- output 'output0'    shape (5, 300, 6)
    -- TRT 8.2 friendly (Mod-free) + end-to-end NMS baked in
"""
import shutil
from pathlib import Path

from ultralytics import YOLO, checks
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

checks()

# ---- export with STATIC batch=5 ----
print("[2] Exporting yolo26n.pt -> ONNX  (opset=12, end2end=True, batch=5, dynamic=False)")
YOLO("yolo26n.pt").export(
    format="onnx",
    opset=12,
    imgsz=320,
    end2end=True,
    simplify=True,
    dynamic=False,
    batch=5,
    half=False,
)
shutil.move("yolo26n.onnx", "yolo26n_std.onnx")
print("    saved: yolo26n_std.onnx")

# ---- surgery (same as dynamic version) ----
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

# ---- prepend Div(255) (static dims) ----
g = m.graph
orig = g.input[0]
orig_name = orig.name
orig_dims = [d.dim_value if d.dim_value > 0 else d.dim_param for d in orig.type.tensor_type.shape.dim]
print(f"    original input '{orig_name}' dims: {orig_dims}")

new_in = helper.make_tensor_value_info(
    "images_raw",
    TensorProto.FLOAT,
    orig_dims,
)
g.initializer.append(helper.make_tensor("div255", TensorProto.FLOAT, [], [255.0]))
div_node = helper.make_node("Div", inputs=["images_raw", "div255"], outputs=[orig_name], name="pp_normalize")
g.input.remove(orig)
g.input.insert(0, new_in)
g.node.insert(0, div_node)
print(f"    prepended Div(255) — input now 'images_raw' shape {orig_dims}")

onnx.checker.check_model(m)
onnx.save(m, "yolo26n_wpost_b5.onnx")
print("    saved: yolo26n_wpost_b5.onnx  (Mod-free + /255 + static batch=5 @ 320x320)")

# ---- inspect ----
print("\n[4] Final ONNX shapes (batch should be 5)")
mf = onnx.load("yolo26n_wpost_b5.onnx")
for tag, vil in (("input", mf.graph.input), ("output", mf.graph.output)):
    for v in vil:
        dims = [(d.dim_param or d.dim_value) for d in v.type.tensor_type.shape.dim]
        print(f"    {tag:6} '{v.name}': {dims}")

# ---- sanity inference batch=5 ----
print("\n[5] ORT sanity: batch=5 inference")
dummy = np.random.randint(0, 256, (5, 3, 640, 640)).astype(np.float32)
sess = ort.InferenceSession("yolo26n_wpost_b5.onnx", providers=["CPUExecutionProvider"])
out = sess.run(None, {sess.get_inputs()[0].name: dummy})[0]
print(f"    out shape: {out.shape}  (expect (5, 300, 6))")
print("done.")
