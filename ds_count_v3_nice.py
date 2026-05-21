"""
ds_count_v3_nice.py
===================
5-camera DeepStream YOLOv26n inference for Jetson Nano (JetPack 4.x / TRT 8.2).

What's different vs ds_count_v2.py
-----------------------------------
This file is built for the **YOLO26n end2end ONNX** produced by the Colab
notebook (`ONNX_Exporter_YOLO26n_BATCH5.py`), where:
  - NMS is BAKED into the graph         → output is final post-NMS detections
  - Mod ops have been replaced          → TRT 8.2 can parse it
  - Div(255) is prepended               → input is [0, 255] (no host /255 needed)
  - dynamic batch [1..5]                 → one engine handles 1..5 streams

Output tensor shape: (batch, 300, 6)
  per row = [x1, y1, x2, y2, score, class_id]
  coords are in 640×640 muxer space (we un-letterbox to original cam resolution)

So we do NOT use the DeepStream-Yolo custom parser. Instead:
  - nvinfer config sets `network-type=100` + `output-tensor-meta=1`
  - The Python probe reads the raw output tensor per frame and decodes it
    with plain numpy — no custom CUDA, no plugin SO required

Same DeepStream pipeline structure as v2:
  5x nvurisrcbin → nvstreammux(batch=5, 640×640, push=40ms) →
      nvinfer → fakesink
  (probe on nvinfer's src pad)

Same robustness features as v2:
  - nvurisrcbin auto-reconnect for RTSP
  - GStreamer bus message handler (ERROR/WARNING/EOS)
  - line-buffered CSV with periodic flush
  - per-camera frame counters
  - SIGINT/SIGTERM graceful shutdown

Run:
    cd /home/pang-1/dev
    python3 ds_count_v3_nice.py

Note on engine + cwd:
    nvinfer resolves `model-engine-file` relative to its own working dir.
    Run from /home/pang-1/dev (where yolo26n_b5.engine lives) for the simplest setup.
"""
import csv
import ctypes
import signal
import sys
import time
from collections import defaultdict

import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds


# ── Config (edit if cameras change) ─────────────────────────────────
SOURCES = {
    "cam-a": "rtsp://172.16.30.111:8554/vdo1",
    "cam-b": "rtsp://172.16.30.111:8554/vdo2",
    "cam-c": "rtsp://172.16.30.111:8554/vdo3",
    "cam-d": "rtsp://172.16.30.111:8554/vdo4",
    "cam-e": "rtsp://172.16.30.111:8554/vdo5",
}

INFER_CONFIG = "config_infer_yolo26.txt"
LOG_CSV      = "detect_log_yolo26_20260521.csv"

CONF_THRESHOLD       = 0.25
REPORT_EVERY_S       = 300
CSV_FLUSH_EVERY_ROWS = 60

MUXER_W, MUXER_H         = 640, 640
BATCHED_PUSH_TIMEOUT_US  = 40_000      # 40 ms

# Output layer geometry produced by the Colab end2end export.
# (300 = max detections per image, 6 = [x1,y1,x2,y2,score,cls])
MAX_DET_PER_IMAGE = 300
DET_FEATURES      = 6


COCO_CLASSES = [
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


# ── State (per process — pipeline runs in main thread) ──────────────
cam_names         = list(SOURCES.keys())
cam_frame_count   = defaultdict(int)
total_counts      = {c: defaultdict(int) for c in cam_names}
batch_count       = 0
last_report_t     = time.time()
rows_since_flush  = 0

csv_file   = None
csv_writer = None
g_main_loop = None


# ────────────────────────────────────────────────────────────────────
#  Helper: read a raw float32 layer buffer into a numpy array
# ────────────────────────────────────────────────────────────────────
def layer_to_numpy(layer):
    """
    Convert NvDsInferLayerInfo whose buffer is float32 into a numpy view.
    Returns a numpy array of the layer's declared shape, dtype float32.
    """
    dims = layer.dims
    # NvDsInferDims has numDims + d[8]; total = product of d[0..numDims-1]
    shape = tuple(int(dims.d[i]) for i in range(dims.numDims))
    n_elem = 1
    for s in shape:
        n_elem *= s
    if n_elem == 0:
        return np.empty(shape, dtype=np.float32)
    # layer.buffer is a C pointer (int address)
    ptr_type = ctypes.POINTER(ctypes.c_float)
    cptr = ctypes.cast(pyds.get_ptr(layer.buffer), ptr_type)
    arr = np.ctypeslib.as_array(cptr, shape=(n_elem,)).copy().reshape(shape)
    return arr


# ────────────────────────────────────────────────────────────────────
#  Probe on nvinfer src pad
# ────────────────────────────────────────────────────────────────────
def infer_src_pad_buffer_probe(pad, info, u_data):
    global batch_count, last_report_t, rows_since_flush

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None:
        return Gst.PadProbeReturn.OK


    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        src_id = frame_meta.source_id
        cam_name = cam_names[src_id] if src_id < len(cam_names) else f"src-{src_id}"
        cam_frame_count[cam_name] += 1
        this_frame = cam_frame_count[cam_name]

        # ── Find the NvDsInferTensorMeta attached to this frame ─────
        tensor_meta = None
        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break
            if user_meta.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                break
            try:
                l_user = l_user.next
            except StopIteration:
                break

        frame_cls = defaultdict(lambda: {"count": 0, "max_conf": 0.0})

        if tensor_meta is not None and tensor_meta.num_output_layers >= 1:
            # We only expect 1 output layer for end2end YOLO26
            layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
            arr = layer_to_numpy(layer)        # shape (300, 6) or (1, 300, 6)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]                   # collapse leading 1

            # Filter by confidence
            keep = arr[:, 4] > CONF_THRESHOLD
            dets = arr[keep]

            for row in dets:
                x1, y1, x2, y2, score, cls = row
                cid = int(cls)
                cname = (COCO_CLASSES[cid]
                         if 0 <= cid < len(COCO_CLASSES)
                         else f"class-{cid}")
                entry = frame_cls[cname]
                entry["count"] += 1
                if score > entry["max_conf"]:
                    entry["max_conf"] = float(score)
                total_counts[cam_name][cname] += 1

        # Emit CSV rows for this frame (per class)
        for cls, info_d in frame_cls.items():
            csv_writer.writerow([
                ts, cam_name, this_frame, cls,
                f"{info_d['max_conf']:.3f}", info_d["count"]
            ])
            rows_since_flush += 1

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    batch_count += 1

    if rows_since_flush >= CSV_FLUSH_EVERY_ROWS:
        csv_file.flush()
        rows_since_flush = 0

    now = time.time()
    if now - last_report_t >= REPORT_EVERY_S:
        print_report()
        last_report_t = now

    return Gst.PadProbeReturn.OK


def print_report():
    print("\n" + "=" * 60)
    print(f"[REPORT] {time.strftime('%Y-%m-%d %H:%M:%S')} | batches={batch_count}")
    for cam in cam_names:
        n = cam_frame_count[cam]
        counts = total_counts[cam]
        if not counts:
            print(f"  {cam:<6} frames={n:>6}  (no detections)")
            continue
        top = sorted(counts.items(), key=lambda x: -x[1])[:6]
        top_s = " ".join(f"{c}={k}" for c, k in top)
        print(f"  {cam:<6} frames={n:>6}  {top_s}")
    print("=" * 60 + "\n")
    sys.stdout.flush()


# ────────────────────────────────────────────────────────────────────
#  GStreamer bus message handler
# ────────────────────────────────────────────────────────────────────
def on_bus_message(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("[BUS] EOS — all streams ended")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        src = message.src.get_name() if message.src else "?"
        print(f"[BUS][ERROR] {src}: {err.message}")
        if dbg: print(f"          debug: {dbg}")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, dbg = message.parse_warning()
        src = message.src.get_name() if message.src else "?"
        print(f"[BUS][WARN ] {src}: {err.message}")
    elif t == Gst.MessageType.STATE_CHANGED:
        if message.src and message.src.get_name() == "pipeline":
            old, new, _ = message.parse_state_changed()
            print(f"[BUS] pipeline: {old.value_nick} -> {new.value_nick}")
    return True


# ────────────────────────────────────────────────────────────────────
#  Pipeline builder
# ────────────────────────────────────────────────────────────────────
def make(factory, name):
    e = Gst.ElementFactory.make(factory, name)
    if e is None:
        raise RuntimeError(f"Cannot create element '{factory}' (name={name})")
    return e


def build_pipeline():
    Gst.init(None)
    pipeline = Gst.Pipeline.new("pipeline")

    streammux = make("nvstreammux", "streammux")
    streammux.set_property("width", MUXER_W)
    streammux.set_property("height", MUXER_H)
    streammux.set_property("batch-size", len(SOURCES))
    streammux.set_property("batched-push-timeout", BATCHED_PUSH_TIMEOUT_US)
    streammux.set_property("live-source", 1)
    pipeline.add(streammux)

    # 5 sources, each through nvurisrcbin (with RTSP auto-reconnect)
    for i, (cam_name, url) in enumerate(SOURCES.items()):
        src = Gst.ElementFactory.make("nvurisrcbin", f"src-{i}")
        if src is None:
            print(f"[WARN] nvurisrcbin missing, fallback uridecodebin for {cam_name}")
            src = make("uridecodebin", f"src-{i}")
        src.set_property("uri", url)
        try:
            src.set_property("rtsp-reconnect-interval", 5)
            src.set_property("rtsp-reconnect-attempts", -1)
            src.set_property("latency", 200)
            src.set_property("select-rtp-protocol", 4)  # TCP
        except Exception:
            pass

        def on_pad_added(src_el, pad, mux=streammux, idx=i, name=cam_name):
            sink_pad = mux.get_request_pad(f"sink_{idx}")
            if sink_pad is None:
                print(f"[ERR] cannot request sink_{idx} for {name}")
                return
            if not sink_pad.is_linked():
                ret = pad.link(sink_pad)
                if ret != Gst.PadLinkReturn.OK:
                    print(f"[ERR] link failed for {name}: {ret}")
                else:
                    print(f"[OK] linked {name} -> streammux sink_{idx}")

        src.connect("pad-added", on_pad_added)
        pipeline.add(src)

    nvinfer = make("nvinfer", "nvinfer")
    nvinfer.set_property("config-file-path", INFER_CONFIG)
    pipeline.add(nvinfer)

    sink = make("fakesink", "sink")
    sink.set_property("sync", 0)
    sink.set_property("async", 0)
    pipeline.add(sink)

    if not streammux.link(nvinfer):
        raise RuntimeError("link streammux -> nvinfer failed")
    if not nvinfer.link(sink):
        raise RuntimeError("link nvinfer -> sink failed")

    nvinfer_src = nvinfer.get_static_pad("src")
    nvinfer_src.add_probe(Gst.PadProbeType.BUFFER,
                          infer_src_pad_buffer_probe, 0)
    return pipeline


# ────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────
def main():
    global csv_file, csv_writer, g_main_loop, last_report_t

    csv_file = open(LOG_CSV, "w", newline="", buffering=1)
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp", "cam", "cam_frame", "class",
        "confidence", "count_this_frame"
    ])

    pipeline = build_pipeline()
    g_main_loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, g_main_loop)

    def _sig(*_):
        print("\n[SIG] stopping...")
        g_main_loop.quit()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    print(f"[INFO] Pipeline starting — YOLO26 end2end, {len(SOURCES)} cameras")
    print(f"[INFO] Sources : {list(SOURCES.keys())}")
    print(f"[INFO] Config  : {INFER_CONFIG}")
    print(f"[INFO] CSV     : {LOG_CSV}")
    print(f"[INFO] Conf threshold: {CONF_THRESHOLD}")
    print(f"[INFO] Ctrl+C to stop\n")

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[FATAL] Unable to set pipeline to PLAYING")
        sys.exit(1)

    last_report_t = time.time()

    try:
        g_main_loop.run()
    finally:
        print("\n[INFO] Stopping pipeline...")
        print_report()
        pipeline.set_state(Gst.State.NULL)
        try:
            csv_file.flush()
            csv_file.close()
        except Exception:
            pass
        print(f"[INFO] CSV saved -> {LOG_CSV}")


if __name__ == "__main__":
    main()
