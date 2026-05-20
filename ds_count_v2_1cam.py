"""
ds_count_v2_1cam.py
===================
SINGLE-CAMERA version of ds_count_v2.py — for performance benchmarking.

Use this to measure baseline per-camera throughput, then compare with the
5-camera batched version (ds_count_v2.py) to see how batching scales.

Differences vs ds_count_v2.py:
  - 1 source only (configurable via SOURCE constant or --source flag)
  - nvstreammux batch-size = 1
  - Default model = yolov8n.engine (the existing batch=1 engine)
  - Adds latency tracking via probe timestamps:
      * per-batch inference latency (p50 / p95 / p99 over rolling window)
      * end-to-end FPS measured at the probe
      * total frames processed
  - CSV path: detect_log_1cam.csv (so it doesn't clobber the 5-cam log)

Run:
    cd /home/pang-1/dev
    python3 ds_count_v2_1cam.py
    python3 ds_count_v2_1cam.py --source rtsp://10.0.11.37:8554/vdo1 --duration 60

Compare with ds_count_v2.py (5 cams) by:
    # Per-camera FPS expected: ~9-15 fps depending on input size + thermal
    # If 5-cam batched achieves > 1/5 of 1-cam FPS per camera, batching helps.
"""
import argparse
import csv
import os
import signal
import statistics
import sys
import time
from collections import defaultdict, deque

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds


# ── Defaults (override with CLI flags) ───────────────────────────
DEFAULT_SOURCE = "rtsp://10.0.11.37:8554/vdo1"
DEFAULT_INFER_CONFIG = "config_infer_yolov8.txt"
DEFAULT_LOG_CSV = "detect_log_1cam.csv"
DEFAULT_MODEL_OVERRIDE = "yolov8n.engine"   # the existing batch=1 engine

REPORT_EVERY_S = 30                          # shorter than 5-cam version, for live perf
CSV_FLUSH_EVERY_ROWS = 60
MUXER_W, MUXER_H = 640, 640
BATCHED_PUSH_TIMEOUT_US = 40_000
LATENCY_WINDOW = 300                         # rolling window for p50/p95/p99

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


# ── State ────────────────────────────────────────────────────────
state = {
    "cam_name": "cam-1",
    "frame_count": 0,
    "total_counts": defaultdict(int),
    "last_report_t": 0.0,
    "rows_since_flush": 0,
    "loop_t0": 0.0,
    "last_probe_t": None,
    "inter_arrival_ms": deque(maxlen=LATENCY_WINDOW),
    "first_frame_t": None,
    "csv_writer": None,
    "csv_file": None,
    "g_main_loop": None,
}


def pct(seq, q):
    if not seq:
        return 0.0
    s = sorted(seq)
    k = max(0, min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1)))))
    return s[k]


# ── Probe on nvinfer src pad ─────────────────────────────────────
def infer_src_pad_buffer_probe(pad, info, u_data):
    """
    Called on every batched buffer leaving nvinfer.
    For 1-cam case the batch holds exactly 1 frame.
    """
    s = state
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None:
        return Gst.PadProbeReturn.OK

    now = time.monotonic()
    if s["first_frame_t"] is None:
        s["first_frame_t"] = now
    if s["last_probe_t"] is not None:
        s["inter_arrival_ms"].append((now - s["last_probe_t"]) * 1000.0)
    s["last_probe_t"] = now

    l_frame = batch_meta.frame_meta_list
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        s["frame_count"] += 1
        frame_no = s["frame_count"]

        frame_cls = defaultdict(lambda: {"count": 0, "max_conf": 0.0})

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            cid = obj_meta.class_id
            conf = float(obj_meta.confidence)
            cname = (COCO_CLASSES[cid]
                     if 0 <= cid < len(COCO_CLASSES)
                     else f"class-{cid}")

            entry = frame_cls[cname]
            entry["count"] += 1
            if conf > entry["max_conf"]:
                entry["max_conf"] = conf
            s["total_counts"][cname] += 1

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        for cls, info_d in frame_cls.items():
            s["csv_writer"].writerow([
                ts, s["cam_name"], frame_no, cls,
                f"{info_d['max_conf']:.3f}", info_d["count"]
            ])
            s["rows_since_flush"] += 1

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    if s["rows_since_flush"] >= CSV_FLUSH_EVERY_ROWS:
        s["csv_file"].flush()
        s["rows_since_flush"] = 0

    if now - s["last_report_t"] >= REPORT_EVERY_S:
        print_report(now)
        s["last_report_t"] = now

    return Gst.PadProbeReturn.OK


def print_report(now=None):
    s = state
    if now is None:
        now = time.monotonic()
    if s["first_frame_t"] is None:
        print("[REPORT] no frames yet")
        return

    elapsed = now - s["first_frame_t"]
    fps_overall = s["frame_count"] / elapsed if elapsed > 0 else 0.0
    arr = list(s["inter_arrival_ms"])

    print()
    print("─" * 70)
    print(f"[REPORT] {time.strftime('%Y-%m-%d %H:%M:%S')}  cam={s['cam_name']}")
    print(f"  frames                : {s['frame_count']}")
    print(f"  elapsed               : {elapsed:.1f}s")
    print(f"  FPS (overall)         : {fps_overall:.2f}")
    if arr:
        print(f"  inter-arrival (ms)    : "
              f"p50={pct(arr,50):.1f}  "
              f"p95={pct(arr,95):.1f}  "
              f"p99={pct(arr,99):.1f}  "
              f"mean={statistics.mean(arr):.1f}  "
              f"min={min(arr):.1f}  "
              f"max={max(arr):.1f}")
        recent_fps = 1000.0 / statistics.mean(arr) if arr else 0
        print(f"  FPS (rolling {LATENCY_WINDOW}f)  : {recent_fps:.2f}")
    if s["total_counts"]:
        top = sorted(s["total_counts"].items(), key=lambda x: -x[1])[:6]
        top_s = " ".join(f"{c}={k}" for c, k in top)
        print(f"  top classes           : {top_s}")
    print("─" * 70)
    sys.stdout.flush()


# ── GStreamer bus message handler ────────────────────────────────
def on_bus_message(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("[BUS] EOS — stream ended")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        src = message.src.get_name() if message.src else "?"
        print(f"[BUS][ERROR] {src}: {err.message}")
        if dbg:
            print(f"          debug: {dbg}")
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


# ── Build pipeline ──────────────────────────────────────────────
def build_pipeline(source_uri: str, infer_config: str):
    Gst.init(None)
    pipeline = Gst.Pipeline.new("pipeline")

    # nvstreammux with batch=1
    streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
    if streammux is None:
        raise RuntimeError("nvstreammux factory failed — DeepStream missing?")
    streammux.set_property("width", MUXER_W)
    streammux.set_property("height", MUXER_H)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", BATCHED_PUSH_TIMEOUT_US)
    streammux.set_property("live-source", 1)
    pipeline.add(streammux)

    # Single source — prefer nvurisrcbin for RTSP reconnect
    src = Gst.ElementFactory.make("nvurisrcbin", "src-0")
    if src is None:
        print("[WARN] nvurisrcbin not available, falling back to uridecodebin")
        src = Gst.ElementFactory.make("uridecodebin", "src-0")
    src.set_property("uri", source_uri)
    try:
        src.set_property("rtsp-reconnect-interval", 5)
        src.set_property("rtsp-reconnect-attempts", -1)
        src.set_property("latency", 200)
        src.set_property("select-rtp-protocol", 4)  # TCP
    except Exception:
        pass

    def on_pad_added(src_el, pad, mux=streammux):
        sink_pad = mux.get_request_pad("sink_0")
        if sink_pad and not sink_pad.is_linked():
            ret = pad.link(sink_pad)
            if ret != Gst.PadLinkReturn.OK:
                print(f"[ERR] link failed: {ret}")
            else:
                print(f"[OK] linked src -> streammux sink_0")

    src.connect("pad-added", on_pad_added)
    pipeline.add(src)

    # nvinfer
    nvinfer = Gst.ElementFactory.make("nvinfer", "nvinfer")
    if nvinfer is None:
        raise RuntimeError("nvinfer factory failed")
    nvinfer.set_property("config-file-path", infer_config)
    pipeline.add(nvinfer)

    # fakesink — discard buffers (probe reads metadata)
    sink = Gst.ElementFactory.make("fakesink", "sink")
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


# ── Main ────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="1-camera DeepStream YOLOv8 benchmarker")
    p.add_argument("--source", default=DEFAULT_SOURCE,
                   help=f"RTSP URI or file:// URI (default: {DEFAULT_SOURCE})")
    p.add_argument("--cam-name", default="cam-1",
                   help="Name to log in CSV (default: cam-1)")
    p.add_argument("--config", default=DEFAULT_INFER_CONFIG,
                   help=f"nvinfer config (default: {DEFAULT_INFER_CONFIG})")
    p.add_argument("--csv", default=DEFAULT_LOG_CSV,
                   help=f"CSV output path (default: {DEFAULT_LOG_CSV})")
    p.add_argument("--duration", type=int, default=0,
                   help="Auto-stop after N seconds (0 = run until Ctrl+C)")
    args = p.parse_args()

    state["cam_name"] = args.cam_name

    # Open CSV
    state["csv_file"] = open(args.csv, "w", newline="", buffering=1)
    state["csv_writer"] = csv.writer(state["csv_file"])
    state["csv_writer"].writerow([
        "timestamp", "cam", "cam_frame", "class",
        "confidence", "count_this_frame"
    ])

    pipeline = build_pipeline(args.source, args.config)
    loop = GLib.MainLoop()
    state["g_main_loop"] = loop

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, loop)

    def _sig(*_):
        print("\n[SIG] stopping...")
        loop.quit()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # Auto-stop timer
    if args.duration > 0:
        def _stop_after():
            print(f"\n[DURATION] {args.duration}s reached — stopping")
            loop.quit()
            return False
        GLib.timeout_add_seconds(args.duration, _stop_after)

    print(f"[INFO] Pipeline starting — 1 camera, batch=1")
    print(f"[INFO] Source : {args.source}")
    print(f"[INFO] Config : {args.config}")
    print(f"[INFO] CSV    : {args.csv}")
    if args.duration > 0:
        print(f"[INFO] Duration: {args.duration}s")
    print(f"[INFO] Report every {REPORT_EVERY_S}s. Ctrl+C to stop.\n")

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[FATAL] Unable to set pipeline to PLAYING")
        sys.exit(1)

    state["loop_t0"] = time.monotonic()
    state["last_report_t"] = state["loop_t0"]

    try:
        loop.run()
    finally:
        print("\n[INFO] Stopping pipeline...")
        print_report()
        pipeline.set_state(Gst.State.NULL)
        try:
            state["csv_file"].flush()
            state["csv_file"].close()
        except Exception:
            pass
        print(f"[INFO] Log saved -> {args.csv}")


if __name__ == "__main__":
    main()
