"""
ds_count_v2.py
==============
DeepStream multi-stream YOLOv8 counter for Jetson Nano (5 RTSP cameras).

Improvements over ds_count.py:
  - batched-push-timeout    : 4_000_000 us -> 40_000 us (40 ms)
  - source element          : uridecodebin -> nvurisrcbin (auto RTSP reconnect)
  - confidence              : actual obj_meta.confidence logged in CSV
  - GStreamer bus           : signal-watch handler logs ERROR / WARNING / EOS
  - CSV                     : line-buffered + explicit flush every N rows
  - per-camera frame_count  : tracked separately (not one global counter)
  - graceful shutdown       : SIGINT/SIGTERM stop MainLoop cleanly
  - removed nvdsosd         : on-screen display is unnecessary for headless count
                              (probe attaches to nvinfer src pad instead)

Run:
    cd /home/pang-1/dev
    python3 ds_count_v2.py
"""
import csv
import signal
import sys
import time
from collections import defaultdict

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds


# ── Config ────────────────────────────────────────────────────────
SOURCES = {
    "cam-a": "rtsp://10.0.11.176:8554/vdo1",
    "cam-b": "rtsp://10.0.11.176:8554/vdo2",
    "cam-c": "rtsp://10.0.11.176:8554/vdo3",
    "cam-d": "rtsp://10.0.11.176:8554/vdo4",
    "cam-e": "rtsp://10.0.11.176:8554/vdo5",
}
INFER_CONFIG = "config_infer_yolov8.txt"
LOG_CSV = "detect_log.csv"
REPORT_EVERY_S = 300
CSV_FLUSH_EVERY_ROWS = 60        # flush to disk every N rows (~2s @ 30fps)

MUXER_W, MUXER_H = 640, 640
BATCHED_PUSH_TIMEOUT_US = 40_000  # 40 ms — was 4 s in v1

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


# ── State ─────────────────────────────────────────────────────────
cam_names = list(SOURCES.keys())

# Per-cam tracking
cam_frame_count = defaultdict(int)
total_counts = {c: defaultdict(int) for c in cam_names}
batch_count = 0

last_report_t = time.time()
rows_since_flush = 0

# Open CSV with line buffering (buffering=1 = newline-flushed)
csv_file = open(LOG_CSV, "w", newline="", buffering=1)
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "timestamp", "cam", "cam_frame", "class",
    "confidence", "count_this_frame"
])

# Main loop handle (for graceful shutdown)
g_main_loop = None


# ── Probe on nvinfer src pad ─────────────────────────────────────
def infer_src_pad_buffer_probe(pad, info, u_data):
    """
    Called on every batched buffer leaving nvinfer.
    Iterate each frame in the batch, then each object meta on the frame.
    """
    global batch_count, last_report_t, rows_since_flush

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        src_id = frame_meta.source_id
        cam_name = (cam_names[src_id]
                    if src_id < len(cam_names)
                    else f"src-{src_id}")
        cam_frame_count[cam_name] += 1
        this_frame = cam_frame_count[cam_name]

        # Aggregate per-class for this frame
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

            total_counts[cam_name][cname] += 1

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Write one CSV row per (frame, class). Confidence = max conf in that frame.
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

    # Periodic flush (line-buffering handles per-row, but force to disk for safety)
    if rows_since_flush >= CSV_FLUSH_EVERY_ROWS:
        csv_file.flush()
        rows_since_flush = 0

    # Periodic report
    now = time.time()
    if now - last_report_t >= REPORT_EVERY_S:
        print_report()
        last_report_t = now

    return Gst.PadProbeReturn.OK


def print_report():
    print("\n" + "=" * 60)
    print(f"[REPORT] {time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"| batches={batch_count}")
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


# ── GStreamer bus message handler ────────────────────────────────
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
        # Log pipeline transitions only (noise filter)
        if message.src and message.src.get_name() == "pipeline":
            old, new, _ = message.parse_state_changed()
            print(f"[BUS] pipeline: {old.value_nick} -> {new.value_nick}")
    return True


# ── Build pipeline ──────────────────────────────────────────────
def build_pipeline():
    Gst.init(None)
    pipeline = Gst.Pipeline.new("pipeline")

    # nvstreammux — batches frames from N sources
    streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
    if streammux is None:
        raise RuntimeError("Unable to create nvstreammux — DeepStream plugins missing?")
    streammux.set_property("width", MUXER_W)
    streammux.set_property("height", MUXER_H)
    streammux.set_property("batch-size", len(SOURCES))
    streammux.set_property("batched-push-timeout", BATCHED_PUSH_TIMEOUT_US)
    streammux.set_property("live-source", 1)
    pipeline.add(streammux)

    # Sources — nvurisrcbin has built-in RTSP keepalive + reconnect
    for i, (cam_name, url) in enumerate(SOURCES.items()):
        src = Gst.ElementFactory.make("nvurisrcbin", f"src-{i}")
        if src is None:
            # Fallback: nvurisrcbin only available in newer DS; use uridecodebin
            print(f"[WARN] nvurisrcbin not available, falling back to uridecodebin "
                  f"for {cam_name}")
            src = Gst.ElementFactory.make("uridecodebin", f"src-{i}")
        src.set_property("uri", url)
        # nvurisrcbin-specific knobs (no-op on uridecodebin fallback)
        try:
            src.set_property("rtsp-reconnect-interval", 5)         # try every 5s
            src.set_property("rtsp-reconnect-attempts", -1)        # infinite
            src.set_property("latency", 200)                       # ms jitter buffer
            src.set_property("select-rtp-protocol", 4)             # 4=TCP only
        except Exception:
            pass

        def on_pad_added(src_el, pad, mux=streammux, idx=i, name=cam_name):
            sink_pad = mux.get_request_pad(f"sink_{idx}")
            if sink_pad is None:
                print(f"[ERR] cannot request sink_{idx} on streammux for {name}")
                return
            if not sink_pad.is_linked():
                ret = pad.link(sink_pad)
                if ret != Gst.PadLinkReturn.OK:
                    print(f"[ERR] link failed for {name}: {ret}")
                else:
                    print(f"[OK] linked {name} -> streammux sink_{idx}")

        src.connect("pad-added", on_pad_added)
        pipeline.add(src)

    # nvinfer — runs TensorRT engine
    nvinfer = Gst.ElementFactory.make("nvinfer", "nvinfer")
    if nvinfer is None:
        raise RuntimeError("Unable to create nvinfer")
    nvinfer.set_property("config-file-path", INFER_CONFIG)
    pipeline.add(nvinfer)

    # fakesink — discard buffers (we read metadata via probe)
    sink = Gst.ElementFactory.make("fakesink", "sink")
    sink.set_property("sync", 0)
    sink.set_property("async", 0)
    pipeline.add(sink)

    # Link static elements
    if not streammux.link(nvinfer):
        raise RuntimeError("Failed to link streammux -> nvinfer")
    if not nvinfer.link(sink):
        raise RuntimeError("Failed to link nvinfer -> sink")

    # Probe on nvinfer src pad (skip OSD, save CPU)
    nvinfer_src = nvinfer.get_static_pad("src")
    nvinfer_src.add_probe(Gst.PadProbeType.BUFFER,
                          infer_src_pad_buffer_probe, 0)

    return pipeline


# ── Main ────────────────────────────────────────────────────────
def main():
    global g_main_loop

    pipeline = build_pipeline()
    g_main_loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, g_main_loop)

    # Signal handlers (clean stop on Ctrl+C / SIGTERM)
    def _sig(*_):
        print("\n[SIG] stopping...")
        g_main_loop.quit()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    print(f"[INFO] Pipeline starting — {len(SOURCES)} cameras, "
          f"muxer={MUXER_W}x{MUXER_H}, push-timeout={BATCHED_PUSH_TIMEOUT_US}us")
    print(f"[INFO] Sources: {list(SOURCES.keys())}")
    print(f"[INFO] Logging to {LOG_CSV}, periodic report every {REPORT_EVERY_S}s")
    print(f"[INFO] Ctrl+C to stop\n")

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[FATAL] Unable to set pipeline to PLAYING")
        sys.exit(1)

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
        print(f"[INFO] Log saved -> {LOG_CSV}")


if __name__ == "__main__":
    main()
