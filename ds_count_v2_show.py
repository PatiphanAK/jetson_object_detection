"""
ds_count_v2_show.py
===================
Same as ds_count_v2.py (5-camera DeepStream YOLOv8 batched inference),
but with an optional --show flag that adds visual tile output.

Modes:
  (no flag)          headless — only CSV log + console report
                     (identical to ds_count_v2.py behavior)

  --show file        tile 5 streams into one grid + draw bounding boxes
                     + encode H.264 via Nano hardware encoder
                     + save to .mp4 (default: tiled_output.mp4)
                     no X display required

  --show rtsp        same tile + OSD, but re-stream as RTSP to view from
                     VLC / other player on your laptop (default port 8555)

  --show screen      open EGL window on local display (only works on the
                     Nano's own monitor, not over SSH)

Examples:
    python3 ds_count_v2_show.py
    python3 ds_count_v2_show.py --show file --show-output ./tiled.mp4
    python3 ds_count_v2_show.py --show rtsp --rtsp-port 8555
        # then on laptop: vlc rtsp://10.0.11.101:8555/ds
    python3 ds_count_v2_show.py --show file --tile-rows 1 --tile-cols 5
"""
import argparse
import csv
import signal
import sys
import time
from collections import defaultdict

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds

# GstRtspServer is imported lazily inside build_show_pipeline() when
# --show rtsp is requested, so users without the rtsp-server gir can
# still use headless / file / screen modes.


# ── Config ───────────────────────────────────────────────────────
SOURCES = {
    "cam-a": "rtsp://10.0.11.37:8554/vdo1",
    "cam-b": "rtsp://10.0.11.37:8554/vdo2",
    "cam-c": "rtsp://10.0.11.37:8554/vdo3",
    "cam-d": "rtsp://10.0.11.37:8554/vdo4",
    "cam-e": "rtsp://10.0.11.37:8554/vdo5",
}
INFER_CONFIG = "config_infer_yolov8.txt"
LOG_CSV = "detect_log.csv"
REPORT_EVERY_S = 300
CSV_FLUSH_EVERY_ROWS = 60
MUXER_W, MUXER_H = 640, 640
BATCHED_PUSH_TIMEOUT_US = 40_000

# Tile output defaults (used when --show enabled)
DEFAULT_TILE_ROWS = 2
DEFAULT_TILE_COLS = 3      # 2x3 = 6 slots, 1 empty (good for 5 streams)
DEFAULT_TILE_W = 1280      # total output width
DEFAULT_TILE_H = 720       # total output height (creates ~427x360 per tile at 3x2)
DEFAULT_OUTPUT_FILE = "tiled_output.mp4"
DEFAULT_RTSP_PORT = 8555
DEFAULT_RTSP_MOUNT = "/ds"

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
cam_names = list(SOURCES.keys())
cam_frame_count = defaultdict(int)
total_counts = {c: defaultdict(int) for c in cam_names}
batch_count = 0
last_report_t = time.time()
rows_since_flush = 0
csv_file = None
csv_writer = None
g_main_loop = None


# ── Probe on nvinfer src pad — same as headless ─────────────────
def infer_src_pad_buffer_probe(pad, info, u_data):
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
        cam_name = cam_names[src_id] if src_id < len(cam_names) else f"src-{src_id}"
        cam_frame_count[cam_name] += 1
        this_frame = cam_frame_count[cam_name]
        frame_cls = defaultdict(lambda: {"count": 0, "max_conf": 0.0})

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            cid = obj_meta.class_id
            conf = float(obj_meta.confidence)
            cname = (COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES)
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
        if message.src and message.src.get_name() == "pipeline":
            old, new, _ = message.parse_state_changed()
            print(f"[BUS] pipeline: {old.value_nick} -> {new.value_nick}")
    return True


# ── Pipeline builders ────────────────────────────────────────────
def make(factory, name):
    e = Gst.ElementFactory.make(factory, name)
    if e is None:
        raise RuntimeError(f"Cannot create element '{factory}' (name={name})")
    return e


def build_sources(pipeline, streammux):
    """Add N nvurisrcbin → streammux sink pads."""
    for i, (cam_name, url) in enumerate(SOURCES.items()):
        src = Gst.ElementFactory.make("nvurisrcbin", f"src-{i}")
        if src is None:
            print(f"[WARN] nvurisrcbin not available, fallback uridecodebin for {cam_name}")
            src = make("uridecodebin", f"src-{i}")
        src.set_property("uri", url)
        try:
            src.set_property("rtsp-reconnect-interval", 5)
            src.set_property("rtsp-reconnect-attempts", -1)
            src.set_property("latency", 200)
            src.set_property("select-rtp-protocol", 4)
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


def attach_probe(nvinfer):
    nvinfer_src = nvinfer.get_static_pad("src")
    nvinfer_src.add_probe(Gst.PadProbeType.BUFFER,
                          infer_src_pad_buffer_probe, 0)


def build_headless_pipeline():
    """
    Original behavior: 5 sources → mux → nvinfer → fakesink (probe reads metadata).
    """
    Gst.init(None)
    pipeline = Gst.Pipeline.new("pipeline")

    streammux = make("nvstreammux", "streammux")
    streammux.set_property("width", MUXER_W)
    streammux.set_property("height", MUXER_H)
    streammux.set_property("batch-size", len(SOURCES))
    streammux.set_property("batched-push-timeout", BATCHED_PUSH_TIMEOUT_US)
    streammux.set_property("live-source", 1)
    pipeline.add(streammux)

    build_sources(pipeline, streammux)

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

    attach_probe(nvinfer)
    return pipeline


def build_show_pipeline(mode, opts):
    """
    Visual pipeline:
        sources → mux → nvinfer → tiler → nvvidconv → nvdsosd → ...
            ... → nvvidconv → nvv4l2h264enc → h264parse → (mux/sink)

    `mode` is one of "file", "rtsp", "screen".
    """
    Gst.init(None)
    pipeline = Gst.Pipeline.new("pipeline")

    streammux = make("nvstreammux", "streammux")
    streammux.set_property("width", MUXER_W)
    streammux.set_property("height", MUXER_H)
    streammux.set_property("batch-size", len(SOURCES))
    streammux.set_property("batched-push-timeout", BATCHED_PUSH_TIMEOUT_US)
    streammux.set_property("live-source", 1)
    pipeline.add(streammux)

    build_sources(pipeline, streammux)

    nvinfer = make("nvinfer", "nvinfer")
    nvinfer.set_property("config-file-path", INFER_CONFIG)
    pipeline.add(nvinfer)

    tiler = make("nvmultistreamtiler", "tiler")
    tiler.set_property("rows", opts["tile_rows"])
    tiler.set_property("columns", opts["tile_cols"])
    tiler.set_property("width", opts["tile_w"])
    tiler.set_property("height", opts["tile_h"])
    pipeline.add(tiler)

    nvvidconv = make("nvvideoconvert", "nvvidconv0")
    pipeline.add(nvvidconv)

    osd = make("nvdsosd", "osd")
    pipeline.add(osd)

    # Link inference + visual chain
    if not streammux.link(nvinfer):
        raise RuntimeError("link streammux -> nvinfer failed")
    if not nvinfer.link(tiler):
        raise RuntimeError("link nvinfer -> tiler failed")
    if not tiler.link(nvvidconv):
        raise RuntimeError("link tiler -> nvvidconv failed")
    if not nvvidconv.link(osd):
        raise RuntimeError("link nvvidconv -> osd failed")

    attach_probe(nvinfer)

    # ── Branch on mode ──
    if mode == "screen":
        sink = make("nveglglessink", "sink")
        sink.set_property("sync", 0)
        pipeline.add(sink)
        if not osd.link(sink):
            raise RuntimeError("link osd -> nveglglessink failed (no display?)")
        return pipeline

    # file / rtsp both need encoding
    nvvidconv2 = make("nvvideoconvert", "nvvidconv_out")
    pipeline.add(nvvidconv2)
    if not osd.link(nvvidconv2):
        raise RuntimeError("link osd -> nvvidconv_out failed")

    caps_filter = make("capsfilter", "caps_nv12")
    caps_filter.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"))
    pipeline.add(caps_filter)
    if not nvvidconv2.link(caps_filter):
        raise RuntimeError("link nvvidconv_out -> caps_filter failed")

    encoder = make("nvv4l2h264enc", "encoder")
    encoder.set_property("bitrate", opts["bitrate_kbps"] * 1000)  # bps
    encoder.set_property("insert-sps-pps", 1)
    encoder.set_property("iframeinterval", 30)
    pipeline.add(encoder)
    if not caps_filter.link(encoder):
        raise RuntimeError("link caps_filter -> encoder failed")

    h264parse = make("h264parse", "h264parse")
    pipeline.add(h264parse)
    if not encoder.link(h264parse):
        raise RuntimeError("link encoder -> h264parse failed")

    if mode == "file":
        muxer = make("qtmux", "muxer")
        pipeline.add(muxer)
        if not h264parse.link(muxer):
            raise RuntimeError("link h264parse -> qtmux failed")
        sink = make("filesink", "sink")
        sink.set_property("location", opts["output_file"])
        sink.set_property("sync", 0)
        pipeline.add(sink)
        if not muxer.link(sink):
            raise RuntimeError("link qtmux -> filesink failed")
        print(f"[INFO] Writing tiled output to: {opts['output_file']}")
        return pipeline

    if mode == "rtsp":
        # Lazy-load GstRtspServer (gir1.2-gst-rtsp-server-1.0 package)
        try:
            gi.require_version("GstRtspServer", "1.0")
            from gi.repository import GstRtspServer
        except (ValueError, ImportError) as e:
            raise RuntimeError(
                "RTSP mode requires GstRtspServer Python binding. "
                "Install with:\n"
                "  sudo apt install -y gir1.2-gst-rtsp-server-1.0 "
                "libgstrtspserver-1.0-0\n"
                f"(import error: {e})"
            )
        rtppay = make("rtph264pay", "rtppay")
        rtppay.set_property("pt", 96)
        pipeline.add(rtppay)
        if not h264parse.link(rtppay):
            raise RuntimeError("link h264parse -> rtppay failed")
        sink = make("udpsink", "sink")
        sink.set_property("host", "127.0.0.1")
        sink.set_property("port", opts["rtsp_port"] + 1)  # internal pipe port
        sink.set_property("async", False)
        sink.set_property("sync", False)
        pipeline.add(sink)
        if not rtppay.link(sink):
            raise RuntimeError("link rtppay -> udpsink failed")

        # Create RTSP server that wraps the UDP feed
        server = GstRtspServer.RTSPServer.new()
        server.set_service(str(opts["rtsp_port"]))
        mounts = server.get_mount_points()
        factory = GstRtspServer.RTSPMediaFactory.new()
        factory.set_launch(
            f"( udpsrc port={opts['rtsp_port'] + 1} buffer-size=524288 "
            f"caps=\"application/x-rtp,media=video,clock-rate=90000,"
            f"encoding-name=H264,payload=96\" ! rtph264depay ! h264parse ! "
            f"rtph264pay name=pay0 pt=96 config-interval=1 )"
        )
        factory.set_shared(True)
        mounts.add_factory(opts["rtsp_mount"], factory)
        server.attach(None)
        print(f"[INFO] RTSP server: rtsp://<this-host>:{opts['rtsp_port']}{opts['rtsp_mount']}")
        # Keep a reference on the pipeline so it isn't GC'd
        pipeline._rtsp_server = server
        return pipeline

    raise ValueError(f"unknown --show mode: {mode}")


# ── Main ────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="5-cam DeepStream YOLOv8 with optional visual tile")
    p.add_argument("--show", choices=["none", "file", "rtsp", "screen"],
                   default="none",
                   help="Visualization mode (default: none = headless)")
    p.add_argument("--show-output", default=DEFAULT_OUTPUT_FILE,
                   help=f"Tiled .mp4 output path for --show file (default: {DEFAULT_OUTPUT_FILE})")
    p.add_argument("--tile-rows", type=int, default=DEFAULT_TILE_ROWS)
    p.add_argument("--tile-cols", type=int, default=DEFAULT_TILE_COLS)
    p.add_argument("--tile-w", type=int, default=DEFAULT_TILE_W)
    p.add_argument("--tile-h", type=int, default=DEFAULT_TILE_H)
    p.add_argument("--bitrate-kbps", type=int, default=4000,
                   help="H.264 bitrate when --show file/rtsp (default 4000)")
    p.add_argument("--rtsp-port", type=int, default=DEFAULT_RTSP_PORT)
    p.add_argument("--rtsp-mount", default=DEFAULT_RTSP_MOUNT)
    p.add_argument("--csv", default=LOG_CSV,
                   help=f"CSV output path (default: {LOG_CSV})")
    args = p.parse_args()

    # Open CSV (always — even in --show mode)
    global csv_file, csv_writer, g_main_loop, last_report_t
    csv_file = open(args.csv, "w", newline="", buffering=1)
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp", "cam", "cam_frame", "class",
        "confidence", "count_this_frame"
    ])

    if args.show == "none":
        pipeline = build_headless_pipeline()
        mode_label = "HEADLESS (CSV log only)"
    else:
        opts = dict(
            tile_rows=args.tile_rows, tile_cols=args.tile_cols,
            tile_w=args.tile_w, tile_h=args.tile_h,
            bitrate_kbps=args.bitrate_kbps,
            output_file=args.show_output,
            rtsp_port=args.rtsp_port, rtsp_mount=args.rtsp_mount,
        )
        pipeline = build_show_pipeline(args.show, opts)
        mode_label = f"SHOW={args.show}  tile={args.tile_rows}x{args.tile_cols}"

    g_main_loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, g_main_loop)

    def _sig(*_):
        print("\n[SIG] stopping...")
        g_main_loop.quit()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    print(f"[INFO] Mode    : {mode_label}")
    print(f"[INFO] Sources : {list(SOURCES.keys())}")
    print(f"[INFO] Config  : {INFER_CONFIG}")
    print(f"[INFO] CSV     : {args.csv}")
    print(f"[INFO] Push timeout: {BATCHED_PUSH_TIMEOUT_US}us")
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
        # Send EOS first so qtmux finalizes the .mp4 cleanly
        if args.show == "file":
            print("[INFO] Sending EOS for clean .mp4 finalize...")
            pipeline.send_event(Gst.Event.new_eos())
            # Wait briefly for EOS to propagate
            bus.timed_pop_filtered(2 * Gst.SECOND,
                                   Gst.MessageType.EOS | Gst.MessageType.ERROR)
        print_report()
        pipeline.set_state(Gst.State.NULL)
        try:
            csv_file.flush()
            csv_file.close()
        except Exception:
            pass
        print(f"[INFO] CSV saved -> {args.csv}")
        if args.show == "file":
            print(f"[INFO] Tiled video -> {args.show_output}")


if __name__ == "__main__":
    main()
