"""
ds_view_v5_viewer.py
====================
Live viewer for the 5-cam YOLO26 + ShuffleNet brand pipeline.

Output: native window on the Nano's HDMI (nveglglessink + nvegltransform).
Switch cameras with keyboard:
   1..5 or a..e -> show that camera full-screen
   0 or g       -> show all 5 in a tiled grid
   q            -> quit

Pipeline:
   5x nvurisrcbin -> nvstreammux(5x640x640, NV12)
        -> nvvideoconvert -> caps(NVMM,RGBA)
        -> nvinfer (YOLO26 end2end)        -> probe attaches:
                                              - NvDsObjectMeta per detection
                                              - DisplayMeta with cumulative counts
                                              - brand_id for cars (ShuffleNet)
        -> nvmultistreamtiler(1x1, show-source=current)
        -> nvdsosd
        -> nvegltransform -> nveglglessink
"""
import csv
import ctypes
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
from collections import defaultdict, deque

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
cuda.init()

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds


# ── Config ──────────────────────────────────────────────────────────
SOURCES = {
    "cam-a": "rtsp://172.16.30.111:8554/vdo1",
    "cam-b": "rtsp://172.16.30.111:8554/vdo2",
    "cam-c": "rtsp://172.16.30.111:8554/vdo3",
    "cam-d": "rtsp://172.16.30.111:8554/vdo4",
    "cam-e": "rtsp://172.16.30.111:8554/vdo5",
}

INFER_CONFIG     = "config_infer_yolo26.txt"
BRAND_ENGINE     = "shufflenet_brand.engine"
LOG_CSV          = "detect_log_v5_viewer.csv"

CONF_THRESHOLD       = 0.25
CAR_CLASS_ID         = 2
BRAND_MAX_BATCH      = 16

MUXER_W, MUXER_H         = 640, 640
TILER_W, TILER_H         = 1280, 720
BATCHED_PUSH_TIMEOUT_US  = 40_000


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


# ── Brand classifier (from v4, same pycuda+TRT with own context) ────
class BrandClassifier:
    IMG = 224
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def __init__(self, engine_path, max_batch=16):
        self.max_batch = max_batch
        self.cuda_ctx = cuda.Device(0).make_context()
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f:
                data = f.read()
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(data)
            self.ctx = self.engine.create_execution_context()
            self.in_idx  = self.engine.get_binding_index("input")
            self.out_idx = self.engine.get_binding_index("output")
            self.in_dtype  = trt.nptype(self.engine.get_binding_dtype(self.in_idx))
            self.out_dtype = trt.nptype(self.engine.get_binding_dtype(self.out_idx))
            self.h_in  = cuda.pagelocked_empty(max_batch*3*self.IMG*self.IMG, self.in_dtype)
            self.h_out = cuda.pagelocked_empty(max_batch*26, self.out_dtype)
            self.d_in  = cuda.mem_alloc(self.h_in.nbytes)
            self.d_out = cuda.mem_alloc(self.h_out.nbytes)
            self.stream = cuda.Stream()
        finally:
            self.cuda_ctx.pop()

    def classify(self, crops):
        n = len(crops)
        if n == 0: return []
        if n > self.max_batch: n = self.max_batch
        self.cuda_ctx.push()
        try:
            batch = np.empty((n, 3, self.IMG, self.IMG), dtype=np.float32)
            for i in range(n):
                img = cv2.resize(crops[i], (self.IMG, self.IMG), interpolation=cv2.INTER_LINEAR)
                arr = img.astype(np.float32) * (1.0/255.0)
                arr = arr.transpose(2, 0, 1)
                batch[i] = arr
            batch = (batch - self.MEAN) / self.STD
            batch_fp16 = batch.astype(self.in_dtype)
            flat = batch_fp16.ravel()
            self.h_in[:flat.size] = flat
            self.ctx.set_binding_shape(self.in_idx, (n, 3, self.IMG, self.IMG))
            cuda.memcpy_htod_async(self.d_in, self.h_in[:flat.size], self.stream)
            self.ctx.execute_async_v2(bindings=[int(self.d_in), int(self.d_out)],
                                      stream_handle=self.stream.handle)
            out_n = n * 26
            cuda.memcpy_dtoh_async(self.h_out[:out_n], self.d_out, self.stream)
            self.stream.synchronize()
            logits = self.h_out[:out_n].reshape(n, 26).astype(np.float32)
            return logits.argmax(axis=1).astype(np.int32).tolist()
        finally:
            self.cuda_ctx.pop()


# ── State ───────────────────────────────────────────────────────────
cam_names         = list(SOURCES.keys())
cam_frame_count   = defaultdict(int)
FPS_WIN           = 30           # sliding window size for FPS (frames)
cam_ts_window     = defaultdict(lambda: deque(maxlen=FPS_WIN))
total_counts      = defaultdict(int)        # global class -> count
batch_count       = 0
csv_file = csv_writer = None
g_main_loop = None
classifier = None
tiler_elem = None
current_show = 0     # -1 = grid, 0..4 = single cam

# scale from muxer (640x640) to tiler-source rect.
# When show-source=N (single cam), tiler outputs the cam at TILER_W x TILER_H.
SCALE_X = TILER_W / MUXER_W
SCALE_Y = TILER_H / MUXER_H

# Per-class color palette for bbox (RGB 0..1 -> set via .red/.green/.blue)
COLORS = [
    (1.0, 0.4, 0.4),  # red
    (0.4, 1.0, 0.4),  # green
    (0.4, 0.7, 1.0),  # blue
    (1.0, 1.0, 0.4),  # yellow
    (1.0, 0.6, 1.0),  # magenta
    (0.4, 1.0, 1.0),  # cyan
]


def layer_to_numpy(layer):
    dims = layer.dims
    shape = tuple(int(dims.d[i]) for i in range(dims.numDims))
    n_elem = 1
    for s in shape: n_elem *= s
    if n_elem == 0:
        return np.empty(shape, dtype=np.float32)
    cptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
    return np.ctypeslib.as_array(cptr, shape=(n_elem,)).copy().reshape(shape)


def get_frame_rgb(gst_buffer, batch_id):
    try:
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), batch_id)
        return n_frame[:, :, :3].copy()
    except Exception:
        return None


def crop_clip(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    x1 = max(0, min(W - 1, int(x1)))
    x2 = max(0, min(W,     int(x2)))
    y1 = max(0, min(H - 1, int(y1)))
    y2 = max(0, min(H,     int(y2)))
    if x2 <= x1 + 1 or y2 <= y1 + 1: return None
    return img[y1:y2, x1:x2]


def set_color(c, rgb):
    c.red, c.green, c.blue, c.alpha = rgb[0], rgb[1], rgb[2], 1.0


def attach_obj_meta(batch_meta, frame_meta, x1, y1, x2, y2, label, cid):
    obj = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
    obj.unique_component_id = 1
    obj.class_id = cid
    obj.confidence = 0.0
    obj.object_id  = 0xffffffffffffffff  # unsigned -1 for "untracked"

    # Coordinates in tiler-output space
    rx = float(x1) * SCALE_X
    ry = float(y1) * SCALE_Y
    rw = max(1.0, float(x2 - x1) * SCALE_X)
    rh = max(1.0, float(y2 - y1) * SCALE_Y)

    rp = obj.rect_params
    rp.left, rp.top, rp.width, rp.height = rx, ry, rw, rh
    rp.border_width = 2
    set_color(rp.border_color, COLORS[cid % len(COLORS)])
    rp.has_bg_color = 0

    # detector_bbox_info too (some DS components consult this)
    db = obj.detector_bbox_info.org_bbox_coords
    db.left, db.top, db.width, db.height = rx, ry, rw, rh

    tp = obj.text_params
    tp.display_text = label
    tp.x_offset = max(0, int(rx))
    tp.y_offset = max(0, int(ry) - 14)
    tp.font_params.font_name = "Serif"
    tp.font_params.font_size = 11
    tp.font_params.font_color.red = 1
    tp.font_params.font_color.green = 1
    tp.font_params.font_color.blue = 1
    tp.font_params.font_color.alpha = 1
    tp.set_bg_clr = 1
    tp.text_bg_clr.red = 0
    tp.text_bg_clr.green = 0
    tp.text_bg_clr.blue = 0
    tp.text_bg_clr.alpha = 0.6

    pyds.nvds_add_obj_meta_to_frame(frame_meta, obj, None)


def attach_overlay_text(batch_meta, frame_meta, lines):
    """Top-left summary text. `lines` is a list of strings; we use up to MAX_DISPLAY_TEXT (16)."""
    if not lines: return
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    n = min(len(lines), 16)
    display_meta.num_labels = n
    for i in range(n):
        tp = display_meta.text_params[i]
        tp.display_text = lines[i]
        tp.x_offset = 10
        tp.y_offset = 8 + i * 18
        tp.font_params.font_name = "Serif"
        tp.font_params.font_size = 12
        tp.font_params.font_color.red = 1
        tp.font_params.font_color.green = 1
        tp.font_params.font_color.blue = 1
        tp.font_params.font_color.alpha = 1
        tp.set_bg_clr = 1
        tp.text_bg_clr.red = 0
        tp.text_bg_clr.green = 0
        tp.text_bg_clr.blue = 0
        tp.text_bg_clr.alpha = 0.55
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)


def infer_src_pad_buffer_probe(pad, info, u_data):
    global batch_count
    gst_buffer = info.get_buffer()
    if not gst_buffer: return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None: return Gst.PadProbeReturn.OK

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    l_frame = batch_meta.frame_meta_list

    # Per-frame: collect car crops to batch-classify after the loop
    pending_cars = []   # list of (frame_meta, x1, y1, x2, y2, score, crop_np)

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        src_id = frame_meta.source_id
        cam_name = cam_names[src_id] if src_id < len(cam_names) else f"src-{src_id}"
        cam_frame_count[cam_name] += 1
        cam_ts_window[cam_name].append(time.monotonic())

        # Find tensor meta
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
            try: l_user = l_user.next
            except StopIteration: break

        if tensor_meta is None or tensor_meta.num_output_layers < 1:
            try: l_frame = l_frame.next
            except StopIteration: break
            continue

        layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
        arr = layer_to_numpy(layer)
        if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
        keep = arr[:, 4] > CONF_THRESHOLD
        dets = arr[keep]

        # Pull frame surface ONCE if any car (for cropping)
        frame_rgb = None
        if any(int(d[5]) == CAR_CLASS_ID for d in dets):
            frame_rgb = get_frame_rgb(gst_buffer, frame_meta.batch_id)

        for row in dets:
            x1, y1, x2, y2, score, cls = row
            cid = int(cls)
            cname = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else f"class-{cid}"
            total_counts[cname] += 1

            label = f"{cname} {score:.2f}"
            if cid == CAR_CLASS_ID and frame_rgb is not None:
                crop = crop_clip(frame_rgb, x1, y1, x2, y2)
                if crop is not None and crop.size > 0:
                    pending_cars.append((frame_meta, batch_meta, x1, y1, x2, y2, score, label, crop))
                    continue  # bbox+label will be attached after brand classification
            # Non-car (or car w/o crop) -> attach directly
            attach_obj_meta(batch_meta, frame_meta, x1, y1, x2, y2, label, cid)

            csv_writer.writerow([ts, cam_name, cam_frame_count[cam_name], cname,
                                 f"{score:.3f}", ""])

        # Per-frame summary overlay (only shows on rendered frame; tiler picks one)
        top = sorted(total_counts.items(), key=lambda x: -x[1])[:5]
        cam_display = (cam_names[current_show] if 0 <= current_show < len(cam_names)
                       else "ALL")
        # Per-cam FPS from sliding window
        fps_strs = []
        total_fps = 0.0
        for cn in cam_names:
            dq = cam_ts_window[cn]
            if len(dq) >= 2:
                dt = dq[-1] - dq[0]
                if dt > 0:
                    f = (len(dq) - 1) / dt
                    total_fps += f
                    fps_strs.append(f"{cn[-1]}={f:4.1f}")
                else:
                    fps_strs.append(f"{cn[-1]}= -- ")
            else:
                fps_strs.append(f"{cn[-1]}= -- ")
        lines = [
            f"[{cam_display}]  press 1..5/a..e to switch  0=all  q=quit",
            "FPS:   " + "  ".join(fps_strs) + f"   total={total_fps:5.1f}",
            "totals: " + "  ".join(f"{c}={k}" for c, k in top),
        ]
        attach_overlay_text(batch_meta, frame_meta, lines)

        try: l_frame = l_frame.next
        except StopIteration: break

    # Classify pending cars in chunks of BRAND_MAX_BATCH
    if pending_cars:
        i = 0
        while i < len(pending_cars):
            chunk = pending_cars[i:i + BRAND_MAX_BATCH]
            crops = [c[-1] for c in chunk]
            try:
                brands = classifier.classify(crops)
            except Exception as e:
                print(f"[WARN] brand classify failed: {e}")
                brands = [-1] * len(chunk)
            for (fm, bm, x1, y1, x2, y2, score, base_label, _), b in zip(chunk, brands):
                lbl = f"{base_label}  brand={b}"
                attach_obj_meta(bm, fm, x1, y1, x2, y2, lbl, CAR_CLASS_ID)
                # CSV row for car w/ brand
                src_id = fm.source_id
                cam_name = cam_names[src_id] if src_id < len(cam_names) else f"src-{src_id}"
                csv_writer.writerow([ts, cam_name, cam_frame_count[cam_name], "car",
                                     f"{score:.3f}", b])
            i += BRAND_MAX_BATCH

    batch_count += 1
    return Gst.PadProbeReturn.OK


# ── Keyboard input (background thread) ──────────────────────────────
def kbd_loop(stop_event):
    if not sys.stdin.isatty():
        print("[KBD] stdin not a TTY — keyboard switching disabled.")
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not stop_event.is_set():
            r, _, _ = select.select([fd], [], [], 0.2)
            if not r: continue
            ch = sys.stdin.read(1)
            if not ch: continue
            handle_key(ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def handle_key(ch):
    global current_show
    c = ch.lower()
    if c in "12345":
        idx = int(c) - 1
        _set_show(idx)
    elif c in "abcde":
        idx = "abcde".index(c)
        _set_show(idx)
    elif c in "0g":
        _set_show(-1)
    elif c == "q":
        print("\n[KBD] q -> quit")
        if g_main_loop is not None: g_main_loop.quit()
    elif c == "?":
        print("[KBD] keys: 1-5 / a-e = switch  0 or g = grid (all)  q = quit")


def _set_show(idx):
    global current_show
    current_show = idx
    if tiler_elem is None: return
    if idx == -1:
        tiler_elem.set_property("rows", 2)
        tiler_elem.set_property("columns", 3)
        tiler_elem.set_property("show-source", -1)
        print("[KBD] showing all (2x3 grid)")
    else:
        tiler_elem.set_property("rows", 1)
        tiler_elem.set_property("columns", 1)
        tiler_elem.set_property("show-source", idx)
        print(f"[KBD] showing {cam_names[idx]} (source {idx})")


# ── Bus / pipeline ──────────────────────────────────────────────────
def on_bus_message(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("[BUS] EOS"); loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        src = message.src.get_name() if message.src else "?"
        print(f"[BUS][ERR ] {src}: {err.message}")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, dbg = message.parse_warning()
        src = message.src.get_name() if message.src else "?"
        print(f"[BUS][WARN] {src}: {err.message}")
    return True


def make(factory, name):
    e = Gst.ElementFactory.make(factory, name)
    if e is None: raise RuntimeError(f"create '{factory}' failed")
    return e


def build_pipeline():
    Gst.init(None)
    pipe = Gst.Pipeline.new("pipeline")

    streammux = make("nvstreammux", "streammux")
    streammux.set_property("width", MUXER_W)
    streammux.set_property("height", MUXER_H)
    streammux.set_property("batch-size", len(SOURCES))
    streammux.set_property("batched-push-timeout", BATCHED_PUSH_TIMEOUT_US)
    streammux.set_property("live-source", 1)
    pipe.add(streammux)

    for i, (cam_name, url) in enumerate(SOURCES.items()):
        src = Gst.ElementFactory.make("nvurisrcbin", f"src-{i}")
        if src is None:
            src = make("uridecodebin", f"src-{i}")
        src.set_property("uri", url)
        try:
            src.set_property("rtsp-reconnect-interval", 5)
            src.set_property("rtsp-reconnect-attempts", -1)
            src.set_property("latency", 200)
            src.set_property("select-rtp-protocol", 4)
        except Exception: pass

        def on_pad_added(_, pad, mux=streammux, idx=i, name=cam_name):
            sink_pad = mux.get_request_pad(f"sink_{idx}")
            if sink_pad and not sink_pad.is_linked():
                pad.link(sink_pad)
                print(f"[OK] linked {name} -> streammux sink_{idx}")
        src.connect("pad-added", on_pad_added)
        pipe.add(src)

    # Force RGBA so probe can read pixels for brand crop
    nvvconv = make("nvvideoconvert", "nvvconv-pre")
    capsf   = make("capsfilter", "capsf-pre")
    capsf.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    pipe.add(nvvconv); pipe.add(capsf)

    nvinfer = make("nvinfer", "nvinfer")
    nvinfer.set_property("config-file-path", INFER_CONFIG)
    pipe.add(nvinfer)

    tiler = make("nvmultistreamtiler", "tiler")
    tiler.set_property("rows", 1)
    tiler.set_property("columns", 1)
    tiler.set_property("width", TILER_W)
    tiler.set_property("height", TILER_H)
    tiler.set_property("show-source", 0)
    pipe.add(tiler)

    osd = make("nvdsosd", "osd")
    pipe.add(osd)

    transform = make("nvegltransform", "egltransform")
    sink = make("nveglglessink", "sink")
    sink.set_property("sync", 0)
    pipe.add(transform); pipe.add(sink)

    if not streammux.link(nvvconv):  raise RuntimeError("link mux->nvvconv")
    if not nvvconv.link(capsf):      raise RuntimeError("link nvvconv->caps")
    if not capsf.link(nvinfer):      raise RuntimeError("link caps->nvinfer")
    if not nvinfer.link(tiler):      raise RuntimeError("link nvinfer->tiler")
    if not tiler.link(osd):          raise RuntimeError("link tiler->osd")
    if not osd.link(transform):      raise RuntimeError("link osd->transform")
    if not transform.link(sink):     raise RuntimeError("link transform->sink")

    nvinfer.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER,
                                            infer_src_pad_buffer_probe, 0)
    return pipe, tiler


def main():
    global csv_file, csv_writer, g_main_loop, classifier, tiler_elem

    print(f"[INFO] Loading brand classifier: {BRAND_ENGINE}")
    classifier = BrandClassifier(BRAND_ENGINE, max_batch=BRAND_MAX_BATCH)
    print("[INFO] classifier ready")

    csv_file = open(LOG_CSV, "w", newline="", buffering=1)
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp","cam","cam_frame","class","confidence","brand_id"])

    pipe, tiler = build_pipeline()
    tiler_elem = tiler

    g_main_loop = GLib.MainLoop()
    bus = pipe.get_bus(); bus.add_signal_watch()
    bus.connect("message", on_bus_message, g_main_loop)

    stop_evt = threading.Event()
    kbd_th = threading.Thread(target=kbd_loop, args=(stop_evt,), daemon=True)
    kbd_th.start()

    def _sig(*_):
        print("\n[SIG] stopping..."); g_main_loop.quit()
    signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)

    print(f"[INFO] v5 viewer — {len(SOURCES)} cameras")
    print(f"[INFO] keys: 1-5 or a-e to switch cam | 0 or g for grid | q to quit | ? help")
    print(f"[INFO] CSV: {LOG_CSV}\n")

    if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("[FATAL] Unable to start pipeline"); sys.exit(1)

    try:
        g_main_loop.run()
    finally:
        print("\n[INFO] Stopping pipeline...")
        stop_evt.set()
        pipe.set_state(Gst.State.NULL)
        try: csv_file.flush(); csv_file.close()
        except Exception: pass
        print(f"[INFO] CSV saved -> {LOG_CSV}")


if __name__ == "__main__":
    main()
