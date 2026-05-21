"""
ds_view_v6_track.py
===================
v5 viewer + nvinfer interval=1 + IOU tracker + brand cache by track_id.

What changed vs v5:
  - nvinfer config: interval=1 (run YOLO every 2nd frame; tracker fills gaps)
  - new nvtracker between nvinfer and tiler (IOU, light)
  - probe1 (nvinfer src): attach NvDsObjectMeta with bbox + simple label "cls 0.92"
  - probe2 (tracker src): for each car obj_meta, brand_cache[track_id] -> classify
                          if missing; update obj.text_params with "car 0.92 brand=21"
  - Brand classifier called only ONCE per track_id (huge CPU saving on repeated cars)
  - CSV log moved to probe2 so we get the cached brand on every tracked frame
"""
import csv
import ctypes
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

INFER_CONFIG  = "config_infer_yolo26.txt"
BRAND_ENGINE  = "shufflenet_brand.engine"
TRACKER_CFG   = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml"
TRACKER_LIB   = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
LOG_CSV       = "detect_log_v6_track.csv"

CONF_THRESHOLD       = 0.25
CAR_CLASS_ID         = 2
BRAND_MAX_BATCH      = 16

MUXER_W, MUXER_H         = 320, 320    # will switch to 320 once smaller engine is ready
TILER_W, TILER_H         = 1280, 720
BATCHED_PUSH_TIMEOUT_US  = 40_000

FPS_WIN = 30

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

COLORS = [
    (1.0, 0.4, 0.4), (0.4, 1.0, 0.4), (0.4, 0.7, 1.0),
    (1.0, 1.0, 0.4), (1.0, 0.6, 1.0), (0.4, 1.0, 1.0),
]


# ── Brand classifier (same as v5) ───────────────────────────────────
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
                img = cv2.resize(crops[i], (self.IMG, self.IMG),
                                 interpolation=cv2.INTER_LINEAR)
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
cam_ts_window     = defaultdict(lambda: deque(maxlen=FPS_WIN))
total_counts      = defaultdict(int)
brand_cache       = {}   # track_id -> brand_id
csv_file = csv_writer = None
g_main_loop = None
classifier = None
tiler_elem = None
current_show = 0


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
    """
    Attach object meta in MUXER (source-frame) coordinates.
    Tiler/OSD will rescale to display correctly because we DON'T pre-scale.
    """
    obj = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
    obj.unique_component_id = 1
    obj.class_id = cid
    obj.confidence = 0.0
    obj.object_id = 0xffffffffffffffff   # tracker will assign real id

    rx = float(x1); ry = float(y1)
    rw = max(1.0, float(x2 - x1))
    rh = max(1.0, float(y2 - y1))

    rp = obj.rect_params
    rp.left, rp.top, rp.width, rp.height = rx, ry, rw, rh
    rp.border_width = 2
    set_color(rp.border_color, COLORS[cid % len(COLORS)])
    rp.has_bg_color = 0

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


# ── probe 1: nvinfer src pad ────────────────────────────────────────
def probe1_after_infer(pad, info, u_data):
    """
    Decode YOLO tensor and attach NvDsObjectMeta with simple bbox + label.
    Tracker downstream will assign track_ids and propagate on skipped frames.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer: return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None: return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

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

        if tensor_meta is not None and tensor_meta.num_output_layers >= 1:
            layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
            arr = layer_to_numpy(layer)
            if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
            keep = arr[:, 4] > CONF_THRESHOLD
            dets = arr[keep]
            for row in dets:
                x1, y1, x2, y2, score, cls = row
                cid = int(cls)
                cname = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else f"cls-{cid}"
                attach_obj_meta(batch_meta, frame_meta,
                                x1, y1, x2, y2,
                                f"{cname} {score:.2f}", cid)
                total_counts[cname] += 1

        try: l_frame = l_frame.next
        except StopIteration: break

    return Gst.PadProbeReturn.OK


# ── probe 2: tracker src pad ────────────────────────────────────────
def probe2_after_tracker(pad, info, u_data):
    """
    Walk tracked objects, classify brands for new car tracks, update labels,
    log to CSV, attach overlay summary text.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer: return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None: return Gst.PadProbeReturn.OK

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    l_frame = batch_meta.frame_meta_list

    # collect pending car classifications across the whole batch
    pending = []   # (obj_ref, track_id, base_label, score, crop)

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        src_id = frame_meta.source_id
        cam_name = cam_names[src_id] if src_id < len(cam_names) else f"src-{src_id}"
        cam_frame_count[cam_name] += 1
        cam_ts_window[cam_name].append(time.monotonic())

        # Pull frame once if any cars in this frame need classification
        frame_rgb = None

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            cid = obj.class_id
            tid = obj.object_id
            score = obj.confidence  # 0 if we set 0; ok
            cname = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else f"cls-{cid}"

            # Default keep the label probe1 set; only update for cars
            if cid == CAR_CLASS_ID:
                bid = brand_cache.get(tid)
                if bid is None and tid != 0xffffffffffffffff:
                    if frame_rgb is None:
                        frame_rgb = get_frame_rgb(gst_buffer, frame_meta.batch_id)
                    if frame_rgb is not None:
                        rp = obj.rect_params
                        crop = crop_clip(frame_rgb,
                                         rp.left, rp.top,
                                         rp.left + rp.width, rp.top + rp.height)
                        if crop is not None and crop.size > 0:
                            pending.append((obj, tid, cname, score, crop))
                if bid is not None:
                    obj.text_params.display_text = f"{cname} brand={bid}  id={tid & 0xffffff}"
                    csv_writer.writerow([ts, cam_name, cam_frame_count[cam_name],
                                         cname, f"{score:.3f}", tid, bid])
                else:
                    obj.text_params.display_text = f"{cname}  id={tid & 0xffffff}"
                    csv_writer.writerow([ts, cam_name, cam_frame_count[cam_name],
                                         cname, f"{score:.3f}", tid, ""])
            # non-cars: nothing more; probe1 set label already

            try: l_obj = l_obj.next
            except StopIteration: break

        # overlay
        top = sorted(total_counts.items(), key=lambda x: -x[1])[:5]
        cam_display = (cam_names[current_show] if 0 <= current_show < len(cam_names)
                       else "ALL")
        fps_strs = []
        total_fps = 0.0
        for cn in cam_names:
            dq = cam_ts_window[cn]
            if len(dq) >= 2 and (dq[-1] - dq[0]) > 0:
                f = (len(dq) - 1) / (dq[-1] - dq[0])
                total_fps += f
                fps_strs.append(f"{cn[-1]}={f:4.1f}")
            else:
                fps_strs.append(f"{cn[-1]}= -- ")
        lines = [
            f"[{cam_display}]  1..5/a..e switch  0=all  q=quit   tracks={len(brand_cache)}",
            "FPS:   " + "  ".join(fps_strs) + f"   total={total_fps:5.1f}",
            "totals: " + "  ".join(f"{c}={k}" for c, k in top),
        ]
        attach_overlay_text(batch_meta, frame_meta, lines)

        try: l_frame = l_frame.next
        except StopIteration: break

    # Batch-classify all new car crops in this batch
    if pending:
        i = 0
        while i < len(pending):
            chunk = pending[i:i + BRAND_MAX_BATCH]
            crops = [c[-1] for c in chunk]
            try:
                brands = classifier.classify(crops)
            except Exception as e:
                print(f"[WARN] brand classify failed: {e}")
                brands = [-1] * len(chunk)
            for (obj, tid, cname, score, _), b in zip(chunk, brands):
                if b >= 0:
                    brand_cache[tid] = int(b)
                    obj.text_params.display_text = f"{cname} brand={b}  id={tid & 0xffffff}"
            i += BRAND_MAX_BATCH

    return Gst.PadProbeReturn.OK


# ── Keyboard ────────────────────────────────────────────────────────
def kbd_loop(stop_event):
    if not sys.stdin.isatty():
        print("[KBD] stdin not a TTY — keyboard disabled.")
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
        _set_show(int(c) - 1)
    elif c in "abcde":
        _set_show("abcde".index(c))
    elif c in "0g":
        _set_show(-1)
    elif c == "q":
        print("\n[KBD] q -> quit")
        if g_main_loop is not None: g_main_loop.quit()
    elif c == "?":
        print("[KBD] keys: 1-5 / a-e = switch  0/g = grid  q = quit")


def _set_show(idx):
    global current_show
    current_show = idx
    if tiler_elem is None: return
    if idx == -1:
        tiler_elem.set_property("rows", 2)
        tiler_elem.set_property("columns", 3)
        tiler_elem.set_property("show-source", -1)
        print("[KBD] grid (2x3)")
    else:
        tiler_elem.set_property("rows", 1)
        tiler_elem.set_property("columns", 1)
        tiler_elem.set_property("show-source", idx)
        print(f"[KBD] showing {cam_names[idx]}")


# ── Bus / pipeline ──────────────────────────────────────────────────
def on_bus_message(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        print(f"[BUS][ERR] {err.message}")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, dbg = message.parse_warning()
        print(f"[BUS][WARN] {err.message}")
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

    nvvconv = make("nvvideoconvert", "nvvconv-pre")
    capsf   = make("capsfilter", "capsf-pre")
    capsf.set_property("caps", Gst.Caps.from_string(
        "video/x-raw(memory:NVMM), format=RGBA"))
    pipe.add(nvvconv); pipe.add(capsf)

    nvinfer = make("nvinfer", "nvinfer")
    nvinfer.set_property("config-file-path", INFER_CONFIG)
    pipe.add(nvinfer)

    tracker = make("nvtracker", "tracker")
    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 384)
    tracker.set_property("gpu-id", 0)
    tracker.set_property("ll-lib-file", TRACKER_LIB)
    tracker.set_property("ll-config-file", TRACKER_CFG)
    tracker.set_property("enable-batch-process", 1)
    pipe.add(tracker)

    tiler = make("nvmultistreamtiler", "tiler")
    tiler.set_property("rows", 1); tiler.set_property("columns", 1)
    tiler.set_property("width", TILER_W); tiler.set_property("height", TILER_H)
    tiler.set_property("show-source", 0)
    pipe.add(tiler)

    osd = make("nvdsosd", "osd")
    pipe.add(osd)

    transform = make("nvegltransform", "egltransform")
    sink = make("nveglglessink", "sink")
    sink.set_property("sync", 0)
    pipe.add(transform); pipe.add(sink)

    for a, b, name in [
        (streammux, nvvconv, "mux->nvvconv"),
        (nvvconv, capsf, "nvvconv->caps"),
        (capsf, nvinfer, "caps->nvinfer"),
        (nvinfer, tracker, "nvinfer->tracker"),
        (tracker, tiler, "tracker->tiler"),
        (tiler, osd, "tiler->osd"),
        (osd, transform, "osd->transform"),
        (transform, sink, "transform->sink"),
    ]:
        if not a.link(b):
            raise RuntimeError(f"link {name} failed")

    nvinfer.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER,
                                            probe1_after_infer, 0)
    tracker.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER,
                                            probe2_after_tracker, 0)
    return pipe, tiler


def main():
    global csv_file, csv_writer, g_main_loop, classifier, tiler_elem

    print(f"[INFO] Loading brand classifier: {BRAND_ENGINE}")
    classifier = BrandClassifier(BRAND_ENGINE, max_batch=BRAND_MAX_BATCH)
    print("[INFO] classifier ready")

    csv_file = open(LOG_CSV, "w", newline="", buffering=1)
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp","cam","cam_frame","class","confidence","track_id","brand_id"])

    pipe, tiler = build_pipeline()
    tiler_elem = tiler

    g_main_loop = GLib.MainLoop()
    bus = pipe.get_bus(); bus.add_signal_watch()
    bus.connect("message", on_bus_message, g_main_loop)

    stop_evt = threading.Event()
    threading.Thread(target=kbd_loop, args=(stop_evt,), daemon=True).start()

    def _sig(*_):
        print("\n[SIG] stopping...")
        g_main_loop.quit()
    signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)

    print(f"[INFO] v6 viewer — interval=1 + IOU tracker + brand cache by track_id")
    print(f"[INFO] keys: 1-5 / a-e switch  0/g grid  q quit  ? help")
    print(f"[INFO] CSV: {LOG_CSV}\n")

    if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("[FATAL] start failed"); sys.exit(1)

    try:
        g_main_loop.run()
    finally:
        print("\n[INFO] stopping pipeline...")
        stop_evt.set()
        pipe.set_state(Gst.State.NULL)
        try: csv_file.flush(); csv_file.close()
        except Exception: pass
        print(f"[INFO] CSV saved -> {LOG_CSV}")
        print(f"[INFO] unique tracks classified: {len(brand_cache)}")


if __name__ == "__main__":
    main()
