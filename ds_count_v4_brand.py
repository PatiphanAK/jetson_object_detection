"""
ds_count_v4_brand.py
====================
v3 + Secondary classifier (ShuffleNet brand classifier) on every "car" detection.

Pipeline:
  5x nvurisrcbin -> nvstreammux(5x640x640, NV12)
                 -> nvvideoconvert -> caps(RGBA) [NEW: so probe can read pixels]
                 -> nvinfer (YOLO26 end2end)
                 -> fakesink
  (probe reads tensor + frame surface, classifies cars, logs to CSV)

On each frame:
  - For each YOLO detection above CONF_THRESHOLD:
      log: timestamp, cam, cam_frame, class, conf, count_this_frame, brand_id
  - If detection is "car", crop bbox, resize 224x224, ImageNet-normalize,
    add to current-batch crops. After scanning the whole batch_meta, run
    ShuffleNet TRT engine in one shot on the stacked crops and back-fill
    each car row's brand_id (argmax over 26 classes).
"""
import csv
import ctypes
import signal
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
cuda.init()  # init driver — context is created per-classifier

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
LOG_CSV          = "detect_log_yolo26_brand.csv"
BRAND_ENGINE     = "shufflenet_brand.engine"

CONF_THRESHOLD       = 0.25
CAR_CLASS_ID         = 2     # COCO 'car'
BRAND_MAX_BATCH      = 16    # engine profile max
REPORT_EVERY_S       = 300
CSV_FLUSH_EVERY_ROWS = 60

MUXER_W, MUXER_H         = 640, 640
BATCHED_PUSH_TIMEOUT_US  = 40_000

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


# ── ShuffleNet brand classifier (pycuda + TRT 8.2) ──────────────────
class BrandClassifier:
    """
    ShuffleNet brand classifier. Manages its own CUDA context so it does not
    collide with DeepStream's primary context.

    Input  : list of (h, w, 3) uint8 RGB crops (any size)
    Output : list[int] of brand class ids (0..25); empty list for empty input.
    """
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

            in_size_elem  = max_batch * 3 * self.IMG * self.IMG
            out_size_elem = max_batch * 26
            self.h_in  = cuda.pagelocked_empty(in_size_elem, self.in_dtype)
            self.h_out = cuda.pagelocked_empty(out_size_elem, self.out_dtype)
            self.d_in  = cuda.mem_alloc(self.h_in.nbytes)
            self.d_out = cuda.mem_alloc(self.h_out.nbytes)
            self.stream = cuda.Stream()
        finally:
            self.cuda_ctx.pop()

    def classify(self, crops):
        n = len(crops)
        if n == 0:
            return []
        if n > self.max_batch:
            n = self.max_batch
        self.cuda_ctx.push()
        try:
            batch = np.empty((n, 3, self.IMG, self.IMG), dtype=np.float32)
            for i in range(n):
                img = cv2.resize(crops[i], (self.IMG, self.IMG), interpolation=cv2.INTER_LINEAR)
                arr = img.astype(np.float32) * (1.0 / 255.0)
                arr = arr.transpose(2, 0, 1)
                batch[i] = arr
            batch = (batch - self.MEAN) / self.STD
            batch_fp16 = batch.astype(self.in_dtype)
            flat = batch_fp16.ravel()
            self.h_in[:flat.size] = flat
            self.ctx.set_binding_shape(self.in_idx, (n, 3, self.IMG, self.IMG))
            cuda.memcpy_htod_async(self.d_in, self.h_in[:flat.size], self.stream)
            bindings = [int(self.d_in), int(self.d_out)]
            self.ctx.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
            out_n = n * 26
            cuda.memcpy_dtoh_async(self.h_out[:out_n], self.d_out, self.stream)
            self.stream.synchronize()
            logits = self.h_out[:out_n].reshape(n, 26).astype(np.float32)
            brands = logits.argmax(axis=1).astype(np.int32)
            return brands.tolist()
        finally:
            self.cuda_ctx.pop()

# ── State ───────────────────────────────────────────────────────────
cam_names         = list(SOURCES.keys())
cam_frame_count   = defaultdict(int)
total_counts      = {c: defaultdict(int) for c in cam_names}
total_brands      = defaultdict(int)      # brand_id -> total cars
batch_count       = 0
last_report_t     = time.time()
rows_since_flush  = 0

csv_file   = None
csv_writer = None
g_main_loop = None
classifier  = None


def layer_to_numpy(layer):
    dims = layer.dims
    shape = tuple(int(dims.d[i]) for i in range(dims.numDims))
    n_elem = 1
    for s in shape:
        n_elem *= s
    if n_elem == 0:
        return np.empty(shape, dtype=np.float32)
    ptr_type = ctypes.POINTER(ctypes.c_float)
    cptr = ctypes.cast(pyds.get_ptr(layer.buffer), ptr_type)
    arr = np.ctypeslib.as_array(cptr, shape=(n_elem,)).copy().reshape(shape)
    return arr


def get_frame_rgb(gst_buffer, batch_id):
    """Pull RGBA frame surface as (H, W, 3) uint8 RGB numpy view (zero-copy where possible)."""
    try:
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), batch_id)
        # n_frame is (H, W, 4) RGBA uint8
        return n_frame[:, :, :3].copy()   # copy -> RGB
    except Exception as e:
        return None


def crop_clip(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    x1 = max(0, min(W - 1, int(x1)))
    x2 = max(0, min(W,     int(x2)))
    y1 = max(0, min(H - 1, int(y1)))
    y2 = max(0, min(H,     int(y2)))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return img[y1:y2, x1:x2]


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

    # Per-batch staging for ShuffleNet: list of (row_index, crop_np)
    pending_rows = []     # list of CSV row lists (mutable for brand_id back-fill)
    car_crops    = []
    car_row_refs = []     # rows in pending_rows that need brand_id filled

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        src_id = frame_meta.source_id
        cam_name = cam_names[src_id] if src_id < len(cam_names) else f"src-{src_id}"
        cam_frame_count[cam_name] += 1
        this_frame = cam_frame_count[cam_name]

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
            try:
                l_user = l_user.next
            except StopIteration:
                break

        if tensor_meta is None or tensor_meta.num_output_layers < 1:
            try: l_frame = l_frame.next
            except StopIteration: break
            continue

        layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
        arr = layer_to_numpy(layer)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        keep = arr[:, 4] > CONF_THRESHOLD
        dets = arr[keep]
        if len(dets) == 0:
            try: l_frame = l_frame.next
            except StopIteration: break
            continue

        # Aggregate per-class counts for log
        frame_cls = defaultdict(lambda: {"count": 0, "max_conf": 0.0})
        car_bboxes_this_frame = []   # list of (x1,y1,x2,y2) for cars
        for row in dets:
            x1, y1, x2, y2, score, cls = row
            cid = int(cls)
            cname = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else f"class-{cid}"
            e = frame_cls[cname]
            e["count"] += 1
            if score > e["max_conf"]:
                e["max_conf"] = float(score)
            total_counts[cam_name][cname] += 1
            if cid == CAR_CLASS_ID:
                car_bboxes_this_frame.append((x1, y1, x2, y2))

        # Emit one CSV row per class for this frame
        car_rows_in_this_frame = []
        for cls, info_d in frame_cls.items():
            row = [ts, cam_name, this_frame, cls,
                   f"{info_d['max_conf']:.3f}", info_d["count"], ""]
            pending_rows.append(row)
            if cls == "car":
                car_rows_in_this_frame.append(row)

        # If any car: pull frame surface ONCE, crop each car, push to batch
        if car_bboxes_this_frame:
            frame_rgb = get_frame_rgb(gst_buffer, frame_meta.batch_id)
            if frame_rgb is not None:
                # We don't get per-car brand row distinction in CSV (one row per class).
                # So we'll classify the HIGHEST-confidence car crop only per frame
                # and put its brand_id in the single 'car' row.
                # Pick highest conf car bbox
                # (we don't carry conf here; re-derive from dets)
                car_dets = dets[(dets[:, 5].astype(int) == CAR_CLASS_ID)]
                # sort by confidence desc
                car_dets = car_dets[np.argsort(-car_dets[:, 4])]
                # take top crop for this frame
                top = car_dets[0]
                crop = crop_clip(frame_rgb, top[0], top[1], top[2], top[3])
                if crop is not None and crop.size > 0:
                    car_crops.append(crop)
                    # back-fill ref: the single 'car' row in this frame
                    if car_rows_in_this_frame:
                        car_row_refs.append(car_rows_in_this_frame[0])
                    else:
                        car_row_refs.append(None)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    # Run brand classifier in batches of BRAND_MAX_BATCH
    if car_crops:
        i = 0
        while i < len(car_crops):
            chunk = car_crops[i:i + BRAND_MAX_BATCH]
            try:
                brands = classifier.classify(chunk)
            except Exception as e:
                print(f"[WARN] brand classify failed: {e}")
                brands = [-1] * len(chunk)
            for j, b in enumerate(brands):
                ref = car_row_refs[i + j]
                if ref is not None:
                    ref[6] = str(b)
                if b >= 0:
                    total_brands[b] += 1
            i += BRAND_MAX_BATCH

    # Write all rows
    for row in pending_rows:
        csv_writer.writerow(row)
        rows_since_flush += 1

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
    if total_brands:
        top_b = sorted(total_brands.items(), key=lambda x: -x[1])[:8]
        bs = " ".join(f"brand_{c}={k}" for c, k in top_b)
        print(f"  brands top:   {bs}")
    print("=" * 60 + "\n")
    sys.stdout.flush()


def on_bus_message(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("[BUS] EOS"); loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        src = message.src.get_name() if message.src else "?"
        print(f"[BUS][ERROR] {src}: {err.message}")
        if dbg: print(f"  debug: {dbg}")
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
        except Exception:
            pass

        def on_pad_added(src_el, pad, mux=streammux, idx=i, name=cam_name):
            sink_pad = mux.get_request_pad(f"sink_{idx}")
            if sink_pad is None: return
            if not sink_pad.is_linked():
                ret = pad.link(sink_pad)
                if ret == Gst.PadLinkReturn.OK:
                    print(f"[OK] linked {name} -> streammux sink_{idx}")

        src.connect("pad-added", on_pad_added)
        pipeline.add(src)

    # NEW: nvvideoconvert + capsfilter to force RGBA so probe can read pixels
    nvvconv = make("nvvideoconvert", "nvvconv")
    capsf   = make("capsfilter", "capsf")
    capsf.set_property("caps",
        Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    pipeline.add(nvvconv); pipeline.add(capsf)

    nvinfer = make("nvinfer", "nvinfer")
    nvinfer.set_property("config-file-path", INFER_CONFIG)
    pipeline.add(nvinfer)

    sink = make("fakesink", "sink")
    sink.set_property("sync", 0); sink.set_property("async", 0)
    pipeline.add(sink)

    if not streammux.link(nvvconv):  raise RuntimeError("link mux->nvvconv failed")
    if not nvvconv.link(capsf):      raise RuntimeError("link nvvconv->caps failed")
    if not capsf.link(nvinfer):      raise RuntimeError("link caps->nvinfer failed")
    if not nvinfer.link(sink):       raise RuntimeError("link nvinfer->sink failed")

    nvinfer_src = nvinfer.get_static_pad("src")
    nvinfer_src.add_probe(Gst.PadProbeType.BUFFER,
                          infer_src_pad_buffer_probe, 0)
    return pipeline


def main():
    global csv_file, csv_writer, g_main_loop, last_report_t, classifier

    print(f"[INFO] Loading brand classifier: {BRAND_ENGINE}")
    classifier = BrandClassifier(BRAND_ENGINE, max_batch=BRAND_MAX_BATCH)
    print("[INFO] classifier ready")

    csv_file = open(LOG_CSV, "w", newline="", buffering=1)
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp", "cam", "cam_frame", "class",
        "confidence", "count_this_frame", "brand_id",
    ])

    pipeline = build_pipeline()
    g_main_loop = GLib.MainLoop()
    bus = pipeline.get_bus(); bus.add_signal_watch()
    bus.connect("message", on_bus_message, g_main_loop)

    def _sig(*_):
        print("\n[SIG] stopping..."); g_main_loop.quit()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    print(f"[INFO] v4 — YOLO26 + ShuffleNet brand classifier, {len(SOURCES)} cameras")
    print(f"[INFO] CSV: {LOG_CSV}")
    print(f"[INFO] Ctrl+C to stop\n")

    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("[FATAL] Unable to set pipeline to PLAYING"); sys.exit(1)

    last_report_t = time.time()
    try:
        g_main_loop.run()
    finally:
        print("\n[INFO] Stopping pipeline...")
        print_report()
        pipeline.set_state(Gst.State.NULL)
        try:
            csv_file.flush(); csv_file.close()
        except Exception:
            pass
        print(f"[INFO] CSV saved -> {LOG_CSV}")


if __name__ == "__main__":
    main()
