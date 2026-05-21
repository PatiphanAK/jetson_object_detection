"""
main.py
=======
DeepStream pipeline: 5x RTSP -> nvinfer (YOLO26 end2end) -> probe
                     -> per-car BrandClassifier (TRT, FP16) + Color extraction.

CSV log only (timestamped {epoch}_v4_brandy.log — self-healing safe).
No stdout REPORT. No fps tracker. No cross-batch counters. Unknown class
masked per Lanta inference.
"""
import configparser
import csv
import ctypes
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

cuda.init()  # init driver — context is created per-classifier

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds

from car_color import top_color_name


# ── Config ──────────────────────────────────────────────────────────
APP_CONFIG = "config.conf"


def _split_csv_items(value):
    return [x.strip() for x in value.replace("\n", ",").split(",") if x.strip()]


def load_runtime_config(path):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    if not parser.read(path):
        raise FileNotFoundError(f"Cannot read config file: {path}")
    if not parser.has_section("app"):
        raise RuntimeError(f"Missing [app] section in {path}")
    if not parser.has_section("sources"):
        raise RuntimeError(f"Missing [sources] section in {path}")

    app = parser["app"]
    sources = dict(parser.items("sources"))
    coco_classes = _split_csv_items(app.get("coco-classes", ""))
    if not sources:
        raise RuntimeError(f"No cameras configured in [sources] of {path}")
    if not coco_classes:
        raise RuntimeError(f"No coco-classes configured in [app] of {path}")

    return {
        "sources": sources,
        "infer_config": app.get("infer-config", path),
        "log_csv": app.get("log-csv", "v4_brandy.log"),
        "brand_engine": app.get("brand-engine", "model/classy.engine"),
        "conf_threshold": app.getfloat("conf-threshold", fallback=0.25),
        "car_class_id": app.getint("car-class-id", fallback=2),
        "brand_max_batch": app.getint("brand-max-batch", fallback=16),
        "muxer_w": app.getint("muxer-width", fallback=640),
        "muxer_h": app.getint("muxer-height", fallback=640),
        "batched_push_timeout_us": app.getint("batched-push-timeout-us", fallback=40000),
        "max_det_per_image": app.getint("max-det-per-image", fallback=300),
        "det_features": app.getint("det-features", fallback=6),
        "coco_classes": coco_classes,
        "unknown_brand_idx": app.getint("unknown-brand-idx", fallback=22),
        "csv_flush_every_rows": app.getint("csv-flush-every-rows", fallback=60),
    }


CFG = load_runtime_config(APP_CONFIG)
SOURCES = CFG["sources"]
INFER_CONFIG = CFG["infer_config"]
LOG_CSV = CFG["log_csv"]
BRAND_ENGINE = CFG["brand_engine"]
CONF_THRESHOLD = CFG["conf_threshold"]
CAR_CLASS_ID = CFG["car_class_id"]
BRAND_MAX_BATCH = CFG["brand_max_batch"]
MUXER_W, MUXER_H = CFG["muxer_w"], CFG["muxer_h"]
BATCHED_PUSH_TIMEOUT_US = CFG["batched_push_timeout_us"]
MAX_DET_PER_IMAGE = CFG["max_det_per_image"]
DET_FEATURES = CFG["det_features"]
COCO_CLASSES = CFG["coco_classes"]
UNKNOWN_BRAND_IDX = CFG["unknown_brand_idx"]
CSV_FLUSH_EVERY_ROWS = CFG["csv_flush_every_rows"]


# ── ShuffleNet brand classifier (pycuda + TRT 8.2) ──────────────────
class BrandClassifier:
    IMG = 224
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    NUM_CLASSES = 26

    def __init__(self, engine_path, max_batch=16, unknown_idx=UNKNOWN_BRAND_IDX):
        self.max_batch = max_batch
        self.unknown_idx = int(unknown_idx)
        self.cuda_ctx = cuda.Device(0).make_context()
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f:
                data = f.read()
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(data)
            self.ctx = self.engine.create_execution_context()

            self.in_idx = self.engine.get_binding_index("input")
            self.out_idx = self.engine.get_binding_index("output")
            self.in_dtype = trt.nptype(self.engine.get_binding_dtype(self.in_idx))
            self.out_dtype = trt.nptype(self.engine.get_binding_dtype(self.out_idx))

            in_size_elem = max_batch * 3 * self.IMG * self.IMG
            out_size_elem = max_batch * self.NUM_CLASSES
            self.h_in = cuda.pagelocked_empty(in_size_elem, self.in_dtype)
            self.h_out = cuda.pagelocked_empty(out_size_elem, self.out_dtype)
            self.d_in = cuda.mem_alloc(self.h_in.nbytes)
            self.d_out = cuda.mem_alloc(self.h_out.nbytes)
            self.stream = cuda.Stream()

            self._resize_buf = np.empty((self.IMG, self.IMG, 3), dtype=np.uint8)
            self._scratch_hwc = np.empty((self.IMG, self.IMG, 3), dtype=np.float32)
            self._batch_f32 = np.empty(
                (max_batch, 3, self.IMG, self.IMG), dtype=np.float32
            )
            self._batch_typed = np.empty(
                (max_batch, 3, self.IMG, self.IMG), dtype=self.in_dtype
            )
            self._inv_255 = np.float32(1.0 / 255.0)
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
            for i in range(n):
                cv2.resize(
                    crops[i],
                    (self.IMG, self.IMG),
                    dst=self._resize_buf,
                    interpolation=cv2.INTER_LINEAR,
                )
                np.multiply(
                    self._resize_buf,
                    self._inv_255,
                    out=self._scratch_hwc,
                    casting="unsafe",
                )
                self._batch_f32[i] = self._scratch_hwc.transpose(2, 0, 1)
            view_f32 = self._batch_f32[:n]
            np.subtract(view_f32, self.MEAN, out=view_f32)
            np.divide(view_f32, self.STD, out=view_f32)
            np.copyto(self._batch_typed[:n], view_f32, casting="unsafe")

            flat_size = n * 3 * self.IMG * self.IMG
            self.h_in[:flat_size] = self._batch_typed[:n].ravel()
            self.ctx.set_binding_shape(self.in_idx, (n, 3, self.IMG, self.IMG))
            cuda.memcpy_htod_async(self.d_in, self.h_in[:flat_size], self.stream)
            self.ctx.execute_async_v2(
                bindings=[int(self.d_in), int(self.d_out)],
                stream_handle=self.stream.handle,
            )
            out_n = n * self.NUM_CLASSES
            cuda.memcpy_dtoh_async(self.h_out[:out_n], self.d_out, self.stream)
            self.stream.synchronize()

            logits = self.h_out[:out_n].reshape(n, self.NUM_CLASSES).astype(np.float32)
            if 0 <= self.unknown_idx < self.NUM_CLASSES:
                logits[:, self.unknown_idx] = -np.inf
            return logits.argmax(axis=1).astype(np.int32).tolist()
        finally:
            self.cuda_ctx.pop()


# ── Globals ─────────────────────────────────────────────────────────
cam_names = list(SOURCES.keys())
cam_frame_count = defaultdict(int)  # bounded: at most len(cam_names) ints
rows_since_flush = 0

csv_file = None
csv_writer = None
g_main_loop = None
classifier = None


# ── Tensor + frame helpers ──────────────────────────────────────────
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
    return np.ctypeslib.as_array(cptr, shape=(n_elem,)).copy().reshape(shape)


def get_frame_rgb(gst_buffer, batch_id):
    """Return (H, W, 4) RGBA *view* — zero-copy at the full-frame level."""
    try:
        return pyds.get_nvds_buf_surface(hash(gst_buffer), batch_id)
    except Exception:
        return None


def crop_clip(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    x1 = max(0, min(W - 1, int(x1)))
    x2 = max(0, min(W, int(x2)))
    y1 = max(0, min(H - 1, int(y1)))
    y2 = max(0, min(H, int(y2)))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        return img[y1:y2, x1:x2, :3].copy()
    return img[y1:y2, x1:x2].copy()


# ── Probe — detect + per-car brand+color, write CSV ────────────────
def infer_src_pad_buffer_probe(pad, info, u_data):
    global rows_since_flush
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None:
        return Gst.PadProbeReturn.OK

    ts = time.time()
    l_frame = batch_meta.frame_meta_list
    pending_rows = []      # one row per (cam, class) per frame
    car_crops = []         # list of RGB crops to classify
    car_row_refs = []      # references into pending_rows to back-fill brand/color

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
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
            continue

        layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
        arr = layer_to_numpy(layer)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        keep = arr[:, 4] > CONF_THRESHOLD
        dets = arr[keep]
        if len(dets) == 0:
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
            continue

        # Aggregate per-class counts for this frame (local, no cross-batch state).
        frame_cls = defaultdict(lambda: {"count": 0, "max_conf": 0.0})
        for row in dets:
            cid = int(row[5])
            cname = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else f"class-{cid}"
            e = frame_cls[cname]
            e["count"] += 1
            if row[4] > e["max_conf"]:
                e["max_conf"] = float(row[4])

        car_rows_in_this_frame = []
        for cls_name, info_d in frame_cls.items():
            pending_rows.append([
                f"{ts:.3f}",
                src_id,
                this_frame,
                cls_name,
                f"{info_d['max_conf']:.3f}",
                info_d["count"],
                "",   # brand_id (back-filled for car)
                "",   # color    (back-filled for car)
            ])
            if cls_name == "car":
                car_rows_in_this_frame.append(pending_rows[-1])

        # Top-confidence car only -> brand + color
        car_dets = dets[dets[:, 5].astype(int) == CAR_CLASS_ID]
        if len(car_dets) > 0:
            frame_rgb = get_frame_rgb(gst_buffer, frame_meta.batch_id)
            if frame_rgb is not None:
                top = car_dets[np.argmax(car_dets[:, 4])]
                crop = crop_clip(frame_rgb, top[0], top[1], top[2], top[3])
                if crop is not None and crop.size > 0:
                    car_crops.append(crop)
                    car_row_refs.append(
                        car_rows_in_this_frame[0] if car_rows_in_this_frame else None
                    )

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    # Brand + color for collected car crops (one classify call per batch).
    if car_crops:
        try:
            brand_ids = classifier.classify(car_crops)
        except Exception as e:
            print(f"[WARN] brand classify failed: {e}")
            brand_ids = [-1] * len(car_crops)
        for i, (ref, brand) in enumerate(zip(car_row_refs, brand_ids)):
            if ref is None:
                continue
            if brand >= 0:
                ref[6] = str(brand)
            try:
                bgr = cv2.cvtColor(car_crops[i], cv2.COLOR_RGB2BGR)
                cname = top_color_name(bgr)
                if cname:
                    ref[7] = cname
            except Exception:
                pass

    # Flush
    for row in pending_rows:
        csv_writer.writerow(row)
        rows_since_flush += 1
    if rows_since_flush >= CSV_FLUSH_EVERY_ROWS:
        csv_file.flush()
        rows_since_flush = 0

    return Gst.PadProbeReturn.OK


# ── Pipeline ────────────────────────────────────────────────────────
def on_bus_message(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("[BUS] EOS")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        src = message.src.get_name() if message.src else "?"
        print(f"[BUS][ERROR] {src}: {err.message}")
        if dbg:
            print(f"  debug: {dbg}")
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
    try:
        streammux.set_property("buffer-pool-size", 8)
        streammux.set_property("nvbuf-memory-type", 4)
    except Exception:
        pass
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
            if sink_pad is None or sink_pad.is_linked():
                return
            if pad.link(sink_pad) == Gst.PadLinkReturn.OK:
                print(f"[OK] linked {name} -> streammux sink_{idx}")

        src.connect("pad-added", on_pad_added)
        pipeline.add(src)

    nvvconv = make("nvvideoconvert", "nvvconv")
    capsf = make("capsfilter", "capsf")
    capsf.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    )
    pipeline.add(nvvconv)
    pipeline.add(capsf)

    queue1 = make("queue", "queue1")
    queue1.set_property("max-size-buffers", 4)
    queue1.set_property("max-size-bytes", 0)
    queue1.set_property("max-size-time", 0)
    queue1.set_property("leaky", 2)
    pipeline.add(queue1)

    nvinfer = make("nvinfer", "nvinfer")
    nvinfer.set_property("config-file-path", INFER_CONFIG)
    pipeline.add(nvinfer)

    sink = make("fakesink", "sink")
    sink.set_property("sync", 0)
    sink.set_property("async", 0)
    pipeline.add(sink)

    if not streammux.link(nvvconv):
        raise RuntimeError("link mux->nvvconv failed")
    if not nvvconv.link(capsf):
        raise RuntimeError("link nvvconv->caps failed")
    if not capsf.link(queue1):
        raise RuntimeError("link caps->queue1 failed")
    if not queue1.link(nvinfer):
        raise RuntimeError("link queue1->nvinfer failed")
    if not nvinfer.link(sink):
        raise RuntimeError("link nvinfer->sink failed")

    nvinfer.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, infer_src_pad_buffer_probe, 0
    )
    return pipeline


def main():
    global csv_file, csv_writer, g_main_loop, classifier, LOG_CSV

    print(f"[INFO] Loading brand classifier: {BRAND_ENGINE}")
    classifier = BrandClassifier(
        BRAND_ENGINE, max_batch=BRAND_MAX_BATCH, unknown_idx=UNKNOWN_BRAND_IDX
    )
    print(
        f"[INFO] classifier ready (Unknown-mask idx={UNKNOWN_BRAND_IDX}, "
        f"{len(SOURCES)} cameras)"
    )

    # Self-healing: prefix start-epoch -> restart never overwrites previous run.
    epoch = int(time.time())
    _p = Path(LOG_CSV)
    LOG_CSV = str(_p.with_name(f"{epoch}_{_p.name}"))
    csv_file = open(LOG_CSV, "w", newline="", buffering=1)
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "ts_unix", "cam_id", "cam_frame", "class",
        "confidence", "count_this_frame", "brand_id", "color",
    ])
    print(f"[INFO] Detection log -> {LOG_CSV}")

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

    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("[FATAL] Unable to set pipeline to PLAYING")
        sys.exit(1)

    try:
        g_main_loop.run()
    finally:
        print("[INFO] Stopping pipeline...")
        pipeline.set_state(Gst.State.NULL)
        try:
            csv_file.flush()
            csv_file.close()
        except Exception:
            pass
        print(f"[INFO] CSV saved -> {LOG_CSV}")


if __name__ == "__main__":
    main()
