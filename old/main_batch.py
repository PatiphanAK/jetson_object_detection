"""
main_batch.py
=============
Multi-camera batched YOLOv8 TensorRT inference for Jetson Nano.

- Reads N RTSP streams in parallel (one grabber thread per camera)
- "Latest-frame-wins" buffering — kernel doesn't queue, no lag drift
- Auto-reconnect when stream drops
- Stacks latest frames into one batch of shape (N, 3, H, W)
- Single TRT execute_async_v2 call (batched dispatch on GPU)
- Splits output back per camera, NMS, draws boxes
- Optional per-camera annotated .mp4 saving
- VRAM budget guard (refuses to allocate beyond budget on Jetson shared memory)
- Headless (no cv2.imshow)
- Clean PyCUDA shutdown (Context.pop) — no SIGABRT on exit

Usage:
    # 5 cameras, dynamic-batch engine, headless, save per-camera MP4
    python main_batch.py \
        --sources cam-a=rtsp://10.0.11.37:8554/vdo1,cam-b=rtsp://10.0.11.37:8554/vdo2,cam-c=rtsp://10.0.11.37:8554/vdo3,cam-d=rtsp://10.0.11.37:8554/vdo4,cam-e=rtsp://10.0.11.37:8554/vdo5 \
        --model yolov8n_b5.engine \
        --save-dir ./out \
        --vram-gb 3.0

    # Quick test with 1 file source
    python main_batch.py --sources test=video.mp4 --model yolov8n_b5.engine
"""
import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import pycuda.driver as cuda
import tensorrt as trt


# ============================================================
#  Constants
# ============================================================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

RTSP_TRANSPORT = "udp"
RTSP_BUFFER_SIZE = 1
RTSP_RECONNECT_SEC = 3
RTSP_FFMPEG_OPTIONS = (
    f"rtsp_transport;{RTSP_TRANSPORT}|"
    "buffer_size;1048576"
)

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
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)


# ============================================================
#  Per-camera RTSP grabber (one thread per stream)
# ============================================================
class CamGrabber(threading.Thread):
    """
    Latest-frame-wins grabber. Each call to .latest() returns the most
    recent frame, never a stale buffered one. Auto-reconnects on EOF.
    """

    def __init__(self, name: str, source):
        super().__init__(daemon=True, name=f"grab-{name}")
        self.cam_name = name
        self.source = source
        self.is_rtsp = isinstance(source, str) and source.startswith("rtsp://")

        self._latest = None
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()

        self.fps_in = 0.0
        self.connected = False
        self._frame_count = 0
        self._t0 = time.monotonic()
        self.total_frames = 0

    def latest(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._stop_evt.set()

    # ------------------------------------------------------------------
    def _open(self):
        if self.is_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = RTSP_FFMPEG_OPTIONS
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, RTSP_BUFFER_SIZE)
        return cap

    def run(self):
        while not self._stop_evt.is_set():
            cap = self._open()
            if not cap.isOpened():
                print(f"[{self.cam_name}] open failed, retry in {RTSP_RECONNECT_SEC}s")
                self.connected = False
                self._stop_evt.wait(RTSP_RECONNECT_SEC)
                continue

            self.connected = True
            print(f"[{self.cam_name}] connected: {self.source}")

            while not self._stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                with self._lock:
                    self._latest = frame
                self._frame_count += 1
                self.total_frames += 1

                dt = time.monotonic() - self._t0
                if dt >= 1.0:
                    self.fps_in = self._frame_count / dt
                    self._frame_count = 0
                    self._t0 = time.monotonic()

            cap.release()
            self.connected = False
            if not self.is_rtsp:
                print(f"[{self.cam_name}] file EOF")
                return
            print(f"[{self.cam_name}] stream lost, reconnecting in {RTSP_RECONNECT_SEC}s")
            self._stop_evt.wait(RTSP_RECONNECT_SEC)


# ============================================================
#  TensorRT batched runner with VRAM guard
# ============================================================
class BatchedTRT:
    def __init__(self, engine_path: str, batch_size: int,
                 input_size: int, vram_bytes: int):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(engine_path)

        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Could not load engine: {engine_path}")

        self.ctx = self.engine.create_execution_context()
        self.batch_size = batch_size
        self.input_size = input_size

        # Discover bindings, set dynamic input shape if needed
        self.input_idx = None
        self.output_idx = None
        self._set_shapes_and_check()

        # Allocate host/device buffers per binding (VRAM guarded)
        self.host_mem = [None] * self.engine.num_bindings
        self.device_mem = [None] * self.engine.num_bindings
        self.bindings = [0] * self.engine.num_bindings
        self.shapes = [None] * self.engine.num_bindings

        total_bytes = 0
        for i in range(self.engine.num_bindings):
            shape = tuple(self.ctx.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            n_elem = int(np.prod(shape))
            nbytes = n_elem * np.dtype(dtype).itemsize
            total_bytes += nbytes
            if total_bytes > vram_bytes:
                raise MemoryError(
                    f"Binding '{self.engine.get_binding_name(i)}' would push GPU "
                    f"allocation to {total_bytes/1e9:.2f} GB, "
                    f"exceeding budget {vram_bytes/1e9:.2f} GB. "
                    f"Lower --batch, --input-size, or raise --vram-gb."
                )
            host = cuda.pagelocked_empty(n_elem, dtype)
            dev = cuda.mem_alloc(nbytes)
            self.host_mem[i] = host
            self.device_mem[i] = dev
            self.bindings[i] = int(dev)
            self.shapes[i] = shape

        self.stream = cuda.Stream()

        self.input_shape = self.shapes[self.input_idx]
        self.output_shape = self.shapes[self.output_idx]
        print(f"TRT engine OK: input={self.input_shape}  output={self.output_shape}")
        print(f"  TRT GPU mem (bindings only): {total_bytes/1e6:.1f} MB / "
              f"budget {vram_bytes/1e6:.0f} MB")

    # ------------------------------------------------------------------
    def _set_shapes_and_check(self):
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            is_in = self.engine.binding_is_input(i)
            if is_in:
                self.input_idx = i
                if -1 in tuple(shape):
                    new_shape = (self.batch_size, 3, self.input_size, self.input_size)
                    print(f"Setting dynamic input '{name}': {tuple(shape)} -> {new_shape}")
                    if not self.ctx.set_binding_shape(i, new_shape):
                        raise RuntimeError(
                            f"set_binding_shape({new_shape}) failed for '{name}'. "
                            f"Engine probably has fixed batch. Rebuild with build_batch.py."
                        )
                else:
                    if tuple(shape)[0] != self.batch_size:
                        raise RuntimeError(
                            f"Engine has fixed batch={shape[0]} but you asked for "
                            f"batch={self.batch_size}. Rebuild with:\n"
                            f"    python build_batch.py --max-batch {self.batch_size}"
                        )
            else:
                self.output_idx = i

        if not self.ctx.all_binding_shapes_specified:
            raise RuntimeError("Not all binding shapes specified after setup.")

    # ------------------------------------------------------------------
    def infer(self, batch_nchw_f32: np.ndarray) -> np.ndarray:
        """
        batch_nchw_f32: shape (N, 3, H, W), dtype float32 in [0,1]
        Returns: numpy array of shape self.output_shape
        """
        flat = np.ascontiguousarray(batch_nchw_f32).ravel()
        np.copyto(self.host_mem[self.input_idx], flat)
        cuda.memcpy_htod_async(self.device_mem[self.input_idx],
                               self.host_mem[self.input_idx],
                               self.stream)
        self.ctx.execute_async_v2(bindings=self.bindings,
                                  stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_mem[self.output_idx],
                               self.device_mem[self.output_idx],
                               self.stream)
        self.stream.synchronize()
        return self.host_mem[self.output_idx].reshape(self.output_shape)

    # ------------------------------------------------------------------
    def cleanup(self):
        for d in self.device_mem:
            if d is not None:
                try: d.free()
                except Exception: pass


# ============================================================
#  Preprocess / postprocess
# ============================================================
def letterbox(img: np.ndarray, new_size: int):
    """
    Resize-with-pad keeping aspect ratio.
    Returns (padded_img, ratio, (pad_x, pad_y), (orig_w, orig_h))
    """
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(h * r), int(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    pad_y = (new_size - nh) // 2
    pad_x = (new_size - nw) // 2
    padded[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return padded, r, (pad_x, pad_y), (w, h)


def postprocess_one(pred: np.ndarray, ratio: float, pad_xy, orig_wh,
                    conf_thres: float, iou_thres: float):
    """
    pred: (84, A) raw output for ONE image (after splitting the batch)
    Returns list of {class_id, class_name, confidence, box}
    """
    # YOLOv8 raw is (84, A). First 4 rows = xywh, next 80 = class scores.
    boxes_raw = pred[:4].T               # (A, 4)
    scores = pred[4:].T                  # (A, 80)
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(scores.shape[0]), class_ids]

    mask = confidences > conf_thres
    boxes_raw = boxes_raw[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    if len(boxes_raw) == 0:
        return []

    # xywh -> xyxy (still in letterboxed coords)
    cx, cy, w, h = boxes_raw.T
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    xyxy_lb = np.stack([x1, y1, x2, y2], axis=1)

    indices = cv2.dnn.NMSBoxes(
        xyxy_lb.tolist(), confidences.tolist(), conf_thres, iou_thres
    )
    if isinstance(indices, tuple) or len(indices) == 0:
        return []
    indices = np.array(indices).flatten()

    pad_x, pad_y = pad_xy
    orig_w, orig_h = orig_wh
    results = []
    for i in indices:
        x1r = (xyxy_lb[i, 0] - pad_x) / ratio
        y1r = (xyxy_lb[i, 1] - pad_y) / ratio
        x2r = (xyxy_lb[i, 2] - pad_x) / ratio
        y2r = (xyxy_lb[i, 3] - pad_y) / ratio
        x1r = max(0, min(orig_w, int(x1r)))
        y1r = max(0, min(orig_h, int(y1r)))
        x2r = max(0, min(orig_w, int(x2r)))
        y2r = max(0, min(orig_h, int(y2r)))
        results.append({
            "class_id": int(class_ids[i]),
            "class_name": COCO_CLASSES[int(class_ids[i])],
            "confidence": float(confidences[i]),
            "box": [x1r, y1r, x2r, y2r],
        })
    return results


def draw(frame: np.ndarray, results: list, header: str = None) -> np.ndarray:
    if header:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(frame, header, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    for r in results:
        x1, y1, x2, y2 = r["box"]
        color = tuple(int(c) for c in COLORS[r["class_id"]])
        label = f"{r['class_name']} {r['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, label, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


# ============================================================
#  Main pipeline
# ============================================================
def parse_sources(spec: str):
    """
    Parse comma-separated source spec. Supported forms:
        rtsp://host/p,rtsp://host/q       -> auto names cam0, cam1
        cam-a=rtsp://...,cam-b=rtsp://... -> explicit names
        cam=video.mp4                     -> local file (ok)
    Returns list of (name, source) tuples.
    """
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            name, src = tok.split("=", 1)
            name = name.strip()
            src = src.strip()
        else:
            name = f"cam{len(out)}"
            src = tok
        if src.isdigit():
            src = int(src)
        out.append((name, src))
    return out


def main():
    p = argparse.ArgumentParser(description="Multi-cam batched YOLOv8 TRT inference")
    p.add_argument("--sources", required=True,
                   help="Comma list of RTSP/file/index. Use name=url for labels.")
    p.add_argument("--model", default="yolov8n_b5.engine",
                   help="TRT engine path (must support requested batch)")
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--vram-gb", type=float, default=3.0,
                   help="Hard ceiling for TRT binding allocations")
    p.add_argument("--save-dir", default=None,
                   help="If set, write per-camera annotated MP4s here")
    p.add_argument("--save-fps", type=int, default=15,
                   help="FPS to write to per-camera MP4s (the loop's own rate)")
    p.add_argument("--log-every", type=int, default=30,
                   help="Print stats every N batched inferences")
    p.add_argument("--first-frame-timeout", type=float, default=20.0,
                   help="Seconds to wait for each camera to deliver first frame")
    args = p.parse_args()

    cams = parse_sources(args.sources)
    if not cams:
        print("ERROR: no cameras parsed from --sources", file=sys.stderr)
        sys.exit(2)
    if len(cams) > 8:
        print(f"WARNING: {len(cams)} cameras requested — Nano may struggle.")

    print(f"Cameras ({len(cams)}):")
    for n, s in cams:
        print(f"  {n}: {s}")

    # ----- Init CUDA context manually (so we can pop() on exit) -----
    cuda.init()
    cuda_dev = cuda.Device(0)
    cuda_ctx = cuda_dev.make_context()

    try:
        # ----- Start grabber threads -----
        grabbers = [CamGrabber(n, s) for n, s in cams]
        for g in grabbers:
            g.start()

        # ----- Build TRT runner -----
        vram_bytes = int(args.vram_gb * (1 << 30))
        runner = BatchedTRT(
            engine_path=args.model,
            batch_size=len(cams),
            input_size=args.input_size,
            vram_bytes=vram_bytes,
        )

        # ----- Wait for first frame from each camera -----
        print(f"Waiting up to {args.first_frame_timeout}s for first frame from each cam...")
        deadline = time.monotonic() + args.first_frame_timeout
        while time.monotonic() < deadline:
            if all(g.latest() is not None for g in grabbers):
                break
            time.sleep(0.2)
        missing = [g.cam_name for g in grabbers if g.latest() is None]
        if missing:
            print(f"WARN: cameras still empty after timeout: {missing} "
                  f"(will run with black placeholder frames)")

        # ----- Per-camera MP4 writers (optional, lazy-init when shape known) -----
        writers = {}
        save_dir = None
        if args.save_dir:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # ----- Signal handlers -----
        stop_evt = threading.Event()
        def _sig(*_): stop_evt.set()
        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)

        # ----- Main batched inference loop -----
        batch_idx = 0
        loop_t0 = time.monotonic()
        last_log_t = loop_t0
        log_window_frames = 0

        in_h = in_w = args.input_size
        input_size = args.input_size

        while not stop_evt.is_set():
            t_iter = time.monotonic()

            # Snapshot latest frame per camera
            frames = []
            metas = []
            for g in grabbers:
                f = g.latest()
                if f is None:
                    f = np.zeros((in_h, in_w, 3), dtype=np.uint8)
                lb, r, pad, orig = letterbox(f, input_size)
                frames.append((g, f, lb))
                metas.append((r, pad, orig))

            # Build batch tensor (N, 3, H, W) BGR->RGB, /255, NCHW
            batch = np.stack([cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
                              for (_, _, lb) in frames])
            batch = batch.astype(np.float32) / 255.0
            batch = np.ascontiguousarray(batch.transpose(0, 3, 1, 2))

            # Infer
            t_infer = time.monotonic()
            out = runner.infer(batch)        # shape (N, 84, A)
            infer_ms = (time.monotonic() - t_infer) * 1000

            # Split + post + (optionally) save per camera
            total_objects = 0
            for ci, ((g, frame_orig, _lb), (r, pad, orig)) in enumerate(zip(frames, metas)):
                per_cam_pred = out[ci]
                dets = postprocess_one(per_cam_pred, r, pad, orig,
                                       args.conf, args.iou)
                total_objects += len(dets)

                if save_dir is not None:
                    annotated = draw(frame_orig.copy(), dets,
                                     header=f"{g.cam_name}  obj={len(dets)}  "
                                            f"infer={infer_ms:.0f}ms")
                    w = writers.get(g.cam_name)
                    if w is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        h, wd = annotated.shape[:2]
                        out_path = str(save_dir / f"{g.cam_name}.mp4")
                        w = cv2.VideoWriter(out_path, fourcc, args.save_fps, (wd, h))
                        writers[g.cam_name] = w
                        print(f"[{g.cam_name}] writing -> {out_path}")
                    w.write(annotated)

            batch_idx += 1
            log_window_frames += 1

            if batch_idx % args.log_every == 0:
                now = time.monotonic()
                dt = now - last_log_t
                fps_loop = log_window_frames / dt if dt > 0 else 0
                ram = psutil.virtual_memory()
                per_cam = " ".join(
                    f"{g.cam_name}:{g.fps_in:.1f}{'' if g.connected else '*'}"
                    for g in grabbers
                )
                print(
                    f"[batch={batch_idx:>5}] "
                    f"infer={infer_ms:.1f}ms "
                    f"batch_fps={fps_loop:.1f} "
                    f"obj_total={total_objects:>3} "
                    f"ram={ram.percent}% ({ram.used // (1 << 20)}MB) "
                    f"cam_in: {per_cam}"
                )
                last_log_t = now
                log_window_frames = 0

    finally:
        print("Shutting down ...")
        try:
            for g in grabbers:
                g.stop()
        except Exception:
            pass
        try:
            for w in writers.values():
                w.release()
        except Exception:
            pass
        try:
            runner.cleanup()
        except Exception:
            pass
        # Critical for Nano: pop CUDA context before exit (avoids SIGABRT)
        try:
            cuda_ctx.pop()
        except Exception:
            pass
        print("Bye.")


if __name__ == "__main__":
    main()
