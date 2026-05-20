import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import psutil
import time
import os

# ─── TensorRT Logger & Log File ──────────────────────────
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
LOG_FILE   = "infer_log.txt"

# ═══════════════════════════════════════════════════════════
# RTSP CONSTANTS  — แก้ตรงนี้อย่างเดียว
# ═══════════════════════════════════════════════════════════
RTSP_URL            = "rtsp://10.0.11.37:8554/vdo1"
RTSP_TRANSPORT      = "udp"          # "tcp" | "udp"ƒ
RTSP_BUFFER_SIZE    = 1             # จำนวน frame ที่ buffer (1 = latency ต่ำสุด)
RTSP_RECONNECT_SEC  = 3              # วินาทีรอก่อน reconnect เมื่อ stream หลุด
RTSP_FFMPEG_OPTIONS = (
    f"rtsp_transport;{RTSP_TRANSPORT}|"
    "buffer_size;1048576"
)
# ═══════════════════════════════════════════════════════════

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


# ─── TensorRT ────────────────────────────────────────────
def load_engine(engine_path: str):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for i in range(engine.num_bindings):
        name  = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        size  = trt.volume(shape)

        host_mem   = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        entry = {
            "name":   name,
            "host":   host_mem,
            "device": device_mem,
            "shape":  tuple(shape),
        }

        if engine.binding_is_input(i):
            inputs.append(entry)
        else:
            outputs.append(entry)

    return inputs, outputs, bindings, stream


def trt_infer(context, inputs, outputs, bindings, stream, blob):
    np.copyto(inputs[0]["host"], blob.ravel())
    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
    stream.synchronize()
    return [out["host"].reshape(out["shape"]) for out in outputs]


# ─── VideoCapture helper ─────────────────────────────────
def open_capture(source) -> cv2.VideoCapture:
    """
    เปิด VideoCapture รองรับ 3 รูปแบบ:
      - int          → webcam index
      - rtsp://...   → RTSP stream  (ใช้ FFMPEG backend + constant vars)
      - string อื่น  → local file
    """
    if isinstance(source, str) and source.startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = RTSP_FFMPEG_OPTIONS
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, RTSP_BUFFER_SIZE)
    return cap


# ─── Pre / Post / Draw ───────────────────────────────────
def preprocess(frame: np.ndarray, input_size: int = 640):
    orig_h, orig_w = frame.shape[:2]
    scale   = input_size / max(orig_h, orig_w)
    new_w   = int(orig_w * scale)
    new_h   = int(orig_h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    blob = padded[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.ascontiguousarray(np.transpose(blob, (2, 0, 1))[np.newaxis])
    return blob, scale, orig_w, orig_h


def postprocess(output, scale, orig_w, orig_h,
                conf_thres=0.25, iou_thres=0.45):
    preds       = output[0].squeeze()       # (84, 8400)
    boxes_raw   = preds[:4].T
    scores      = preds[4:].T

    class_ids   = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    mask        = confidences > conf_thres
    boxes_raw   = boxes_raw[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    if len(boxes_raw) == 0:
        return []

    x1   = boxes_raw[:, 0] - boxes_raw[:, 2] / 2
    y1   = boxes_raw[:, 1] - boxes_raw[:, 3] / 2
    x2   = boxes_raw[:, 0] + boxes_raw[:, 2] / 2
    y2   = boxes_raw[:, 1] + boxes_raw[:, 3] / 2
    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    indices = cv2.dnn.NMSBoxes(
        xyxy.tolist(), confidences.tolist(), conf_thres, iou_thres
    )

    results = []
    for i in indices.flatten():
        x1r = max(0, int(xyxy[i, 0] / scale))
        y1r = max(0, int(xyxy[i, 1] / scale))
        x2r = min(orig_w, int(xyxy[i, 2] / scale))
        y2r = min(orig_h, int(xyxy[i, 3] / scale))
        results.append({
            "class_id":   int(class_ids[i]),
            "class_name": COCO_CLASSES[int(class_ids[i])],
            "confidence": float(confidences[i]),
            "box":        [x1r, y1r, x2r, y2r],
        })
    return results


def draw(frame: np.ndarray, results: list) -> np.ndarray:
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


# ─── Log ─────────────────────────────────────────────────
def log_stats(frame_id: int, ms: float, n_objects: int):
    mem  = psutil.virtual_memory()
    line = (
        f"frame={frame_id} "
        f"infer_ms={ms:.1f} "
        f"objects={n_objects} "
        f"ram_used_mb={mem.used // 1024**2} "
        f"ram_percent={mem.percent} "
        f"ram_available_mb={mem.available // 1024**2}\n"
    )
    with open(LOG_FILE, "a") as f:
        f.write(line)
    print(f"[LOG] {line.strip()}")


# ─── Main loop ───────────────────────────────────────────
def run(source, engine_path: str, save_path: str = None,
        conf_thres=0.25, iou_thres=0.45, log_every=30):

    engine  = load_engine(engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    print(f"Engine loaded: {engine_path}")

    is_rtsp = isinstance(source, str) and source.startswith("rtsp://")

    cap = open_capture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    print(f"Source opened: {source}")

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        print(f"Saving → {save_path}")

    open(LOG_FILE, "w").close()

    frame_id = 0
    while True:
        ret, frame = cap.read()

        # ─── reconnect เมื่อ RTSP stream หลุด ────────────
        if not ret:
            if is_rtsp:
                print(f"[WARN] Stream lost — reconnecting in {RTSP_RECONNECT_SEC}s ...")
                cap.release()
                time.sleep(RTSP_RECONNECT_SEC)
                cap = open_capture(source)
                continue
            else:
                break   # ไฟล์ / webcam → จบ loop ปกติ

        t0 = time.perf_counter()
        blob, scale, orig_w, orig_h = preprocess(frame)
        raw_out = trt_infer(context, inputs, outputs, bindings, stream, blob)
        results = postprocess(raw_out, scale, orig_w, orig_h, conf_thres, iou_thres)
        ms = (time.perf_counter() - t0) * 1000

        frame = draw(frame, results)
        cv2.putText(frame,
                    f"Inference: {ms:.1f}ms | Objects: {len(results)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if writer:
            writer.write(frame)

        cv2.imshow("YOLOv8 TensorRT", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):   # กด q เพื่อออก
            print("[INFO] User pressed 'q' — stopping.")
            break

        if frame_id % log_every == 0:
            log_stats(frame_id, ms, len(results))

        frame_id += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done — {frame_id} frames | log → {LOG_FILE}")


# ─── Entry ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 TensorRT Inference")
    parser.add_argument("--source",    default=RTSP_URL,
                        help="RTSP URL, video file path, or webcam index")
    parser.add_argument("--model",     default="yolov8n.engine")
    parser.add_argument("--save",      default=None,
                        help="Output video path (optional)")
    parser.add_argument("--conf",      type=float, default=0.25)
    parser.add_argument("--iou",       type=float, default=0.45)
    parser.add_argument("--log-every", type=int,   default=30)
    args = parser.parse_args()

    # digit string → webcam index, อื่นๆ ส่งตรง (รองรับ RTSP URL / file path)
    source = int(args.source) if args.source.isdigit() else args.source

    run(source, args.model, args.save, args.conf, args.iou, args.log_every)
