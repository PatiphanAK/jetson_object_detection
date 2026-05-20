import cv2
import numpy as np
import onnxruntime as ort
import argparse

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

# สี per class (BGR)
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)


def preprocess(frame: np.ndarray, input_size: int = 640):
    orig_h, orig_w = frame.shape[:2]
    scale = input_size / max(orig_h, orig_w)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    blob = padded[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis]
    return blob, scale, orig_w, orig_h


def postprocess(output, scale, orig_w, orig_h,
                conf_thres=0.25, iou_thres=0.45):
    preds = output[0].squeeze()   # (84, 8400)
    boxes_raw = preds[:4].T
    scores = preds[4:].T

    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    mask = confidences > conf_thres
    boxes_raw = boxes_raw[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_raw) == 0:
        return []

    x1 = boxes_raw[:, 0] - boxes_raw[:, 2] / 2
    y1 = boxes_raw[:, 1] - boxes_raw[:, 3] / 2
    x2 = boxes_raw[:, 0] + boxes_raw[:, 2] / 2
    y2 = boxes_raw[:, 1] + boxes_raw[:, 3] / 2
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
            "class_id": int(class_ids[i]),
            "class_name": COCO_CLASSES[int(class_ids[i])],
            "confidence": float(confidences[i]),
            "box": [x1r, y1r, x2r, y2r]
        })
    return results


def draw(frame: np.ndarray, results: list) -> np.ndarray:
    for r in results:
        x1, y1, x2, y2 = r["box"]
        cid = r["class_id"]
        color = tuple(int(c) for c in COLORS[cid])
        label = f"{r['class_name']} {r['confidence']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, label, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def run(source, model_path: str, save_path: str = None,
        conf_thres=0.25, iou_thres=0.45):

    # load model
    sess = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    print(f"Provider: {sess.get_providers()[0]}")

    # open source — int = webcam index, str = file path
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    # video writer (optional)
    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        print(f"Saving → {save_path}")

    frame_id = 0
    import time
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        blob, scale, orig_w, orig_h = preprocess(frame)
        outputs = sess.run(None, {input_name: blob})
        results = postprocess(outputs, scale, orig_w, orig_h, conf_thres, iou_thres)
        ms = (time.perf_counter() - t0) * 1000

        frame = draw(frame, results)

        # FPS overlay
        fps_text = f"Inference: {ms:.1f}ms | Objects: {len(results)}"
        cv2.putText(frame, fps_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if writer:
            writer.write(frame)

        # cv2.imshow("YOLOv8n Detection", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     print("Stopped by user")
        #     break

        frame_id += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done — {frame_id} frames processed")


# ─── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  default="0",           help="video file path หรือ webcam index (0,1,...)")
    parser.add_argument("--model",   default="yolov8n.onnx", help="ONNX model path")
    parser.add_argument("--save",    default=None,           help="save output video path เช่น out.mp4")
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--iou",     type=float, default=0.45)
    args = parser.parse_args()

    # auto-convert "0","1" → int สำหรับ webcam
    source = int(args.source) if args.source.isdigit() else args.source

    run(source, args.model, args.save, args.conf, args.iou)
