import argparse
import json
import queue
import threading
import time

from ultralytics import YOLO
import cv2
import numpy as np
import os
import csv
import torch
import torch.nn as nn
from datetime import datetime
from torchvision import models

try:
    from vehicle_color_classifier import predict_vehicle_color as rule_predict_vehicle_color
except Exception:
    rule_predict_vehicle_color = None

# 1. โหลดโมเดล YOLO
PT_MODEL_PATH = 'yolotune.pt'
ENGINE_MODEL_PATH = 'yolo26n.engine'
MODEL_PATH = ENGINE_MODEL_PATH if os.path.exists(ENGINE_MODEL_PATH) else PT_MODEL_PATH
SOURCE = r'CCTV\cctv3_20260521_162250.mp4'
EDGE_MODE = True
CONF_THRESHOLD = 0.30
IOU_THRESHOLD = 0.45
IMGSZ = 224
FRAME_SKIP = 1
USE_TRAFFIC_LINE_ROI = True
TRAFFIC_LINE_ROI_MARGIN = 96
MIN_BOX_WIDTH = 8
MIN_BOX_HEIGHT = 8
MIN_TRUCK_BUS_CONF = 0.50
CROP_PADDING_X = 30
CROP_PADDING_Y = 20
CROP_SQUARE = False
MIN_CROP_SIZE = 96
ROUTE_LOG_PATH = 'crops/{camera}/vehicle_routes.csv'
CROP_OUTPUT_DIR = 'crops/{camera}/vehicles'
SAVE_CROPS = False
DRAW_ONLY_AFTER_ORIGIN = True
MIN_ROUTE_SEEN_FRAMES = 2
MIN_ROUTE_GAP_FRAMES = 2
LINE_CROSS_MIN_MOVE = 8
LINE_HIT_TOLERANCE = 8
DISPLAY = False
DISPLAY_MAX_WIDTH = 1280
DISPLAY_MAX_HEIGHT = 720
PRINT_EVERY_N_FRAMES = 15
MAX_FRAMES = 0
SAVE_ASYNC = True
SAVE_QUEUE_SIZE = 64
JPEG_QUALITY = 90
REQUIRE_ENGINE_ON_JETSON = True
DECODE_ERROR_LIMIT = 30
CAPTURE_BUFFER_SIZE = 1
LIVE_SOURCE_INITIAL_RETRIES = 3
LIVE_RECONNECT_DELAY_SEC = 1.0
LIVE_RECONNECT_MAX_ATTEMPTS = 0
ENABLE_CROP_CLASSIFIER = True
CLASSIFIER_MODEL_PATH = 'best.pt'
CLASSIFIER_LABELS_PATH = 'label_map.json'
CLASSIFIER_INPUT_SIZE = 224
CLASSIFIER_ENABLED_CLASSES = {'car', 'motorbike', 'bus', 'truck'}
CLASSIFIER_MIN_CONFIDENCE = 0.0
CLASSIFIER_DEVICE = 'auto'
COLOR_CROP_SHRINK_X = 0.18
COLOR_CROP_SHRINK_Y = 0.22
SAVE_COLOR_DEBUG = False
COLOR_DEBUG_DIR = 'crops/color_debug'
ENABLE_COLOR_DETECTION = True
COLOR_ENABLED_CLASSES = {'car', 'motorbike', 'bus', 'truck'}
COLOR_MIN_CONFIDENCE = 0.18
COLOR_LOW_CONF_SHADOW_LIMIT = 0.55
BODY_COLOR_MIN_CHROMA_RATIO = 0.35
BODY_COLOR_MIN_NEUTRAL_RATIO = 0.30
BODY_NEUTRAL_OVERRIDE_CONF = 0.15
BODY_STRONG_CHROMA_CONF = 0.70
BODY_BLACK_RATIO_OVERRIDE = 0.16
MOTORBIKE_CHROMA_BONUS = 0.20
MOTORBIKE_MIN_CHROMA_RATIO = 0.20
MOTORBIKE_MIN_COLOR_QUALITY = 0.16
USE_BG_SUBTRACTION_FOR_COLOR = False
BG_HISTORY = 600
BG_VAR_THRESHOLD = 32
USE_RULE_BASED_COLOR_CLASSIFIER = True
RULE_COLOR_MAX_WIDTH = 256
RULE_COLOR_MIN_MASK_KEPT_PERCENT = 4.0
RULE_COLOR_ACCEPT_CONFIDENCES = {'high', 'medium'}
RULE_COLOR_CONFIDENCE_SCORE = {
    'high': 0.90,
    'medium': 0.60,
    'low': 0.30,
}

# Configure these line coordinates for each camera.
# First crossed line = origin, next different crossed line = destination.
# Used to scale line coordinates when the camera resolution changes.
DEFAULT_CAMERA = 'cctv3'
TRAFFIC_LINE_FRAME_SIZE = (1920, 1080)

CCTV0_TRAFFIC_LINES = [
    {'name': 'down_main', 'points': ((702, 937), (1644, 684))},
    {'name': 'top_main', 'points': ((638, 521), (1200, 446))},
]

CCTV1_TRAFFIC_LINES = [
    {'name': 'top_main',  'points': ((499, 1075), (1862, 470))},
    {'name': 'bottom_main', 'points': ((91, 525), (1503, 358))},
]

CCTV2_TRAFFIC_LINES = [
    {'name': 'left_main', 'points': ((1116, 549), (351, 798))},
    {'name': 'right_main', 'points': ((1168, 874), (1500, 646))},
]

CCTV3_TRAFFIC_LINES = [
    {'name': 'top_main', 'points': ((959, 413), (602, 416))},
    {'name': 'bottom_main', 'points': ((1078, 905), (123, 871))},
    {'name': 'left_branch', 'points': ((388, 516), (329, 571))},
    {'name': 'right_branch', 'points': ((965, 413), (1009, 583))},
]

CCTV4_TRAFFIC_LINES = [
    {'name': 'top_main', 'points': ((1109, 302), (1477, 299))},
    {'name': 'bottom_main', 'points': ((603, 595), (1086, 1079))},
    {'name': 'left_main', 'points': ((1672, 350), (1555, 1079))},
]

CCTV5_TRAFFIC_LINES = [
    {'name': 'main', 'points': ((1488, 525), (0, 1078))},
    {'name': 'left_main', 'points': ((99, 361), (110, 1026))},
    {'name': 'right_main', 'points': ((1382, 403), (767, 267))},
]

CCTV6_TRAFFIC_LINES = [
    {'name': 'right_branch', 'points': ((1149, 494), (1227, 1079))},
    {'name': 'bottom_main', 'points': ((5, 642), (915, 1079))},
    {'name': 'top_main', 'points': ((509, 277), (993, 325))},
]

CCTV7_TRAFFIC_LINES = [
    {'name': 'top_main', 'points': ((75, 368), (765, 272))},
    {'name': 'left_main', 'points': ((1, 649), (528, 573))},
    {'name': 'right_main', 'points': ((623, 901), (1557, 597))},
]

CCTV8_TRAFFIC_LINES = [
    {'name': 'bottom_left_main', 'points': ((262, 543), (0, 1044))},
    {'name': 'top_left_main', 'points': ((359, 366), (298, 461))},
    {'name': 'top_branch', 'points': ((734, 309), (1068, 317))},
    {'name': 'right_main', 'points': ((1912, 794), (1369, 391))},
]

CCTV9_TRAFFIC_LINES = [
    {'name': 'bottom_main', 'points': ((11, 487), (418, 772))},
    {'name': 'right', 'points': ((1816, 569), (1891, 986))},
    {'name': 'center', 'points': ((1277, 512), (1164, 880))},
    {'name': 'top_main', 'points': ((1264, 369), (278, 321))},
]

CAMERA_CONFIGS = {
    'cctv0': {
        'frame_size': TRAFFIC_LINE_FRAME_SIZE,
        'traffic_lines': CCTV0_TRAFFIC_LINES,
    },
    'cctv1': {
        'frame_size': TRAFFIC_LINE_FRAME_SIZE,
        'traffic_lines': CCTV1_TRAFFIC_LINES,
    },
    'cctv2': {
        'frame_size': TRAFFIC_LINE_FRAME_SIZE,
        'traffic_lines': CCTV2_TRAFFIC_LINES,
    },
    'cctv3': {
        'frame_size': TRAFFIC_LINE_FRAME_SIZE,
        'traffic_lines': CCTV3_TRAFFIC_LINES,
    },
    'cctv4': {
        'frame_size': TRAFFIC_LINE_FRAME_SIZE,
        'traffic_lines': CCTV4_TRAFFIC_LINES,
    },
    'cctv5': {
        'frame_size': TRAFFIC_LINE_FRAME_SIZE,
        'traffic_lines': CCTV5_TRAFFIC_LINES,
    },
    'cctv6': {
        'frame_size': TRAFFIC_LINE_FRAME_SIZE,
        'traffic_lines': CCTV6_TRAFFIC_LINES,
    },
    'cctv7': {
        'frame_size': TRAFFIC_LINE_FRAME_SIZE,
        'traffic_lines': CCTV7_TRAFFIC_LINES,
    },
    'cctv8': {
        'frame_size': TRAFFIC_LINE_FRAME_SIZE,
        'traffic_lines': CCTV8_TRAFFIC_LINES,
    },
    'cctv9': {
        'frame_size': TRAFFIC_LINE_FRAME_SIZE,
        'traffic_lines': CCTV9_TRAFFIC_LINES,
    },
}

# คลาส COCO ที่เป็นรถ
VEHICLE_CLASSES = {
    0: 'car',
    1: 'motorbike',
    2: 'bus',
    3: 'truck'
}

# สร้างโฟลเดอร์สำหรับบันทึก
class AsyncImageWriter:
    def __init__(self, enabled=True, maxsize=SAVE_QUEUE_SIZE, jpeg_quality=JPEG_QUALITY):
        self.enabled = enabled
        self.jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        self.queue = None
        self.worker = None
        if self.enabled:
            self.queue = queue.Queue(maxsize=maxsize)
            self.worker = threading.Thread(target=self._run, daemon=True)
            self.worker.start()

    def _run(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                filename, image = item
                cv2.imwrite(filename, image, self.jpeg_params)
            finally:
                self.queue.task_done()

    def write(self, filename, image):
        if image is None or image.size == 0:
            return False
        image_to_write = image.copy()
        if not self.enabled:
            return cv2.imwrite(filename, image_to_write, self.jpeg_params)
        try:
            self.queue.put_nowait((filename, image_to_write))
            return True
        except queue.Full:
            return cv2.imwrite(filename, image_to_write, self.jpeg_params)

    def close(self):
        if not self.enabled:
            return
        self.queue.put(None)
        self.queue.join()
        self.worker.join(timeout=5)


class MultiTaskMobileNetV3(nn.Module):
    def __init__(self, num_brand_classes, num_color_classes, dropout=0.2):
        super().__init__()
        base = models.mobilenet_v3_small(weights=None)
        self.backbone = nn.Module()
        self.backbone.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        in_features = base.classifier[0].in_features
        self.brand_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_brand_classes),
        )
        self.color_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_color_classes),
        )

    def forward(self, image):
        features = self.backbone.features(image)
        pooled = self.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        return {
            'brand': self.brand_head(flattened),
            'color': self.color_head(flattened),
        }


class MultiTaskCropClassifier:
    def __init__(
        self,
        model_path,
        labels_path=None,
        input_size=CLASSIFIER_INPUT_SIZE,
        device_name=CLASSIFIER_DEVICE
    ):
        self.enabled = False
        self.model = None
        self.input_size = input_size
        self.device = self._select_device(device_name)
        self.label_maps = {'brand': {}, 'color': {}}
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if not model_path:
            return
        if not os.path.exists(model_path):
            print(f"Warning: multitask classifier model not found: {model_path}")
            return

        try:
            checkpoint = self._load_checkpoint(model_path)
            state_dict = self._get_state_dict(checkpoint)
            brand_count = int(state_dict['brand_head.1.bias'].numel())
            color_count = int(state_dict['color_head.1.bias'].numel())
            label_lists = self._load_label_maps(labels_path, checkpoint, brand_count, color_count)
            self.label_maps = {
                'brand': {index: label for index, label in enumerate(label_lists['brand'])},
                'color': {index: label for index, label in enumerate(label_lists['color'])},
            }
            self.input_size = self._resolve_input_size(checkpoint, input_size)
            self.model = MultiTaskMobileNetV3(brand_count, color_count)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
            self.model.eval()
            self.enabled = True
            print(
                f"Multi-task classifier: {model_path} | labels: {labels_path or 'checkpoint'} | "
                f"input_size: {self.input_size} | device: {self.device}"
            )
        except Exception as exc:
            print(f"Warning: multitask classifier init failed; crop classification disabled: {exc}")

    @staticmethod
    def _select_device(device_name):
        if device_name == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device_name)
        if device.type == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA classifier device requested but unavailable; using CPU.")
            return torch.device('cpu')
        return device

    @staticmethod
    def _load_checkpoint(model_path):
        try:
            return torch.load(model_path, map_location='cpu', weights_only=True)
        except TypeError:
            return torch.load(model_path, map_location='cpu')

    @staticmethod
    def _get_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ('model_state', 'model_state_dict', 'state_dict'):
                state_dict = checkpoint.get(key)
                if isinstance(state_dict, dict):
                    break
            else:
                state_dict = checkpoint
        else:
            raise ValueError('unsupported checkpoint format')

        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
        return state_dict

    @staticmethod
    def _resolve_input_size(checkpoint, default_size):
        args = checkpoint.get('args') if isinstance(checkpoint, dict) else None
        if isinstance(args, dict) and args.get('image_size'):
            return int(args['image_size'])
        for key in ('image_size', 'img_size'):
            if isinstance(checkpoint, dict) and checkpoint.get(key):
                return int(checkpoint[key])
        return int(default_size)

    def _load_label_maps(self, labels_path, checkpoint, brand_count, color_count):
        labels = self._normalize_label_maps(checkpoint.get('label_maps') if isinstance(checkpoint, dict) else None)
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as file:
                labels.update(self._normalize_label_maps(json.load(file)))
        elif labels_path:
            fallback_path = 'label_maps.json' if labels_path == 'label_map.json' else None
            if fallback_path and os.path.exists(fallback_path):
                with open(fallback_path, 'r', encoding='utf-8') as file:
                    labels.update(self._normalize_label_maps(json.load(file)))
            else:
                print(f"Warning: classifier labels not found: {labels_path}; using checkpoint labels.")

        labels.setdefault('brand', [str(index) for index in range(brand_count)])
        labels.setdefault('color', [str(index) for index in range(color_count)])
        if len(labels['brand']) != brand_count:
            raise ValueError(f"brand label count mismatch: {len(labels['brand'])} != {brand_count}")
        if len(labels['color']) != color_count:
            raise ValueError(f"color label count mismatch: {len(labels['color'])} != {color_count}")
        return labels

    @staticmethod
    def _labels_from_mapping(value):
        if isinstance(value, list):
            return [str(item) for item in value]
        if not isinstance(value, dict):
            return None

        index_pairs = []
        for key, label in value.items():
            try:
                index_pairs.append((int(key), str(label)))
            except (TypeError, ValueError):
                index_pairs = []
                break
        if index_pairs:
            return [label for _, label in sorted(index_pairs)]

        class_pairs = []
        for label, index in value.items():
            try:
                class_pairs.append((int(index), str(label)))
            except (TypeError, ValueError):
                return None
        return [label for _, label in sorted(class_pairs)] if class_pairs else None

    @classmethod
    def _normalize_label_maps(cls, data):
        if not isinstance(data, dict):
            return {}
        labels = {}
        for key in ('brand', 'color'):
            direct = cls._labels_from_mapping(data.get(key))
            if direct:
                labels[key] = direct

        aliases = {
            'brand': ('brands', 'idx_to_brand', 'brand_idx_to_label'),
            'color': ('colors', 'idx_to_color', 'color_idx_to_label'),
        }
        for target_key, alias_keys in aliases.items():
            if target_key in labels:
                continue
            for alias_key in alias_keys:
                alias = cls._labels_from_mapping(data.get(alias_key))
                if alias:
                    labels[target_key] = alias
                    break
        return labels

    def predict(self, crop_img):
        if not self.enabled or crop_img is None or crop_img.size == 0:
            return 'unknown', 0.0, 'unknown', 0.0

        try:
            image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            tensor = image.astype(np.float32) / 255.0
            tensor = (tensor - self.mean) / self.std
            tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :].astype(np.float32)
            tensor = torch.from_numpy(tensor).to(self.device)
            with torch.inference_mode():
                outputs = self.model(tensor)
                brand_probs = torch.softmax(outputs['brand'], dim=1)[0].detach().cpu().numpy()
                color_probs = torch.softmax(outputs['color'], dim=1)[0].detach().cpu().numpy()
        except Exception as exc:
            print(f"\nWarning: multitask crop classifier failed once and is now disabled: {exc}")
            self.enabled = False
            return 'unknown', 0.0, 'unknown', 0.0

        brand_index = int(np.argmax(brand_probs))
        color_index = int(np.argmax(color_probs))
        brand = self.label_maps['brand'].get(brand_index, str(brand_index))
        color = self.label_maps['color'].get(color_index, str(color_index))
        return brand, float(brand_probs[brand_index]), color, float(color_probs[color_index])


def parse_source(value):
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def is_live_source(source):
    parsed = parse_source(source)
    if isinstance(parsed, int):
        return True
    text = str(source).strip().lower()
    return text.startswith(('rtsp://', 'rtsps://', 'http://', 'https://', 'udp://', 'tcp://'))


def open_video_capture(source):
    cap = cv2.VideoCapture(parse_source(source))
    if CAPTURE_BUFFER_SIZE > 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAPTURE_BUFFER_SIZE)
    return cap


def open_capture_with_retries(source, live_source, retries=LIVE_SOURCE_INITIAL_RETRIES):
    attempts = max(1, retries if live_source else 1)
    for attempt in range(1, attempts + 1):
        cap = open_video_capture(source)
        if cap.isOpened():
            return cap
        cap.release()
        if not live_source or attempt >= attempts:
            break
        print(f"Cannot open source: {source} | retry {attempt}/{attempts}")
        time.sleep(LIVE_RECONNECT_DELAY_SEC)
    return None


def read_initial_frame(cap, source, live_source, retries=LIVE_SOURCE_INITIAL_RETRIES):
    attempts = max(1, retries if live_source else 1)
    for attempt in range(1, attempts + 1):
        ret, frame = cap.read() if cap is not None and cap.isOpened() else (False, None)
        if ret and is_valid_frame(frame):
            return cap, frame
        if not live_source or attempt >= attempts:
            break
        if cap is not None:
            cap.release()
        print(f"Cannot read first frame: {source} | retry {attempt}/{attempts}")
        time.sleep(LIVE_RECONNECT_DELAY_SEC)
        cap = open_video_capture(source)
    return cap, None


def format_runtime_path(path_template, camera_name):
    if not path_template:
        return path_template
    try:
        return path_template.format(camera=camera_name)
    except (KeyError, ValueError):
        return path_template


def ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def load_classifier_labels(labels_path):
    if not labels_path or not os.path.exists(labels_path):
        print(f"Warning: classifier labels not found: {labels_path}")
        return {}
    with open(labels_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    raw_labels = data.get('idx_to_label') or data.get('idx_to_class') or data
    return {int(index): label for index, label in raw_labels.items()}


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / max(float(np.sum(exp)), 1e-12)


def safe_filename_part(value):
    value = str(value).strip().replace(' ', '-')
    safe_chars = []
    for char in value:
        if char.isalnum() or char in ('-', '_'):
            safe_chars.append(char)
    return ''.join(safe_chars) or 'unknown'


def is_jetson_device():
    if os.path.exists('/etc/nv_tegra_release'):
        return True
    model_path = '/proc/device-tree/model'
    if os.path.exists(model_path):
        try:
            with open(model_path, 'r', encoding='utf-8', errors='ignore') as file:
                return 'jetson' in file.read().lower()
        except OSError:
            return False
    return False


def resolve_model_path(model_path, require_engine_on_jetson=REQUIRE_ENGINE_ON_JETSON):
    if require_engine_on_jetson and is_jetson_device() and not model_path.lower().endswith('.engine'):
        raise RuntimeError(
            f"Jetson mode requires a TensorRT .engine model, got: {model_path}. "
            f"Export/copy {ENGINE_MODEL_PATH} or run with --allow-pt-on-jetson for testing only."
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path


def scale_traffic_lines(traffic_lines, frame_shape, base_size=TRAFFIC_LINE_FRAME_SIZE):
    if not base_size:
        return traffic_lines
    base_w, base_h = base_size
    frame_h, frame_w = frame_shape[:2]
    if base_w <= 0 or base_h <= 0 or (base_w == frame_w and base_h == frame_h):
        return traffic_lines
    scale_x = frame_w / base_w
    scale_y = frame_h / base_h
    scaled_lines = []
    for line in traffic_lines:
        scaled_points = []
        for x, y in line['points']:
            scaled_points.append((int(round(x * scale_x)), int(round(y * scale_y))))
        scaled_lines.append({'name': line['name'], 'points': tuple(scaled_points)})
    return scaled_lines


def get_traffic_lines_roi(traffic_lines, frame_shape, margin=TRAFFIC_LINE_ROI_MARGIN):
    if not traffic_lines:
        return None
    frame_h, frame_w = frame_shape[:2]
    points = []
    for line in traffic_lines:
        points.extend(line['points'])
    points = np.array(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(points)
    return (
        max(0, x - margin),
        max(0, y - margin),
        min(frame_w, x + w + margin),
        min(frame_h, y + h + margin),
    )


def estimate_color_from_crop(crop_img, class_name):
    if not ENABLE_COLOR_DETECTION or class_name not in COLOR_ENABLED_CLASSES:
        return 'unknown', 0.0, 0.0
    if USE_RULE_BASED_COLOR_CLASSIFIER and rule_predict_vehicle_color is not None:
        try:
            result = rule_predict_vehicle_color(crop_img, max_width=RULE_COLOR_MAX_WIDTH)
            color = str(result.get('color') or 'unknown').lower()
            confidence_label = str(result.get('confidence') or 'low').lower()
            mask_kept_percent = float(result.get('mask_kept_percent') or 0.0)
            color_conf = RULE_COLOR_CONFIDENCE_SCORE.get(confidence_label, 0.0)
            if (
                color != 'unknown'
                and confidence_label in RULE_COLOR_ACCEPT_CONFIDENCES
                and mask_kept_percent >= RULE_COLOR_MIN_MASK_KEPT_PERCENT
            ):
                return color, color_conf, 0.0
        except Exception as exc:
            print(f"\nWarning: rule-based color classifier failed; fallback HSV color used: {exc}")
    return estimate_vehicle_color(crop_img, class_name, None)


def classify_vehicle_attributes_if_enabled(
    crop_classifier,
    crop_img,
    class_name,
    enabled_classes=CLASSIFIER_ENABLED_CLASSES,
    min_confidence=CLASSIFIER_MIN_CONFIDENCE
):
    if class_name not in enabled_classes:
        return 'unknown', 0.0, 'unknown', 0.0
    if crop_classifier is None or not crop_classifier.enabled:
        return 'unknown', 0.0, 'unknown', 0.0
    brand, brand_conf, color, color_conf = crop_classifier.predict(crop_img)
    if brand_conf < min_confidence:
        brand = 'unknown'
    if color_conf < min_confidence:
        color = 'unknown'
    return brand, brand_conf, color, color_conf


def should_print_frame(frame_count, print_every):
    return print_every <= 1 or frame_count % print_every == 0


def is_valid_frame(frame):
    return frame is not None and hasattr(frame, 'size') and frame.size > 0


# --- Simple Centroid Tracker to avoid recounting same vehicle across frames ---
class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=120):
        self.nextObjectID = 0
        self.objects = {}           # objectID -> centroid (x, y)
        self.bboxes = {}            # objectID -> bbox (x1, y1, x2, y2)
        self.detections = {}        # objectID -> latest detection dict
        self.disappeared = {}       # objectID -> frames disappeared
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        self.totalCount = 0         # unique objects seen
        self.last_deregistered = []

    def register(self, detection):
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        cX = (x1 + x2) // 2
        cY = (y1 + y2) // 2
        objectID = self.nextObjectID
        self.objects[objectID] = (cX, cY)
        self.bboxes[objectID] = bbox
        self.detections[objectID] = detection
        self.disappeared[objectID] = 0
        self.nextObjectID += 1
        self.totalCount += 1
        return objectID

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.bboxes[objectID]
        del self.detections[objectID]
        del self.disappeared[objectID]
        self.last_deregistered.append(objectID)

    def update(self, detections):
        """
        detections: list of dicts with bbox=(x1, y1, x2, y2)
        returns: (tracked_detections, newly_registered_ids)
        """
        self.last_deregistered = []
        newly_registered = []
        if len(detections) == 0:
            # mark existing as disappeared
            to_deregister = []
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    to_deregister.append(objectID)
            for oid in to_deregister:
                self.deregister(oid)
            return self.detections.copy(), newly_registered

        inputCentroids = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cX = (x1 + x2) // 2
            cY = (y1 + y2) // 2
            inputCentroids.append((cX, cY))

        # If no existing objects, register all
        if len(self.objects) == 0:
            for detection in detections:
                oid = self.register(detection)
                newly_registered.append(oid)
            return self.detections.copy(), newly_registered

        # Otherwise, match detections to existing objects using distance plus IoU.
        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())
        objectBboxes = [self.bboxes[objectID] for objectID in objectIDs]

        D = []
        IOU = []
        for row_index, oc in enumerate(objectCentroids):
            row = []
            iou_row = []
            for col_index, ic in enumerate(inputCentroids):
                d = (oc[0] - ic[0]) ** 2 + (oc[1] - ic[1]) ** 2
                row.append(d)
                iou_row.append(bbox_iou(objectBboxes[row_index], detections[col_index]['bbox']))
            D.append(row)
            IOU.append(iou_row)

        # Greedy matching
        usedRows = set()
        usedCols = set()
        matches = []  # list of (row, col)
        match_candidates = []
        for r in range(len(D)):
            for c in range(len(D[r])):
                distance_score = D[r][c] / max(1, self.maxDistance ** 2)
                cost = distance_score - (IOU[r][c] * 0.75)
                match_candidates.append((cost, D[r][c], IOU[r][c], r, c))
        match_candidates.sort(key=lambda x: x[0])

        for (_, dist, iou, r, c) in match_candidates:
            if r in usedRows or c in usedCols:
                continue
            if dist > self.maxDistance ** 2 and iou < 0.15:
                continue
            usedRows.add(r)
            usedCols.add(c)
            matches.append((r, c))

        unmatchedRows = set(range(0, len(objectCentroids))) - usedRows
        unmatchedCols = set(range(0, len(inputCentroids))) - usedCols

        # Update matched
        for (r, c) in matches:
            objectID = objectIDs[r]
            cX, cY = inputCentroids[c]
            self.objects[objectID] = (cX, cY)
            self.bboxes[objectID] = detections[c]['bbox']
            self.detections[objectID] = detections[c]
            self.disappeared[objectID] = 0

        # increase disappeared for unmatched rows
        for r in unmatchedRows:
            objectID = objectIDs[r]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.maxDisappeared:
                self.deregister(objectID)

        # register unmatched incoming boxes
        for c in unmatchedCols:
            oid = self.register(detections[c])
            newly_registered.append(oid)

        return self.detections.copy(), newly_registered

# ---------------------------------------------------------------------------

# ==========================================
# Geometry helpers
# ==========================================


def expand_bbox(bbox, frame_shape, padding_x, padding_y=None, square=False, min_size=0):
    x1, y1, x2, y2 = bbox
    if padding_y is None:
        padding_y = padding_x

    height, width = frame_shape[:2]
    x1 = x1 - padding_x
    y1 = y1 - padding_y
    x2 = x2 + padding_x
    y2 = y2 + padding_y

    crop_w = x2 - x1
    crop_h = y2 - y1
    target_w = max(crop_w, min_size)
    target_h = max(crop_h, min_size)
    if square:
        target_w = target_h = max(target_w, target_h)

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    x1 = int(round(center_x - target_w / 2))
    y1 = int(round(center_y - target_h / 2))
    x2 = int(round(center_x + target_w / 2))
    y2 = int(round(center_y + target_h / 2))

    return (
        max(0, x1),
        max(0, y1),
        min(width, x2),
        min(height, y2)
    )

def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0

def bbox_intersection_over_smaller(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    smaller = min(area_a, area_b)
    return intersection / smaller if smaller > 0 else 0

def bbox_center_inside(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    cx = (ix1 + ix2) / 2
    cy = (iy1 + iy2) / 2
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2

def is_vehicle_like_box(detection):
    x1, y1, x2, y2 = detection['bbox']
    width = x2 - x1
    height = y2 - y1
    if width < MIN_BOX_WIDTH or height < MIN_BOX_HEIGHT:
        return False

    aspect = width / max(1, height)
    class_id = detection['class_id']
    confidence = detection['confidence']

    if class_id in (5, 7) and confidence < MIN_TRUCK_BUS_CONF:
        return False
    if class_id in (2, 5, 7):
        return 0.85 <= aspect <= 5.5
    if class_id == 3:
        return 0.35 <= aspect <= 4.5
    return True




def filter_detections(detections, iou_threshold=0.65):
    filtered = []
    for detection in detections:
        if not is_vehicle_like_box(detection):
            continue
        filtered.append(detection)

    filtered.sort(
        key=lambda item: (
            item['confidence'],
            (item['bbox'][2] - item['bbox'][0]) * (item['bbox'][3] - item['bbox'][1])
        ),
        reverse=True
    )
    kept = []
    for detection in filtered:
        duplicate = False
        for kept_detection in kept:
            same_class = detection['class_id'] == kept_detection['class_id']
            iou = bbox_iou(detection['bbox'], kept_detection['bbox'])
            covered = bbox_intersection_over_smaller(detection['bbox'], kept_detection['bbox'])
            center_inside = bbox_center_inside(detection['bbox'], kept_detection['bbox'])
            if same_class and (iou >= iou_threshold or covered >= 0.35 or center_inside):
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


def offset_detection_bbox(detection, offset_x, offset_y):
    x1, y1, x2, y2 = detection['bbox']
    detection['bbox'] = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
    return detection

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def _orientation(a, b, c):
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else 2

def _on_segment(a, b, c):
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )

def segments_intersect(a, b, c, d):
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(a, c, b):
        return True
    if o2 == 0 and _on_segment(a, d, b):
        return True
    if o3 == 0 and _on_segment(c, a, d):
        return True
    if o4 == 0 and _on_segment(c, b, d):
        return True
    return False

def crossed_line(previous_center, current_center, line_points, min_move=LINE_CROSS_MIN_MOVE):
    if previous_center is None or current_center is None:
        return False
    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    if (dx * dx + dy * dy) < (min_move * min_move):
        return False
    return segments_intersect(previous_center, current_center, line_points[0], line_points[1])

def point_to_segment_distance(point, a, b):
    px, py = point
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = ((px - ax) * dx + (py - ay) * dy) / float(dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5

def bbox_touches_line(bbox, line_points, tolerance=LINE_HIT_TOLERANCE):
    x1, y1, x2, y2 = bbox
    a, b = line_points
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    edges = list(zip(corners, corners[1:] + corners[:1]))

    for edge_start, edge_end in edges:
        if segments_intersect(edge_start, edge_end, a, b):
            return True

    if x1 - tolerance <= a[0] <= x2 + tolerance and y1 - tolerance <= a[1] <= y2 + tolerance:
        return True
    if x1 - tolerance <= b[0] <= x2 + tolerance and y1 - tolerance <= b[1] <= y2 + tolerance:
        return True

    center = bbox_center(bbox)
    if point_to_segment_distance(center, a, b) <= tolerance:
        return True
    return False

def hit_traffic_line(previous_center, current_center, bbox, line_points, tolerance=LINE_HIT_TOLERANCE):
    return crossed_line(previous_center, current_center, line_points) or bbox_touches_line(bbox, line_points, tolerance)

def draw_traffic_lines(frame, traffic_lines):
    colors = [(0, 255, 255), (255, 180, 0), (255, 0, 255), (0, 200, 255), (180, 255, 0)]
    for index, line in enumerate(traffic_lines):
        color = colors[index % len(colors)]
        p1, p2 = line['points']
        cv2.line(frame, p1, p2, color, 3)
        cv2.circle(frame, p1, 5, color, -1)
        cv2.circle(frame, p2, 5, color, -1)
        cv2.putText(
            frame,
            line['name'],
            (p1[0], max(20, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

def route_counts(route_states, completed_routes):
    entered = sum(1 for state in route_states.values() if state.get('origin') is not None)
    return entered, completed_routes

def fit_display_frame(frame, max_width=DISPLAY_MAX_WIDTH, max_height=DISPLAY_MAX_HEIGHT):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

def clean_foreground_mask(mask):
    if mask is None or mask.size == 0:
        return None
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def shrink_bbox_for_color(bbox, frame_shape, shrink_x=COLOR_CROP_SHRINK_X, shrink_y=COLOR_CROP_SHRINK_Y):
    x1, y1, x2, y2 = bbox
    height, width = frame_shape[:2]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    dx = int(box_w * shrink_x)
    dy = int(box_h * shrink_y)
    return (
        max(0, x1 + dx),
        max(0, y1 + dy),
        min(width, x2 - dx),
        min(height, y2 - dy)
    )

def prepare_color_debug_image(crop_img, class_name='vehicle', fg_mask=None):
    if crop_img is None or crop_img.size == 0:
        return None

    height, width = crop_img.shape[:2]
    if height < 4 or width < 4:
        return None

    if class_name == 'motorbike':
        y1 = int(height * 0.35)
        y2 = int(height * 0.95)
        x1 = int(width * 0.08)
        x2 = int(width * 0.92)
        size = (72, 72)
        kernel = (15, 15)
    else:
        y1 = int(height * 0.12)
        y2 = int(height * 0.88)
        x1 = int(width * 0.10)
        x2 = int(width * 0.90)
        size = (64, 64)
        kernel = (21, 21)

    roi = crop_img[y1:y2, x1:x2]
    if roi.size == 0:
        roi = crop_img
    roi = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
    roi = cv2.GaussianBlur(roi, kernel, 0)
    if fg_mask is not None and fg_mask.size:
        mask_roi = fg_mask[y1:y2, x1:x2]
        if mask_roi.size:
            mask_roi = cv2.resize(mask_roi, size, interpolation=cv2.INTER_NEAREST)
            roi = cv2.bitwise_and(roi, roi, mask=mask_roi)
    return roi

def estimate_vehicle_color(crop_img, class_name='vehicle', fg_mask=None):
    if crop_img is None or crop_img.size == 0:
        return 'unknown', 0.0, 0.0

    height, width = crop_img.shape[:2]
    if height < 4 or width < 4:
        return 'unknown', 0.0, 0.0

    if class_name == 'motorbike':
        return estimate_motorbike_color(crop_img, fg_mask)

    y1 = int(height * 0.12)
    y2 = int(height * 0.88)
    x1 = int(width * 0.10)
    x2 = int(width * 0.90)
    roi = crop_img[y1:y2, x1:x2]
    if roi.size == 0:
        roi = crop_img
        fg_roi = fg_mask
    else:
        fg_roi = fg_mask[y1:y2, x1:x2] if fg_mask is not None and fg_mask.size else None

    roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
    if fg_roi is not None and fg_roi.size:
        fg_roi = cv2.resize(fg_roi, (64, 64), interpolation=cv2.INTER_NEAREST)
    roi = cv2.GaussianBlur(roi, (21, 21), 0)
    mask_shape = np.zeros((64, 64), dtype=np.uint8)
    cv2.ellipse(mask_shape, (32, 32), (25, 22), 0, 0, 360, 255, -1)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    combined_mask = mask_shape > 0
    if fg_roi is not None and np.count_nonzero(fg_roi) > 20:
        combined_mask = combined_mask & (fg_roi > 0)
    h = hsv[:, :, 0][combined_mask].reshape(-1)
    s = hsv[:, :, 1][combined_mask].reshape(-1)
    v = hsv[:, :, 2][combined_mask].reshape(-1)

    base_valid = v > 25
    if not np.any(base_valid):
        return 'black', 0.9, 1.0

    h = h[base_valid]
    s = s[base_valid]
    v = v[base_valid]

    median_v = float(np.median(v))
    shadow_threshold = max(35.0, median_v * 0.62)
    shadow_mask = (v < shadow_threshold) & (s < 95)
    highlight_mask = (v > 245) & (s < 30)
    clean_mask = (~shadow_mask) & (~highlight_mask) & (v > 40)
    shadow_ratio = float(np.mean(shadow_mask))
    raw_median_s = int(np.median(s))
    raw_median_v = int(np.median(v))

    if np.count_nonzero(clean_mask) >= max(24, int(len(v) * 0.18)):
        h_clean = h[clean_mask]
        s_clean = s[clean_mask]
        v_clean = v[clean_mask]
    else:
        h_clean = h
        s_clean = s
        v_clean = v

    dark_ratio = float(np.mean(v < 85))
    very_dark_ratio = float(np.mean(v < 55))
    if dark_ratio >= BODY_BLACK_RATIO_OVERRIDE and raw_median_v < 175 and raw_median_s < 95:
        return 'black', max(dark_ratio, very_dark_ratio), shadow_ratio

    color_name, color_conf = dominant_body_color(
        h_clean,
        s_clean,
        v_clean,
        dark_ratio,
        very_dark_ratio,
        raw_median_s,
        raw_median_v
    )
    color_name, color_conf = neutral_body_override(crop_img, color_name, color_conf)
    color_name, color_conf = apply_color_quality_gate(color_name, color_conf, shadow_ratio)
    return color_name, color_conf, shadow_ratio

def neutral_body_override(crop_img, color_name, color_conf):
    if color_name in ('black', 'white', 'silver', 'gray', 'unknown'):
        return color_name, color_conf
    if color_conf >= BODY_STRONG_CHROMA_CONF:
        return color_name, color_conf

    neutral_color, neutral_conf, neutral_chroma = estimate_neutral_body_color(crop_img)
    if neutral_conf >= BODY_NEUTRAL_OVERRIDE_CONF and neutral_chroma < BODY_COLOR_MIN_CHROMA_RATIO:
        return neutral_color, max(neutral_conf, color_conf)
    return color_name, color_conf

def estimate_neutral_body_color(crop_img):
    height, width = crop_img.shape[:2]
    rois = [
        crop_img[int(height * 0.06):int(height * 0.42), int(width * 0.08):int(width * 0.92)],
        crop_img[int(height * 0.08):int(height * 0.70), int(width * 0.05):int(width * 0.28)],
        crop_img[int(height * 0.08):int(height * 0.70), int(width * 0.72):int(width * 0.95)],
    ]

    best_color = 'unknown'
    best_conf = 0.0
    best_chroma = 1.0
    for roi in rois:
        if roi.size == 0:
            continue
        roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
        roi = cv2.GaussianBlur(roi, (15, 15), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].reshape(-1)
        v = hsv[:, :, 2].reshape(-1)
        valid = v > 35
        if not np.any(valid):
            continue
        s = s[valid]
        v = v[valid]
        chroma_ratio = float(np.mean((s >= 58) & (v >= 50)))
        scores = {
            'white': float(np.mean((s < 65) & (v >= 160))),
            'black': float(np.mean(v < 85)),
            'silver': float(np.mean((s < 70) & (v >= 115) & (v < 175))),
            'gray': float(np.mean((s < 75) & (v >= 85) & (v < 160))),
        }
        color, conf = max(scores.items(), key=lambda item: item[1])
        if conf > best_conf:
            best_color = color
            best_conf = conf
            best_chroma = chroma_ratio
    return best_color, best_conf, best_chroma

def dominant_body_color(
    h_values,
    s_values,
    v_values,
    dark_ratio=0.0,
    very_dark_ratio=0.0,
    raw_median_s=0,
    raw_median_v=255
):
    if len(v_values) == 0:
        return 'unknown', 0.0

    chroma_mask = (s_values >= 58) & (v_values >= 50)
    chroma_ratio = float(np.mean(chroma_mask))

    neutral_masks = {
        'white': (s_values < 55) & (v_values >= 175),
        'black': v_values < 85,
        'silver': (s_values < 60) & (v_values >= 120) & (v_values < 175),
        'gray': (s_values < 70) & (v_values >= 85) & (v_values < 160),
    }
    neutral_scores = {
        color: float(np.mean(mask))
        for color, mask in neutral_masks.items()
    }
    neutral_color, neutral_conf = max(neutral_scores.items(), key=lambda item: item[1])

    if neutral_conf >= BODY_COLOR_MIN_NEUTRAL_RATIO and chroma_ratio < 0.45:
        return neutral_color, neutral_conf

    color_name, color_conf = dominant_color_blur_median(
        h_values,
        s_values,
        v_values,
        dark_ratio,
        very_dark_ratio,
        raw_median_s,
        raw_median_v
    )

    if color_name not in ('black', 'white', 'silver', 'gray', 'unknown'):
        if chroma_ratio < BODY_COLOR_MIN_CHROMA_RATIO:
            if neutral_conf >= 0.20:
                return neutral_color, neutral_conf
            return 'unknown', color_conf

    return color_name, color_conf

def apply_color_quality_gate(color_name, color_conf, shadow_ratio):
    if color_name == 'unknown':
        return color_name, color_conf
    if color_conf < COLOR_MIN_CONFIDENCE and shadow_ratio > COLOR_LOW_CONF_SHADOW_LIMIT:
        return 'unknown', color_conf
    if color_conf < 0.10:
        return 'unknown', color_conf
    return color_name, color_conf

def dominant_color_blur_median(
    h_values,
    s_values,
    v_values,
    dark_ratio=0.0,
    very_dark_ratio=0.0,
    raw_median_s=0,
    raw_median_v=255
):
    if len(v_values) == 0:
        return 'unknown', 0.0

    median_h = int(np.median(h_values))
    median_s = int(np.median(s_values))
    median_v = int(np.median(v_values))

    if median_s < 45 and median_v >= 185:
        white_mask = (s_values < 55) & (v_values >= 175)
        return 'white', float(np.mean(white_mask)) if len(white_mask) else 0.0

    if (
        dark_ratio >= 0.30
        or very_dark_ratio >= 0.18
        or (raw_median_v < 135 and raw_median_s < 75)
        or (median_v < 140 and median_s < 70 and dark_ratio >= 0.18)
    ):
        return 'black', max(dark_ratio, very_dark_ratio)

    color = map_hsv_to_color(median_h, median_s, median_v)

    if color == 'unknown':
        return color, 0.0

    if color in ('black', 'white', 'silver', 'gray'):
        same_mask = np.array([map_hsv_to_color(int(h), int(s), int(v)) == color for h, s, v in zip(h_values, s_values, v_values)])
    else:
        same_mask = color_membership_mask(color, h_values, s_values, v_values)
    confidence = float(np.mean(same_mask)) if len(same_mask) else 0.0
    return color, confidence

def estimate_motorbike_color(crop_img, fg_mask=None):
    height, width = crop_img.shape[:2]
    roi_specs = [
        ('lower_full', 0.08, 0.35, 0.92, 0.95),
        ('lower_left', 0.05, 0.38, 0.48, 0.95),
        ('lower_right', 0.52, 0.38, 0.95, 0.95),
        ('front_body', 0.45, 0.28, 0.98, 0.88),
        ('rear_body', 0.02, 0.28, 0.55, 0.88),
    ]
    candidates = []
    for _, rx1, ry1, rx2, ry2 in roi_specs:
        x1 = int(width * rx1)
        y1 = int(height * ry1)
        x2 = int(width * rx2)
        y2 = int(height * ry2)
        roi = crop_img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        fg_roi = fg_mask[y1:y2, x1:x2] if fg_mask is not None and fg_mask.size else None
        color, conf, shadow_ratio, quality, chroma_ratio = estimate_motorbike_roi_color(roi, fg_roi)
        candidates.append((quality, color, conf, shadow_ratio, chroma_ratio))

    if not candidates:
        return 'unknown', 0.0, 0.0

    candidates.sort(key=lambda item: item[0], reverse=True)
    quality, color, conf, shadow_ratio, chroma_ratio = candidates[0]
    if quality < MOTORBIKE_MIN_COLOR_QUALITY:
        return 'unknown', conf, shadow_ratio
    if color not in ('black', 'white', 'silver', 'gray', 'unknown') and chroma_ratio < MOTORBIKE_MIN_CHROMA_RATIO:
        return 'unknown', conf, shadow_ratio
    color, conf = apply_color_quality_gate(color, conf, shadow_ratio)
    return color, conf, shadow_ratio

def estimate_motorbike_roi_color(roi, fg_mask=None):
    roi = cv2.resize(roi, (72, 72), interpolation=cv2.INTER_AREA)
    if fg_mask is not None and fg_mask.size:
        fg_mask = cv2.resize(fg_mask, (72, 72), interpolation=cv2.INTER_NEAREST)
    roi = cv2.GaussianBlur(roi, (15, 15), 0)
    mask_shape = np.zeros((72, 72), dtype=np.uint8)
    cv2.ellipse(mask_shape, (36, 40), (31, 24), 0, 0, 360, 255, -1)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    combined_mask = mask_shape > 0
    if fg_mask is not None and np.count_nonzero(fg_mask) > 20:
        combined_mask = combined_mask & (fg_mask > 0)
    h = hsv[:, :, 0][combined_mask].reshape(-1)
    s = hsv[:, :, 1][combined_mask].reshape(-1)
    v = hsv[:, :, 2][combined_mask].reshape(-1)

    valid = v > 28
    if not np.any(valid):
        return 'black', 0.9, 1.0, 0.0, 0.0

    h = h[valid]
    s = s[valid]
    v = v[valid]

    median_v = float(np.median(v))
    shadow_mask = (v < max(32.0, median_v * 0.55)) & (s < 95)
    highlight_mask = (v > 245) & (s < 30)
    usable = (~shadow_mask) & (~highlight_mask) & (v > 35)
    shadow_ratio = float(np.mean(shadow_mask))

    if np.count_nonzero(usable) >= max(20, int(len(v) * 0.15)):
        h_use = h[usable]
        s_use = s[usable]
        v_use = v[usable]
    else:
        h_use = h
        s_use = s
        v_use = v

    chroma = (s_use >= 50) & (v_use >= 45)
    chroma_ratio = float(np.mean(chroma)) if len(chroma) else 0.0
    if np.count_nonzero(chroma) >= max(16, int(len(h_use) * 0.07)):
        color, conf = dominant_chroma_color(h_use[chroma], s_use[chroma], v_use[chroma])
        if color != 'unknown' and conf >= 0.18:
            final_conf = chroma_ratio * conf
            quality = final_conf + (chroma_ratio * MOTORBIKE_CHROMA_BONUS) - (shadow_ratio * 0.15)
            return color, final_conf, shadow_ratio, quality, chroma_ratio

    dark_ratio = float(np.mean(v < 85))
    very_dark_ratio = float(np.mean(v < 55))
    color, conf = dominant_color_blur_median(
        h_use,
        s_use,
        v_use,
        dark_ratio=dark_ratio,
        very_dark_ratio=very_dark_ratio,
        raw_median_s=int(np.median(s)),
        raw_median_v=int(np.median(v))
    )
    quality = conf - (shadow_ratio * 0.20)
    if color in ('black', 'gray', 'silver'):
        quality -= 0.08
    return color, conf, shadow_ratio, quality, chroma_ratio

def dominant_chroma_color(h_values, s_values, v_values):
    color_masks = {
        'red': ((h_values <= 10) | (h_values >= 170)) & (s_values >= 45) & (v_values >= 40),
        'orange': (h_values > 10) & (h_values <= 24) & (s_values >= 45) & (v_values >= 42),
        'yellow': (h_values > 24) & (h_values <= 38) & (s_values >= 40) & (v_values >= 48),
        'green': (h_values > 38) & (h_values <= 85) & (s_values >= 38) & (v_values >= 40),
        'blue': (h_values > 85) & (h_values <= 130) & (s_values >= 38) & (v_values >= 38),
        'purple': (h_values > 130) & (h_values < 170) & (s_values >= 38) & (v_values >= 38),
    }
    total = max(1, len(h_values))
    scores = {
        color: float(np.count_nonzero(mask)) / float(total)
        for color, mask in color_masks.items()
    }
    return max(scores.items(), key=lambda item: item[1])

def color_membership_mask(color, h, s, v):
    if color == 'red':
        return ((h <= 10) | (h >= 170)) & (s >= 42) & (v >= 40)
    if color == 'orange':
        return (h > 10) & (h <= 24) & (s >= 45) & (v >= 45)
    if color == 'yellow':
        return (h > 24) & (h <= 38) & (s >= 40) & (v >= 50)
    if color == 'green':
        return (h > 38) & (h <= 85) & (s >= 38) & (v >= 40)
    if color == 'blue':
        return (h > 85) & (h <= 130) & (s >= 38) & (v >= 38)
    if color == 'purple':
        return (h > 130) & (h < 170) & (s >= 38) & (v >= 38)
    return np.zeros_like(h, dtype=bool)

def map_hsv_to_color(h_value, s_value, v_value):
    if v_value < 80:
        return 'black'
    if s_value < 45 and v_value >= 185:
        return 'white'
    if s_value < 55 and 120 <= v_value < 185:
        return 'silver'
    if s_value < 58:
        return 'gray'
    if h_value <= 10 or h_value >= 170:
        return 'red'
    if h_value <= 24:
        return 'orange'
    if h_value <= 38:
        return 'yellow'
    if h_value <= 85:
        return 'green'
    if h_value <= 130:
        return 'blue'
    if h_value < 170:
        return 'purple'
    return 'unknown'

# ==========================================
# 2. ตรวจจับแค่รถในพื้นที่สนใจ และ crop
# ==========================================

def detect_and_crop_vehicles(
    source,
    crop_padding_x=CROP_PADDING_X,
    crop_padding_y=CROP_PADDING_Y,
    crop_square=CROP_SQUARE,
    min_crop_size=MIN_CROP_SIZE,
    model_path=MODEL_PATH,
    conf_threshold=CONF_THRESHOLD,
    iou_threshold=IOU_THRESHOLD,
    imgsz=IMGSZ,
    frame_skip=FRAME_SKIP,
    use_traffic_line_roi=USE_TRAFFIC_LINE_ROI,
    display=DISPLAY,
    print_every=PRINT_EVERY_N_FRAMES,
    max_frames=MAX_FRAMES,
    route_log_path=ROUTE_LOG_PATH,
    crop_output_dir=CROP_OUTPUT_DIR,
    save_crops=SAVE_CROPS,
    save_async=SAVE_ASYNC,
    require_engine_on_jetson=REQUIRE_ENGINE_ON_JETSON,
    traffic_lines=None,
    enable_classifier=ENABLE_CROP_CLASSIFIER,
    classifier_model_path=CLASSIFIER_MODEL_PATH,
    classifier_labels_path=CLASSIFIER_LABELS_PATH,
    classifier_input_size=CLASSIFIER_INPUT_SIZE,
    classifier_device=CLASSIFIER_DEVICE,
    classifier_enabled_classes=CLASSIFIER_ENABLED_CLASSES,
    classifier_min_confidence=CLASSIFIER_MIN_CONFIDENCE,
    camera_name=DEFAULT_CAMERA,
    traffic_line_frame_size=None,
    traffic_line_roi_margin=TRAFFIC_LINE_ROI_MARGIN,
    line_hit_tolerance=LINE_HIT_TOLERANCE,
    draw_only_after_origin=DRAW_ONLY_AFTER_ORIGIN
):
    """
    ตรวจจับรถ (เฉพาะในพื้นที่สนใจ) และบันทึก crop
    
    Args:
        source: ที่มาของภาพ (ไฟล์, กล้อง หรือ video)
        crop_padding_x: ระยะเว้นซ้าย-ขวาของ crop (pixels)
        crop_padding_y: ระยะเว้นบน-ล่างของ crop (pixels)
    """
    model_path = resolve_model_path(model_path, require_engine_on_jetson)
    use_model_class_filter = model_path.lower().endswith('.pt')
    live_source = is_live_source(source)
    cap = open_capture_with_retries(source, live_source)
    if cap is None or not cap.isOpened():
        print(f"Cannot open source: {source}")
        return
    frame_count = 0
    vehicle_count = 0
    
    # ได้เฟรมแรกเพื่อใช้ตั้งค่าขนาดภาพและ ROI จาก traffic lines
    cap, first_frame = read_initial_frame(cap, source, live_source)
    if first_frame is None:
        print("❌ ไม่สามารถอ่านไฟล์ได้")
        if cap is not None:
            cap.release()
        return
    
    camera_config = CAMERA_CONFIGS.get(camera_name, CAMERA_CONFIGS[DEFAULT_CAMERA])
    if traffic_lines is None:
        traffic_lines = camera_config['traffic_lines']
    if traffic_line_frame_size is None:
        traffic_line_frame_size = camera_config.get('frame_size', TRAFFIC_LINE_FRAME_SIZE)
    active_traffic_lines = scale_traffic_lines(
        traffic_lines,
        first_frame.shape,
        traffic_line_frame_size
    )

    pending_frame = first_frame
    model = YOLO(model_path)
    # สร้าง tracker เพื่อติดตาม object และนับเฉพาะ vehicle ที่ไม่ซ้ำ
    tracker = CentroidTracker(maxDisappeared=50, maxDistance=120)
    inference_roi = (
        get_traffic_lines_roi(active_traffic_lines, first_frame.shape, traffic_line_roi_margin)
        if use_traffic_line_roi else None
    )
    saved_ids = set()
    completed_routes = 0
    route_states = {}
    route_log_file = None
    route_log_writer = None
    crop_output_dir = format_runtime_path(crop_output_dir, camera_name)
    route_log_path = format_runtime_path(route_log_path, camera_name)
    if save_crops:
        os.makedirs(crop_output_dir, exist_ok=True)
    image_writer = AsyncImageWriter(enabled=save_async) if save_crops else None
    crop_classifier = MultiTaskCropClassifier(
        classifier_model_path,
        classifier_labels_path,
        classifier_input_size,
        classifier_device
    ) if enable_classifier else None
    if route_log_path:
        ensure_parent_dir(route_log_path)
        should_write_header = not os.path.exists(route_log_path) or os.path.getsize(route_log_path) == 0
        route_log_file = open(route_log_path, 'a', newline='', encoding='utf-8')
        route_log_writer = csv.writer(route_log_file)
        if should_write_header:
            route_log_writer.writerow([
                'timestamp',
                'camera',
                'object_id',
                'class_id',
                'class_name',
                'confidence',
                'origin',
                'destination',
                'route',
                'frame',
                'x1',
                'y1',
                'x2',
                'y2',
                'vehicle_color',
                'color_confidence',
                'shadow_ratio',
                'brand',
                'brand_confidence',
                'crop_saved',
                'filename'
            ])

    print(
        f"Camera: {camera_name} | Model: {model_path} | source: {source} | imgsz: {imgsz} | "
        f"frame_skip: {frame_skip} | display: {display} | classifier: {enable_classifier} | save_crops: {save_crops} | "
        f"roi_margin: {traffic_line_roi_margin} | line_tolerance: {line_hit_tolerance}"
    )
    if inference_roi is not None:
        print(f"Inference ROI: {inference_roi}")

    decode_errors = 0
    reconnect_attempts = 0
    while cap.isOpened() or live_source:
        if pending_frame is not None:
            ret, frame = True, pending_frame
            pending_frame = None
        else:
            ret, frame = cap.read() if cap is not None and cap.isOpened() else (False, None)
        if not ret:
            if live_source:
                if cap is not None:
                    cap.release()
                reconnect_attempts += 1
                if LIVE_RECONNECT_MAX_ATTEMPTS and reconnect_attempts > LIVE_RECONNECT_MAX_ATTEMPTS:
                    print("\nStopped: live source reconnect limit reached.")
                    break
                if should_print_frame(reconnect_attempts, print_every):
                    print(f"\nWarning: lost source, reconnecting ({reconnect_attempts})")
                time.sleep(LIVE_RECONNECT_DELAY_SEC)
                cap = open_video_capture(source)
                if cap.isOpened():
                    reconnect_attempts = 0
                continue
            break
        if not is_valid_frame(frame):
            decode_errors += 1
            if should_print_frame(decode_errors, print_every):
                print(f"\nWarning: skipped invalid decoded frame ({decode_errors}/{DECODE_ERROR_LIMIT})")
            if decode_errors >= DECODE_ERROR_LIMIT:
                print("\nStopped: too many invalid decoded frames. Try re-encoding the video.")
                break
            continue
        decode_errors = 0
        
        frame_count += 1
        if max_frames and frame_count > max_frames:
            print(f"\nStopped: reached max_frames={max_frames}.")
            break
        run_inference = ((frame_count - 1) % max(1, frame_skip)) == 0
        frame_display = frame.copy() if display else None
        if display:
            draw_traffic_lines(frame_display, active_traffic_lines)

        if not run_inference:
            if display:
                for oid, detection in tracker.detections.items():
                    if tracker.disappeared.get(oid, 0) > 0:
                        continue
                    if DRAW_ONLY_AFTER_ORIGIN:
                        state = route_states.get(oid, {})
                        if state.get('origin') is None:
                            continue
                    x1, y1, x2, y2 = detection['bbox']
                    label = f"ID {oid} {detection['class_name']} {detection['confidence']:.2f}"
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 180, 255), 2)
                    cv2.putText(frame_display, label,
                               (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 2)
            entered_count, vehicle_count = route_counts(route_states, completed_routes)
            if should_print_frame(frame_count, print_every):
                print(
                    f"Frame: {frame_count} | Entered: {entered_count} | Cropped: {vehicle_count} | skip",
                    end='\r'
                )
            if display:
                cv2.imshow('Vehicle Route Pipeline', fit_display_frame(frame_display))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        if inference_roi is not None:
            roi_x1, roi_y1, roi_x2, roi_y2 = inference_roi
            inference_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            offset_x, offset_y = roi_x1, roi_y1
        else:
            inference_frame = frame
            offset_x, offset_y = 0, 0

        # ตรวจจับ
        predict_args = {
            'verbose': False,
            'conf': conf_threshold,
            'iou': iou_threshold,
            'imgsz': imgsz,
        }
        if use_model_class_filter:
            predict_args['classes'] = list(VEHICLE_CLASSES.keys())
        results = model(inference_frame, **predict_args)
        
        # เก็บ boxes ที่ผ่านเงื่อนไข vehicle หลังตัด ROI ด้วย traffic lines แล้ว
        valid_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls not in VEHICLE_CLASSES or conf < conf_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'class_id': cls,
                    'class_name': VEHICLE_CLASSES[cls],
                    'confidence': conf
                }
                if offset_x or offset_y:
                    detection = offset_detection_bbox(detection, offset_x, offset_y)
                valid_detections.append(detection)
        valid_detections = filter_detections(valid_detections)
        
        # อัพเดต tracker และบันทึก crop เฉพาะ object ที่เพิ่งถูก register
        previous_centroids = tracker.objects.copy()
        objects, newly_registered = tracker.update(valid_detections)
        for removed_id in tracker.last_deregistered:
            route_states.pop(removed_id, None)
            saved_ids.discard(removed_id)
        for oid, detection in objects.items():
            if tracker.disappeared.get(oid, 0) > 0:
                continue
            state = route_states.setdefault(oid, {
                'origin': None,
                'origin_frame': None,
                'crossed_lines': set(),
                'best_bbox': detection['bbox'],
                'best_conf': detection['confidence'],
                'best_class_id': detection['class_id'],
                'best_class_name': detection['class_name'],
                'seen_frames': 0,
            })
            state['seen_frames'] = state.get('seen_frames', 0) + 1

            if detection['confidence'] >= state['best_conf']:
                state['best_bbox'] = detection['bbox']
                state['best_conf'] = detection['confidence']
                state['best_class_id'] = detection['class_id']
                state['best_class_name'] = detection['class_name']

            if state['seen_frames'] < MIN_ROUTE_SEEN_FRAMES:
                continue
            if state['origin'] is not None:
                continue

            previous_center = previous_centroids.get(oid)
            current_center = tracker.objects.get(oid)
            for line in active_traffic_lines:
                line_name = line['name']
                if line_name in state['crossed_lines']:
                    continue
                if not hit_traffic_line(previous_center, current_center, detection['bbox'], line['points'], line_hit_tolerance):
                    continue

                state['crossed_lines'].add(line_name)
                if state['origin'] is None:
                    state['origin'] = line_name
                    state['origin_frame'] = frame_count
                    break
                if line_name == state['origin']:
                    break
                break

        if display:
            for oid, detection in objects.items():
                if tracker.disappeared.get(oid, 0) > 0:
                    continue
                if draw_only_after_origin:
                    state = route_states.get(oid, {})
                    if state.get('origin') is None:
                        continue
                x1, y1, x2, y2 = detection['bbox']
                label = f"ID {oid} {detection['class_name']} {detection['confidence']:.2f}"
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_display, label,
                           (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for oid, detection in objects.items():
            if oid in saved_ids or tracker.disappeared.get(oid, 0) > 0:
                continue
            previous_center = previous_centroids.get(oid)
            current_center = tracker.objects.get(oid)
            state = route_states.setdefault(oid, {
                'origin': None,
                'origin_frame': None,
                'crossed_lines': set(),
                'best_bbox': detection['bbox'],
                'best_conf': detection['confidence'],
                'best_class_id': detection['class_id'],
                'best_class_name': detection['class_name'],
                'seen_frames': 0,
            })

            if detection['confidence'] >= state['best_conf']:
                state['best_bbox'] = detection['bbox']
                state['best_conf'] = detection['confidence']
                state['best_class_id'] = detection['class_id']
                state['best_class_name'] = detection['class_name']

            if state.get('origin') is not None and state.get('origin_frame') is not None:
                if frame_count - state['origin_frame'] < MIN_ROUTE_GAP_FRAMES:
                    continue

            for line in active_traffic_lines:
                line_name = line['name']
                if line_name in state['crossed_lines']:
                    continue
                if not hit_traffic_line(previous_center, current_center, detection['bbox'], line['points'], line_hit_tolerance):
                    continue

                state['crossed_lines'].add(line_name)
                if state['origin'] is None:
                    state['origin'] = line_name
                    state['origin_frame'] = frame_count
                    break
                if line_name == state['origin']:
                    break

                origin = state['origin']
                destination = line_name
                route_name = f'{origin}_to_{destination}'
                crop_bbox = state.get('best_bbox', detection['bbox'])
                x1_pad, y1_pad, x2_pad, y2_pad = expand_bbox(
                    crop_bbox,
                    frame.shape,
                    crop_padding_x,
                    crop_padding_y,
                    crop_square,
                    min_crop_size
                )
                crop_img = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                if crop_img.size == 0:
                    continue
                class_name = state.get('best_class_name', detection['class_name'])
                conf = state.get('best_conf', detection['confidence'])
                vehicle_color = 'unknown'
                color_conf = 0.0
                shadow_ratio = 0.0
                brand, brand_conf, vehicle_color, color_conf = classify_vehicle_attributes_if_enabled(
                    crop_classifier,
                    crop_img,
                    class_name,
                    classifier_enabled_classes,
                    classifier_min_confidence
                )
                if not enable_classifier or crop_classifier is None or not crop_classifier.enabled:
                    vehicle_color, color_conf, shadow_ratio = estimate_color_from_crop(crop_img, class_name)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                color_suffix = f'_{safe_filename_part(vehicle_color)}' if vehicle_color.lower() != 'unknown' else ''
                brand_suffix = f'_{safe_filename_part(brand)}' if brand.lower() != 'unknown' else ''
                filename = ''
                crop_saved = False
                if save_crops:
                    filename = os.path.join(
                        crop_output_dir,
                        f'{timestamp}_id{oid}_{route_name}_{class_name}{color_suffix}{brand_suffix}_{conf:.2f}.jpg'
                    )
                    crop_saved = image_writer.write(filename, crop_img)
                saved_ids.add(oid)
                completed_routes += 1

                if route_log_writer:
                    route_log_writer.writerow([
                        timestamp,
                        camera_name,
                        oid,
                        state.get('best_class_id', detection['class_id']),
                        class_name,
                        f'{conf:.4f}',
                        origin,
                        destination,
                        route_name,
                        frame_count,
                        x1_pad,
                        y1_pad,
                        x2_pad,
                        y2_pad,
                        vehicle_color,
                        f'{color_conf:.4f}',
                        f'{shadow_ratio:.4f}',
                        brand,
                        f'{brand_conf:.4f}',
                        int(crop_saved),
                        filename
                    ])
                    route_log_file.flush()
                color_text = f" {vehicle_color}" if vehicle_color.lower() != 'unknown' else ""
                brand_text = f" {brand} {brand_conf:.2f}" if brand.lower() != 'unknown' else ""
                print(f"\nRoute crop: ID {oid} {route_name} {class_name}{color_text}{brand_text} {conf:.2f}")
                break
            continue

        # ปริ้นสถานะ (frame + unique vehicles seen)
        entered_count, vehicle_count = route_counts(route_states, completed_routes)
        if should_print_frame(frame_count, print_every):
            print(f"Frame: {frame_count} | Entered: {entered_count} | Cropped: {vehicle_count}", end='\r')

        # แสดงภาพ
        if display:
            cv2.imshow('Vehicle Route Pipeline', fit_display_frame(frame_display))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if route_log_file:
        route_log_file.close()
    if image_writer:
        image_writer.close()
    if display:
        cv2.destroyAllWindows()
    vehicle_count = completed_routes
    print(f"\n✅ เสร็จ! พบรถทั้งหมด: {vehicle_count} คัน")


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle route cropper optimized for Jetson Nano.")
    parser.add_argument("--source", default=SOURCE, help="Video path or camera index.")
    parser.add_argument("--model", default=MODEL_PATH, help="YOLO .engine/.pt model path.")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="Inference image size.")
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP, help="Run inference every N frames.")
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY_N_FRAMES, help="Print status every N frames.")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES, help="Stop after N frames; 0 means process the full source.")
    parser.add_argument("--route-log", default=ROUTE_LOG_PATH, help="CSV path for completed routes.")
    parser.add_argument("--no-route-log", action="store_true", help="Disable route CSV logging.")
    parser.add_argument("--crop-dir", default=CROP_OUTPUT_DIR, help="Directory for saved crops. Supports {camera}.")
    parser.add_argument("--save-crops", dest="save_crops", action="store_true", help="Save crop images for debug/audit.")
    parser.add_argument("--no-save-crops", dest="save_crops", action="store_false", help="Process crops in RAM only.")
    parser.add_argument("--camera", choices=sorted(CAMERA_CONFIGS.keys()), default=DEFAULT_CAMERA, help="Camera traffic-line config.")
    parser.add_argument("--display", action="store_true", default=DISPLAY, help="Show OpenCV preview window.")
    parser.add_argument("--no-display", dest="display", action="store_false", help="Disable OpenCV preview window.")
    parser.add_argument("--no-line-roi", dest="use_traffic_line_roi", action="store_false", help="Disable ROI crop around traffic lines.")
    parser.add_argument("--line-roi-margin", type=int, default=TRAFFIC_LINE_ROI_MARGIN, help="Margin around traffic lines for inference ROI.")
    parser.add_argument("--line-hit-tolerance", type=int, default=LINE_HIT_TOLERANCE, help="Pixel tolerance when a bbox touches a traffic line.")
    parser.add_argument("--draw-all-detections", dest="draw_only_after_origin", action="store_false", help="Show boxes before vehicles touch an origin line.")
    parser.add_argument("--draw-after-origin", dest="draw_only_after_origin", action="store_true", help="Only show boxes after vehicles touch an origin line.")
    parser.add_argument("--sync-save", dest="save_async", action="store_false", help="Save crops synchronously.")
    parser.add_argument("--allow-pt-on-jetson", dest="require_engine_on_jetson", action="store_false", help="Allow .pt model on Jetson for testing.")
    parser.add_argument("--classifier-model", default=CLASSIFIER_MODEL_PATH, help="Multi-task best.pt crop classifier path.")
    parser.add_argument("--classifier-labels", default=CLASSIFIER_LABELS_PATH, help="Multi-task label_map.json path.")
    parser.add_argument("--classifier-img-size", type=int, default=CLASSIFIER_INPUT_SIZE, help="Classifier input image size fallback.")
    parser.add_argument("--classifier-device", default=CLASSIFIER_DEVICE, help="Classifier device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--classifier-classes", nargs="+", default=sorted(CLASSIFIER_ENABLED_CLASSES), help="Vehicle classes to classify after crop.")
    parser.add_argument("--classifier-min-conf", type=float, default=CLASSIFIER_MIN_CONFIDENCE, help="Minimum confidence to keep best.pt brand/color labels.")
    parser.add_argument("--no-classifier", dest="enable_classifier", action="store_false", help="Disable crop classification.")
    parser.set_defaults(
        use_traffic_line_roi=USE_TRAFFIC_LINE_ROI,
        save_crops=SAVE_CROPS,
        save_async=SAVE_ASYNC,
        require_engine_on_jetson=REQUIRE_ENGINE_ON_JETSON,
        enable_classifier=ENABLE_CROP_CLASSIFIER,
        draw_only_after_origin=DRAW_ONLY_AFTER_ORIGIN,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    detect_and_crop_vehicles(
        source=args.source,
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        frame_skip=args.frame_skip,
        use_traffic_line_roi=args.use_traffic_line_roi,
        display=args.display,
        print_every=args.print_every,
        max_frames=args.max_frames,
        route_log_path=None if args.no_route_log else args.route_log,
        crop_output_dir=args.crop_dir,
        save_crops=args.save_crops,
        save_async=args.save_async,
        require_engine_on_jetson=args.require_engine_on_jetson,
        enable_classifier=args.enable_classifier,
        classifier_model_path=args.classifier_model,
        classifier_labels_path=args.classifier_labels,
        classifier_input_size=args.classifier_img_size,
        classifier_device=args.classifier_device,
        classifier_enabled_classes=set(args.classifier_classes),
        classifier_min_confidence=args.classifier_min_conf,
        camera_name=args.camera,
        traffic_line_roi_margin=args.line_roi_margin,
        line_hit_tolerance=args.line_hit_tolerance,
        draw_only_after_origin=args.draw_only_after_origin,
    )


if __name__ == "__main__":
    main()
