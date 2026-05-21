#!/usr/bin/env python3
"""
Car Color Pixel Detector (Jetson Nano)
--------------------------------------
- Detects vehicles with the existing YOLO ONNX model (best_vehicle_detector_416.onnx)
- Crops each detection and finds the top-3 dominant pixel colors via K-Means
- Maps each cluster centroid to a named color from color_mapping.txt
  (Black, Blue, Brown, Gold, Green, Grey, Orange, Red, Silver, White, Yellow, Unknown)

Usage:
  python3 car_color_pixel_detector.py --input vdo2.mp4 --output vdo2_colors.mkv
  python3 car_color_pixel_detector.py --input rtsp://user:pass@host/stream
  python3 car_color_pixel_detector.py --input image.jpg --output result.jpg
  python3 car_color_pixel_detector.py --input 0   # webcam
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

# ----------------------------------------------------------------------
# 1. Paths & CLI
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Car Color Pixel Detector for Jetson Nano")
parser.add_argument(
    "--input",
    type=str,
    default=os.path.join(BASE_DIR, "vdo2.mp4"),
    help="Path to video, image, RTSP URL, or webcam index (e.g. 0)",
)
parser.add_argument(
    "--output",
    type=str,
    default=os.path.join(BASE_DIR, "vdo2_colors.mkv"),
    help="Output video/image path. Use 'none' to disable saving.",
)
parser.add_argument(
    "--detector",
    type=str,
    default=os.path.join(BASE_DIR, "best_vehicle_detector_416.onnx"),
    help="YOLO ONNX detector path",
)
parser.add_argument(
    "--conf", type=float, default=0.50, help="YOLO confidence threshold"
)
parser.add_argument(
    "--k", type=int, default=3, help="Number of dominant colors per crop"
)
parser.add_argument("--show", action="store_true", help="Display window (requires GUI)")
parser.add_argument(
    "--white-min",
    type=float,
    default=0.08,
    help="If 'White' appears in top-K with at least this share (0-1), "
    "promote it to rank 1. Set 0 to disable. Default 0.08 (8%%).",
)
args = parser.parse_args()

DETECTOR_ONNX_PATH = args.detector
YOLO_CONF_THRESHOLD = args.conf
TOP_K_COLORS = max(1, args.k)
WHITE_PROMOTE_MIN = max(0.0, args.white_min)
YOLO_INPUT_SIZE = 416
YOLO_CLASSES = {0: "car", 1: "motorcycle", 2: "truck", 3: "bus"}

# ----------------------------------------------------------------------
# 2. Reference color palette (LAB nearest-color matching)
# ----------------------------------------------------------------------
# Each entry: (name, (R, G, B), thai_label)
# Classification finds the palette entry whose LAB value is closest to the
# query centroid — perceptually uniform, so "Silver" vs "Bronze Silver" vs
# "Bronze Gray" land correctly without manually-tuned HSV cutoffs.
REFERENCE_PALETTE = [
    ("Black", (0, 0, 0), "ดำ"),
    ("White", (255, 255, 255), "ขาว"),
    ("Gray", (128, 128, 128), "เทา"),
    ("Metallic Green", (74, 124, 111), "เขียวเมทาลิค"),
    ("Chartreuse", (180, 215, 50), "เหลืองเขียว"),
    ("Blue", (135, 206, 235), "ฟ้า"),
    ("Charcoal", (54, 69, 79), "เทาเข้ม"),
    ("Silver", (192, 192, 192), "เงิน"),
    ("Gold", (212, 175, 55), "ทอง"),
    ("Navy Blue", (0, 0, 128), "น้ำเงิน"),
    ("Slate Blue", (112, 128, 144), "เทาฟ้า"),
    ("Bronze", (140, 120, 83), "บรอนซ์"),
    ("Red", (220, 20, 60), "แดง"),
    ("Maroon", (128, 0, 0), "แดงเลือดหมู"),
    ("Pink", (255, 182, 193), "ชมพู"),
    ("Bronze Gold", (175, 142, 63), "บรอนซ์ทอง"),
    ("Bronze Gray", (130, 120, 108), "บรอนซ์เทา"),
    ("Bronze Silver", (169, 163, 154), "บรอนซ์เงิน"),
    ("Orange", (255, 140, 0), "ส้ม"),
    ("Yellow", (255, 230, 0), "เหลือง"),
    ("Green", (0, 128, 0), "เขียว"),
    ("Light Green", (144, 238, 144), "เขียวอ่อน"),
    ("Dark Green", (0, 100, 0), "เขียวเข้ม"),
    ("Olive Green", (107, 142, 35), "เขียวขี้ม้า"),
]

UNKNOWN_NAME = "Unknown"

# Two-tone rules: (set_of_names_A, set_of_names_B, output_label).
# When BOTH sides contribute at least TWO_TONE_MIN of the pixel share, output
# the combined name instead of the single dominant colour.
TWO_TONE_RULES = [
    ({"Blue", "Navy Blue", "Slate Blue"}, {"White"}, "Blue-White"),
    ({"Red", "Maroon"}, {"White"}, "Red-White"),
    (
        {"Yellow"},
        {"Green", "Dark Green", "Light Green", "Olive Green", "Chartreuse"},
        "Yellow-Green",
    ),
]
TWO_TONE_MIN = 0.18  # each side must carry at least this share


# Build display swatches (BGR) and the LAB lookup table at module load.
def _bgr_from_rgb(rgb):
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


NAME_BGR_SWATCH = {name: _bgr_from_rgb(rgb) for name, rgb, _ in REFERENCE_PALETTE}
NAME_BGR_SWATCH.update(
    {
        "Blue-White": (235, 206, 135),
        "Red-White": (60, 20, 220),
        "Yellow-Green": (0, 230, 255),
        UNKNOWN_NAME: (100, 100, 100),
    }
)

# Classification anchors: real cars rarely hit the single canonical RGB above
# (e.g. "Blue" is sky-blue but most blue cars are darker, "Maroon" misses
# truly dark wine, "White" misses shadowed white body panels). For each
# palette name we list a handful of RGB anchors that all classify back to
# that name — LAB nearest-anchor then picks the right label even for off-
# centre car colours.
CLASSIFICATION_ANCHORS = [
    # name,          (R, G, B)
    ("Black", (0, 0, 0)),
    ("Black", (15, 15, 18)),
    ("Black", (28, 30, 35)),
    ("Charcoal", (54, 69, 79)),
    ("Charcoal", (45, 55, 65)),
    ("Charcoal", (70, 78, 85)),
    ("Gray", (128, 128, 128)),
    ("Gray", (100, 100, 105)),
    ("Gray", (115, 118, 120)),
    ("Silver", (192, 192, 192)),
    ("Silver", (170, 170, 172)),
    ("Silver", (180, 182, 185)),
    ("White", (255, 255, 255)),
    ("White", (235, 235, 235)),
    ("White", (218, 220, 222)),
    ("White", (205, 208, 212)),
    # Off-white / cloudy-CCTV whites: user prefers these to read as White, not
    # Silver. Anchors kept above Silver-typical brightness (L > 80 in LAB) so
    # darker mid-tone Silver/Gray pixels still classify correctly.
    ("White", (195, 200, 208)),
    ("Bronze Silver", (169, 163, 154)),
    ("Bronze Gray", (130, 120, 108)),
    ("Bronze Gray", (110, 100, 90)),
    ("Bronze", (140, 120, 83)),
    ("Bronze Gold", (175, 142, 63)),
    ("Slate Blue", (112, 128, 144)),
    ("Slate Blue", (90, 105, 125)),
    ("Red", (220, 20, 60)),
    ("Red", (200, 40, 50)),
    ("Red", (180, 30, 40)),
    ("Maroon", (128, 0, 0)),
    ("Maroon", (95, 20, 25)),
    ("Maroon", (70, 30, 40)),
    ("Maroon", (60, 40, 50)),
    ("Pink", (255, 182, 193)),
    ("Orange", (255, 140, 0)),
    ("Orange", (220, 110, 30)),
    ("Yellow", (255, 230, 0)),
    ("Yellow", (235, 210, 30)),
    ("Gold", (212, 175, 55)),
    ("Gold", (200, 165, 60)),
    ("Blue", (135, 206, 235)),
    ("Blue", (60, 140, 210)),
    ("Blue", (40, 100, 190)),
    ("Blue", (30, 85, 170)),
    ("Navy Blue", (0, 0, 128)),
    ("Navy Blue", (20, 30, 100)),
    ("Navy Blue", (40, 50, 90)),
    ("Navy Blue", (25, 35, 70)),
    ("Green", (0, 128, 0)),
    ("Green", (40, 130, 40)),
    ("Dark Green", (0, 100, 0)),
    ("Dark Green", (20, 70, 30)),
    ("Dark Green", (30, 55, 35)),
    ("Light Green", (144, 238, 144)),
    ("Olive Green", (107, 142, 35)),
    ("Olive Green", (80, 100, 40)),
    ("Metallic Green", (74, 124, 111)),
    ("Metallic Green", (60, 100, 90)),
    ("Chartreuse", (180, 215, 50)),
]

_ANCHOR_BGR = np.array(
    [(rgb[2], rgb[1], rgb[0]) for _, rgb in CLASSIFICATION_ANCHORS],
    dtype=np.uint8,
).reshape(-1, 1, 3)
_ANCHOR_LAB = (
    cv2.cvtColor(_ANCHOR_BGR, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
)
_ANCHOR_NAMES = [name for name, _ in CLASSIFICATION_ANCHORS]


def classify_bgr(bgr_pixel):
    """Return the palette name whose nearest LAB anchor matches `bgr_pixel`."""
    bgr = np.asarray(bgr_pixel, dtype=np.uint8).reshape(1, 1, 3)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
    dists = np.linalg.norm(_ANCHOR_LAB - lab, axis=1)
    return _ANCHOR_NAMES[int(np.argmin(dists))]


# ----------------------------------------------------------------------
# 3. Lighting normalization & texture features
# ----------------------------------------------------------------------
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe_lab(bgr):
    """Equalize the L-channel of LAB. Reduces day/night/CCTV lighting variance
    while preserving hue. Cited by stefanbo92/color-detector SVM pipeline:
    "yields better contrast on various illuminations".
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = _CLAHE.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _dominant_name(roi_bgr, sample_size=1000):
    """Return the single most-populous palette name in a small ROI.
    Used by the spatial half-split detector — runs a tiny K-Means and picks
    the cluster with the most members."""
    if roi_bgr is None or roi_bgr.size == 0:
        return None, 0.0
    roi = apply_clahe_lab(roi_bgr)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    s = hsv[..., 1]
    keep = ~((v > 240) & (s < 25))
    keep &= v > 12
    kept = roi[keep]
    if kept.shape[0] < 50:
        kept = roi.reshape(-1, 3)
    if kept.shape[0] > sample_size:
        idx = np.random.choice(kept.shape[0], sample_size, replace=False)
        kept = kept[idx]
    pixels = kept.astype(np.float32)
    cc = min(4, pixels.shape[0])
    if cc < 1:
        return None, 0.0
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
    _, labels, centers = cv2.kmeans(pixels, cc, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    centers_u8 = centers.astype(np.uint8).reshape(-1, 1, 3)
    centers_lab = (
        cv2.cvtColor(centers_u8, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    )
    name_counts = {}
    labels = labels.flatten()
    for i in range(cc):
        cnt = int(np.sum(labels == i))
        if cnt == 0:
            continue
        dists = np.linalg.norm(_ANCHOR_LAB - centers_lab[i], axis=1)
        n = _ANCHOR_NAMES[int(np.argmin(dists))]
        name_counts[n] = name_counts.get(n, 0) + cnt
    if not name_counts:
        return None, 0.0
    total = labels.size
    name, count = max(name_counts.items(), key=lambda x: x[1])
    return name, count / total


def detect_spatial_two_tone(crop_bgr):
    """Look for an upper/lower colour split that matches a TWO_TONE rule.
    Most two-tone cars (and Thai taxis especially) are horizontally banded:
    yellow body + green/red roof, blue roof + white body, etc. Running
    K-Means independently on the top and bottom strips surfaces this even
    when each half's pixel count is too small to trigger the global
    share-based rule.

    Returns the two-tone label (e.g. 'Yellow-Green') or None.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if h < 40 or w < 40:
        return None

    # Mid 60% of width avoids edge slivers (mirrors, neighbouring vehicles).
    mx1, mx2 = int(w * 0.20), int(w * 0.80)
    # Skip the very top/bottom (sky, road, wheels).
    upper = crop_bgr[int(h * 0.10) : int(h * 0.50), mx1:mx2]
    lower = crop_bgr[int(h * 0.55) : int(h * 0.90), mx1:mx2]

    up_name, up_share = _dominant_name(upper)
    lo_name, lo_share = _dominant_name(lower)
    if not up_name or not lo_name or up_name == lo_name:
        return None
    # Each half must be reasonably uniform in its dominant colour, otherwise
    # we're probably looking at a single-colour body with noise.
    if up_share < 0.35 or lo_share < 0.35:
        return None

    for set_a, set_b, label in TWO_TONE_RULES:
        if (up_name in set_a and lo_name in set_b) or (
            up_name in set_b and lo_name in set_a
        ):
            return label
    return None


def _apply_two_tone(top):
    """If two reference colours both carry at least TWO_TONE_MIN share, emit
    a single two-tone entry (e.g. 'Blue-White') at rank 1 with their combined
    share. Otherwise return top unchanged."""
    share_of = {n: s for n, s, _ in top}
    bgr_of = {n: b for n, _, b in top}
    for set_a, set_b, label in TWO_TONE_RULES:
        share_a = sum(share_of.get(n, 0.0) for n in set_a)
        share_b = sum(share_of.get(n, 0.0) for n in set_b)
        if share_a >= TWO_TONE_MIN and share_b >= TWO_TONE_MIN:
            combined = (
                label,
                share_a + share_b,
                NAME_BGR_SWATCH.get(label, (200, 200, 200)),
            )
            others = [t for t in top if t[0] not in set_a and t[0] not in set_b]
            return [combined] + others
    return top


def _apply_white_grey_silver_rule(top):
    """User rule: if Gray, Silver, AND White all appear in the top set, the car
    is White (CCTV often splits a true white panel across these three names)."""
    names = {n for n, _, _ in top}
    if {"Gray", "Silver", "White"}.issubset(names):
        white_entry = next(t for t in top if t[0] == "White")
        rest = [t for t in top if t[0] != "White"]
        return [white_entry] + rest
    return top


# ----------------------------------------------------------------------
# 4. Color extraction (K-Means on pixel population)
# ----------------------------------------------------------------------
def extract_top_colors(crop_bgr, k=3, sample_size=2000):
    """
    Return list of (name, percentage, swatch_bgr) sorted by population (desc).

    Pipeline:
      1. Center 50% region (avoid background, road, sky at edges)
      2. CLAHE on L-channel (normalize lighting)
      3. Mask out blown-out highlights and near-black noise
      4. K-Means cluster in BGR, classify each centroid by HSV name
      5. Disambiguate White vs Silver using V-channel texture variance
         (metallic flakes produce high std, matte white is smooth)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return []

    h, w = crop_bgr.shape[:2]
    if h < 12 or w < 12:
        return []

    # 1. Center 50% region
    cy1, cy2 = int(h * 0.25), int(h * 0.75)
    cx1, cx2 = int(w * 0.25), int(w * 0.75)
    roi = crop_bgr[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return []

    # 2. CLAHE on L-channel
    roi = apply_clahe_lab(roi)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v_channel = hsv[..., 2]
    s_channel = hsv[..., 1]

    # 3. Filter glass reflections and deep shadows
    keep_mask = ~((v_channel > 240) & (s_channel < 25))
    keep_mask &= v_channel > 12
    kept_pixels_bgr = roi[keep_mask]

    if kept_pixels_bgr.shape[0] < 50:
        kept_pixels_bgr = roi.reshape(-1, 3)

    # 4. Random sub-sample for K-Means
    n_pixels = kept_pixels_bgr.shape[0]
    if n_pixels > sample_size:
        idx = np.random.choice(n_pixels, sample_size, replace=False)
        kept_pixels_bgr = kept_pixels_bgr[idx]

    pixels_f32 = kept_pixels_bgr.astype(np.float32)
    cluster_count = min(max(k, 3) + 2, pixels_f32.shape[0])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
    _, labels, centers_bgr = cv2.kmeans(
        pixels_f32, cluster_count, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()

    # 5. Classify each centroid via nearest-LAB to the reference palette,
    #    aggregate pixel counts per palette name.
    centers_bgr_u8 = centers_bgr.astype(np.uint8).reshape(-1, 1, 3)
    centers_lab = (
        cv2.cvtColor(centers_bgr_u8, cv2.COLOR_BGR2LAB)
        .reshape(-1, 3)
        .astype(np.float32)
    )

    name_counts = {}
    name_bgr_sum = {}
    total = labels.size
    for c_idx in range(cluster_count):
        member_count = int(np.sum(labels == c_idx))
        if member_count == 0:
            continue
        dists = np.linalg.norm(_ANCHOR_LAB - centers_lab[c_idx], axis=1)
        name = _ANCHOR_NAMES[int(np.argmin(dists))]
        name_counts[name] = name_counts.get(name, 0) + member_count
        sum_bgr = name_bgr_sum.get(name, np.zeros(3, dtype=np.float64))
        name_bgr_sum[name] = sum_bgr + centers_bgr[c_idx] * member_count

    results = []
    for name, count in name_counts.items():
        avg_bgr = (name_bgr_sum[name] / count).astype(int).tolist()
        results.append((name, count / total, tuple(avg_bgr)))

    results.sort(key=lambda x: -x[1])
    top = results[:k]

    # 6. Spatial two-tone detection (upper-half vs lower-half).
    #    Catches Thai taxis, white-roof police cars, and other horizontally
    #    banded paint schemes that the global share-based rule misses when
    #    one band is small in pixel count but distinctive in position.
    spatial_label = detect_spatial_two_tone(crop_bgr)
    if spatial_label:
        sw = NAME_BGR_SWATCH.get(spatial_label, (200, 200, 200))
        # Compute combined share from existing top results (best-effort).
        related_share = 0.0
        for set_a, set_b, lbl in TWO_TONE_RULES:
            if lbl == spatial_label:
                for n, s, _ in top:
                    if n in set_a or n in set_b:
                        related_share += s
                break
        combined_share = max(
            0.5, related_share
        )  # at least 0.5 because two halves match
        top = [(spatial_label, combined_share, sw)] + [
            t for t in top if t[0] != spatial_label
        ][: k - 1]
    else:
        # 7. Content-based two-tone fallback (Blue-White, Red-White, Yellow-Green).
        top = _apply_two_tone(top)

    # 8. User rule: if {Gray, Silver, White} all appear in top-K -> White.
    top = _apply_white_grey_silver_rule(top)

    # 8. White-priority fallback. If White is in top-K with at least
    #    WHITE_PROMOTE_MIN share AND no chromatic colour appears in the
    #    top-K, move White to rank 1. Any chromatic in top-K blocks this so
    #    we don't relabel a dark-green/dark-blue car as White.
    if WHITE_PROMOTE_MIN > 0 and top:
        chromatic_names = {
            "Red",
            "Maroon",
            "Orange",
            "Yellow",
            "Gold",
            "Bronze",
            "Bronze Gold",
            "Green",
            "Dark Green",
            "Light Green",
            "Olive Green",
            "Metallic Green",
            "Chartreuse",
            "Blue",
            "Navy Blue",
            "Slate Blue",
            "Pink",
            "Blue-White",
            "Red-White",
            "Yellow-Green",
        }
        any_chromatic = any(n in chromatic_names for n, _, _ in top)
        if not any_chromatic:
            for i, (name, share, _bgr) in enumerate(top):
                if name == "White" and i > 0 and share >= WHITE_PROMOTE_MIN:
                    top = [top[i]] + [t for j, t in enumerate(top) if j != i]
                    break
    return top


# ----------------------------------------------------------------------
# 4. YOLO preprocessing / postprocessing
# ----------------------------------------------------------------------
def preprocess_yolo(frame, input_shape=(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE)):
    resized = cv2.resize(frame, input_shape)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    transposed = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(transposed, axis=0)


def parse_detections(raw, frame_w, frame_h, conf_thr):
    """raw shape: [N, 6] -> [x1, y1, x2, y2, score, class_id] in 416 space."""
    detections = []
    sx, sy = frame_w / YOLO_INPUT_SIZE, frame_h / YOLO_INPUT_SIZE
    for box in raw:
        score = float(box[4])
        if score < conf_thr:
            continue
        cls_id = int(box[5])
        if cls_id not in YOLO_CLASSES:
            continue
        x1 = int(max(0, min(box[0] * sx, frame_w - 1)))
        y1 = int(max(0, min(box[1] * sy, frame_h - 1)))
        x2 = int(max(0, min(box[2] * sx, frame_w - 1)))
        y2 = int(max(0, min(box[3] * sy, frame_h - 1)))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        detections.append((x1, y1, x2, y2, score, cls_id))
    return detections


# ----------------------------------------------------------------------
# 5. Drawing helpers
# ----------------------------------------------------------------------
def draw_results(frame, x1, y1, x2, y2, cls_name, score, top_colors):
    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    header = f"{cls_name} {score:.2f}"
    cv2.rectangle(frame, (x1, y1 - 22), (x1 + 8 * len(header) + 8, y1), (0, 255, 0), -1)
    cv2.putText(
        frame,
        header,
        (x1 + 4, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    # Color bar: stacked horizontal segments, width proportional to share
    bar_w = max(60, x2 - x1)
    bar_h = 16
    bar_y = y2 + 4
    if bar_y + bar_h > frame.shape[0]:
        bar_y = y1 - 22 - bar_h - 4
    if bar_y < 0:
        bar_y = max(0, y2 - bar_h - 2)

    cursor_x = x1
    total_share = sum(p for _, p, _ in top_colors) or 1.0
    for name, share, _avg_bgr in top_colors:
        seg_w = max(2, int(bar_w * share / total_share))
        swatch = NAME_BGR_SWATCH.get(name, NAME_BGR_SWATCH[UNKNOWN_NAME])
        cv2.rectangle(
            frame, (cursor_x, bar_y), (cursor_x + seg_w, bar_y + bar_h), swatch, -1
        )
        cv2.rectangle(
            frame, (cursor_x, bar_y), (cursor_x + seg_w, bar_y + bar_h), (0, 0, 0), 1
        )
        cursor_x += seg_w

    # Text legend below the bar
    text_y = bar_y + bar_h + 14
    for i, (name, share, _avg_bgr) in enumerate(top_colors):
        line = f"{i + 1}. {name} {share * 100:.0f}%"
        ty = text_y + i * 14
        if ty > frame.shape[0] - 2:
            break
        cv2.putText(
            frame,
            line,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


# ----------------------------------------------------------------------
# 6. ONNX Runtime session (CUDA -> CPU fallback)
# ----------------------------------------------------------------------
def load_detector(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Detector ONNX not found: {model_path}")

    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
        print("YOLO: using CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    print(f"YOLO providers: {providers}")
    return ort.InferenceSession(model_path, providers=providers)


# ----------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------
def detect_source_kind(src):
    if src.isdigit():
        return "stream"
    if src.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return "stream"
    ext = os.path.splitext(src)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp"):
        return "image"
    return "video"


def open_capture(src):
    """Open a cv2.VideoCapture for video/stream/webcam sources."""
    if src.isdigit():
        return cv2.VideoCapture(int(src))

    if src.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        gst = (
            f"rtspsrc location={src} protocols=tcp latency=0 ! "
            "rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! "
            "video/x-raw, format=BGRx ! videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=true sync=false"
        )
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        print("Hardware RTSP pipeline failed; falling back to FFmpeg.")

    return cv2.VideoCapture(src)


def run_image(detector, det_input_name, det_output_name, src, out_path):
    frame = cv2.imread(src)
    if frame is None:
        print(f"Failed to read image: {src}")
        sys.exit(1)
    annotated = process_frame(detector, det_input_name, det_output_name, frame)
    if out_path.lower() != "none":
        cv2.imwrite(out_path, annotated)
        print(f"Saved annotated image: {out_path}")
    # if args.show:
        # cv2.imshow("Car Color Detector", annotated)
        # cv2.waitKey(0)


def process_frame(detector, det_input_name, det_output_name, frame):
    h, w = frame.shape[:2]
    inp = preprocess_yolo(frame)
    out = detector.run([det_output_name], {det_input_name: inp})[0][0]
    detections = parse_detections(out, w, h, YOLO_CONF_THRESHOLD)

    for x1, y1, x2, y2, score, cls_id in detections:
        crop = frame[y1:y2, x1:x2]
        top_colors = extract_top_colors(crop, k=TOP_K_COLORS)
        if not top_colors:
            continue
        draw_results(
            frame,
            x1,
            y1,
            x2,
            y2,
            YOLO_CLASSES.get(cls_id, "vehicle"),
            score,
            top_colors,
        )
    return frame


def run_video(detector, det_input_name, det_output_name, src, out_path):
    cap = open_capture(src)
    if cap is None or not cap.isOpened():
        print(f"Failed to open source: {src}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    writer = None
    if out_path.lower() != "none":
        fourcc = cv2.VideoWriter_fourcc(
            *"XVID" if out_path.endswith(".mkv") else "mp4v"
        )
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"Writing output to: {out_path}")

    frame_count = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated = process_frame(detector, det_input_name, det_output_name, frame)

        if writer is not None:
            writer.write(annotated)
        # if args.show:
        #     cv2.imshow("Car Color Detector", annotated)
        #     if cv2.waitKey(1) & 0xFF == ord("q"):
        #         break

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - t0
            print(f"frame={frame_count} avg_fps={frame_count / elapsed:.2f}")

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"Done. Processed {frame_count} frames in {time.time() - t0:.1f}s")


def main():
    detector = load_detector(DETECTOR_ONNX_PATH)
    det_input_name = detector.get_inputs()[0].name
    det_output_name = detector.get_outputs()[0].name

    src = args.input
    kind = detect_source_kind(src)

    if kind == "image":
        run_image(detector, det_input_name, det_output_name, src, args.output)
    else:
        run_video(detector, det_input_name, det_output_name, src, args.output)


if __name__ == "__main__":
    main()
