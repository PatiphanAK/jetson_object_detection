"""
car_color.py
============
Reusable car-color extraction module. Pure CV (K-Means + LAB nearest-color),
no model file, no GPU. Designed to be called from a hot pipeline probe.

Extracted from car_color_pixel_detector.py (CLI tool) so the same color logic
can be shared between the standalone tool and the live DeepStream pipeline.

Public API:
    extract_top_colors(crop_bgr, k=3, sample_size=2000)
        -> list[(name: str, share: float, swatch_bgr: tuple)]

    top_color_name(crop_bgr, k=3, sample_size=2000) -> str | None
        Convenience: just the rank-1 name (or None if crop too small).
"""

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Reference palette (LAB nearest-color matching)
# ----------------------------------------------------------------------
# Each entry: (name, (R, G, B), thai_label)
REFERENCE_PALETTE = [
    ("Black",          (0,   0,   0),   "ดำ"),
    ("White",          (255, 255, 255), "ขาว"),
    ("Gray",           (128, 128, 128), "เทา"),
    ("Metallic Green", (74,  124, 111), "เขียวเมทาลิค"),
    ("Chartreuse",     (180, 215, 50),  "เหลืองเขียว"),
    ("Blue",           (135, 206, 235), "ฟ้า"),
    ("Charcoal",       (54,  69,  79),  "เทาเข้ม"),
    ("Silver",         (192, 192, 192), "เงิน"),
    ("Gold",           (212, 175, 55),  "ทอง"),
    ("Navy Blue",      (0,   0,   128), "น้ำเงิน"),
    ("Slate Blue",     (112, 128, 144), "เทาฟ้า"),
    ("Bronze",         (140, 120, 83),  "บรอนซ์"),
    ("Red",            (220, 20,  60),  "แดง"),
    ("Maroon",         (128, 0,   0),   "แดงเลือดหมู"),
    ("Pink",           (255, 182, 193), "ชมพู"),
    ("Bronze Gold",    (175, 142, 63),  "บรอนซ์ทอง"),
    ("Bronze Gray",    (130, 120, 108), "บรอนซ์เทา"),
    ("Bronze Silver",  (169, 163, 154), "บรอนซ์เงิน"),
    ("Orange",         (255, 140, 0),   "ส้ม"),
    ("Yellow",         (255, 230, 0),   "เหลือง"),
    ("Green",          (0,   128, 0),   "เขียว"),
    ("Light Green",    (144, 238, 144), "เขียวอ่อน"),
    ("Dark Green",     (0,   100, 0),   "เขียวเข้ม"),
    ("Olive Green",    (107, 142, 35),  "เขียวขี้ม้า"),
]

UNKNOWN_NAME = "Unknown"

# Two-tone rules: (set_of_names_A, set_of_names_B, output_label).
TWO_TONE_RULES = [
    ({"Blue", "Navy Blue", "Slate Blue"}, {"White"}, "Blue-White"),
    ({"Red", "Maroon"},                   {"White"}, "Red-White"),
    ({"Yellow"},
     {"Green", "Dark Green", "Light Green", "Olive Green", "Chartreuse"},
     "Yellow-Green"),
]
TWO_TONE_MIN = 0.18
WHITE_PROMOTE_MIN = 0.08

# Pre-compute LAB lookup table
_REF_BGR = np.array(
    [(rgb[2], rgb[1], rgb[0]) for _, rgb, _ in REFERENCE_PALETTE],
    dtype=np.uint8,
).reshape(-1, 1, 3)
_REF_LAB = cv2.cvtColor(_REF_BGR, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
_REF_NAMES = [name for name, _, _ in REFERENCE_PALETTE]

# Single shared CLAHE instance (thread-safe enough for our single-probe usage)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def classify_bgr(bgr_pixel):
    """Return the palette name closest in CIE-LAB to a single BGR sample."""
    bgr = np.asarray(bgr_pixel, dtype=np.uint8).reshape(1, 1, 3)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
    dists = np.linalg.norm(_REF_LAB - lab, axis=1)
    return _REF_NAMES[int(np.argmin(dists))]


def apply_clahe_lab(bgr):
    """Equalize the L-channel of LAB. Reduces lighting variance."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = _CLAHE.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _apply_two_tone(top):
    """Emit a single two-tone entry when both sides carry >= TWO_TONE_MIN."""
    share_of = {n: s for n, s, _ in top}
    for set_a, set_b, label in TWO_TONE_RULES:
        share_a = sum(share_of.get(n, 0.0) for n in set_a)
        share_b = sum(share_of.get(n, 0.0) for n in set_b)
        if share_a >= TWO_TONE_MIN and share_b >= TWO_TONE_MIN:
            combined = (label, share_a + share_b, (200, 200, 200))
            others = [t for t in top if t[0] not in set_a and t[0] not in set_b]
            return [combined] + others
    return top


def _apply_white_grey_silver_rule(top):
    """{Gray, Silver, White} together -> White."""
    names = {n for n, _, _ in top}
    if {"Gray", "Silver", "White"}.issubset(names):
        white_entry = next(t for t in top if t[0] == "White")
        rest = [t for t in top if t[0] != "White"]
        return [white_entry] + rest
    return top


def _apply_white_promote(top, threshold=WHITE_PROMOTE_MIN):
    """Promote White to rank-1 if present with >= threshold share and no
    clearly chromatic dominator."""
    if threshold <= 0 or not top:
        return top
    chromatic = {"Red", "Maroon", "Orange", "Yellow", "Gold", "Bronze",
                 "Bronze Gold", "Green", "Dark Green", "Light Green",
                 "Olive Green", "Metallic Green", "Chartreuse",
                 "Blue", "Navy Blue", "Slate Blue", "Pink",
                 "Blue-White", "Red-White", "Yellow-Green"}
    top_name, top_share, _ = top[0]
    if top_name in chromatic and top_share >= 0.20:
        return top
    for i, (name, share, _bgr) in enumerate(top):
        if name == "White" and i > 0 and share >= threshold:
            return [top[i]] + [t for j, t in enumerate(top) if j != i]
    return top


def extract_top_colors(crop_bgr, k=3, sample_size=2000):
    """Return list of (name, share_0_1, avg_bgr) sorted desc by share.

    Input: BGR uint8 ndarray (H, W, 3). Returns [] if crop is too small or empty.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    h, w = crop_bgr.shape[:2]
    if h < 12 or w < 12:
        return []

    # Center 50% region (avoid background / road / sky)
    cy1, cy2 = int(h * 0.25), int(h * 0.75)
    cx1, cx2 = int(w * 0.25), int(w * 0.75)
    roi = crop_bgr[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return []

    roi = apply_clahe_lab(roi)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v_channel = hsv[..., 2]
    s_channel = hsv[..., 1]
    keep_mask = ~((v_channel > 240) & (s_channel < 25))
    keep_mask &= (v_channel > 12)
    kept_pixels_bgr = roi[keep_mask]
    if kept_pixels_bgr.shape[0] < 50:
        kept_pixels_bgr = roi.reshape(-1, 3)

    n_pixels = kept_pixels_bgr.shape[0]
    if n_pixels > sample_size:
        idx = np.random.choice(n_pixels, sample_size, replace=False)
        kept_pixels_bgr = kept_pixels_bgr[idx]

    pixels_f32 = kept_pixels_bgr.astype(np.float32)
    cluster_count = min(max(k, 3) + 2, pixels_f32.shape[0])
    if cluster_count < 2:
        return []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
    _, labels, centers_bgr = cv2.kmeans(
        pixels_f32, cluster_count, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()
    centers_bgr_u8 = centers_bgr.astype(np.uint8).reshape(-1, 1, 3)
    centers_lab = cv2.cvtColor(centers_bgr_u8, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

    name_counts = {}
    name_bgr_sum = {}
    total = labels.size
    for c_idx in range(cluster_count):
        member_count = int(np.sum(labels == c_idx))
        if member_count == 0:
            continue
        dists = np.linalg.norm(_REF_LAB - centers_lab[c_idx], axis=1)
        name = _REF_NAMES[int(np.argmin(dists))]
        name_counts[name] = name_counts.get(name, 0) + member_count
        sum_bgr = name_bgr_sum.get(name, np.zeros(3, dtype=np.float64))
        name_bgr_sum[name] = sum_bgr + centers_bgr[c_idx] * member_count

    results = []
    for name, count in name_counts.items():
        avg_bgr = tuple((name_bgr_sum[name] / count).astype(int).tolist())
        results.append((name, count / total, avg_bgr))
    results.sort(key=lambda x: -x[1])
    top = results[:k]

    top = _apply_two_tone(top)
    top = _apply_white_grey_silver_rule(top)
    top = _apply_white_promote(top)
    return top


def top_color_name(crop_bgr, k=3, sample_size=2000):
    """Convenience: just the rank-1 color name, or None."""
    top = extract_top_colors(crop_bgr, k=k, sample_size=sample_size)
    if not top:
        return None
    return top[0][0]
