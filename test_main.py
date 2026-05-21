"""
test_main.py
============
Tests for the zero-copy / bounded-queue / static-allocation properties of
main.py.

Run with:
    ./venv/bin/python -m unittest -v test_main
"""

import os
import sys
import tempfile
import textwrap
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _make_rgba(h=120, w=160, fill=None):
    """Build a deterministic (H, W, 4) uint8 RGBA frame."""
    a = np.zeros((h, w, 4), dtype=np.uint8)
    if fill is None:
        ys = np.arange(h, dtype=np.uint8).reshape(-1, 1)
        xs = np.arange(w, dtype=np.uint8).reshape(1, -1)
        a[:, :, 0] = ys
        a[:, :, 1] = xs
        a[:, :, 2] = (ys + xs).astype(np.uint8)
        a[:, :, 3] = 255
    else:
        a[:] = fill
    return a


# ────────────────────────────────────────────────────────────────────
# 1. Pure-function tests — no GStreamer / no TRT needed
# ────────────────────────────────────────────────────────────────────
class CropClipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import main as _main

        cls.main = _main

    def test_clips_to_image_bounds(self):
        img = _make_rgba(100, 200)
        out = self.main.crop_clip(img, -50, -50, 999, 999)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (100, 200, 3))

    def test_drops_alpha_channel(self):
        img = _make_rgba(50, 50)
        out = self.main.crop_clip(img, 10, 10, 40, 40)
        self.assertEqual(out.shape[2], 3, "alpha channel must be dropped")
        self.assertEqual(out.dtype, np.uint8)

    def test_returns_none_for_degenerate_bbox(self):
        img = _make_rgba(50, 50)
        self.assertIsNone(self.main.crop_clip(img, 10, 10, 10, 10))
        self.assertIsNone(self.main.crop_clip(img, 10, 10, 11, 11))
        self.assertIsNone(self.main.crop_clip(img, 30, 30, 5, 5))

    def test_crop_is_a_copy_not_a_view(self):
        """Critical: caller releases the source frame buffer after the probe
        returns. The crop must own its memory."""
        img = _make_rgba(80, 80)
        out = self.main.crop_clip(img, 10, 10, 30, 30)
        self.assertIsNotNone(out)
        before = int(out[0, 0, 0])
        img[10, 10, 0] = 42
        after = int(out[0, 0, 0])
        self.assertEqual(before, after, "crop must not alias the source frame")

    def test_only_copies_crop_region_not_full_frame(self):
        """Zero-copy property: total bytes copied <= crop area, not frame area."""
        img = _make_rgba(640, 640)
        out = self.main.crop_clip(img, 100, 100, 150, 180)
        self.assertEqual(out.shape, (80, 50, 3))
        self.assertLess(out.nbytes, 50 * 80 * 4)

    def test_accepts_rgb_input_too(self):
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        out = self.main.crop_clip(img, 5, 5, 15, 15)
        self.assertEqual(out.shape, (10, 10, 3))


# ────────────────────────────────────────────────────────────────────
# 2. Config loader tests
# ────────────────────────────────────────────────────────────────────
class ConfigLoaderTests(unittest.TestCase):
    def test_parses_full_config(self):
        import main as _main

        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            fh.write(
                textwrap.dedent("""
                [property]
                model-engine-file=model/detector.engine

                [sources]
                A=rtsp://h/a
                B=rtsp://h/b

                [app]
                infer-config=config.conf
                log-csv=out.csv
                brand-engine=model/brand.engine
                conf-threshold=0.4
                car-class-id=2
                brand-max-batch=8
                muxer-width=320
                muxer-height=320
                coco-classes=person,car,truck
            """).lstrip()
            )
            path = fh.name
        try:
            cfg = _main.load_runtime_config(path)
        finally:
            os.unlink(path)

        self.assertEqual(cfg["sources"], {"A": "rtsp://h/a", "B": "rtsp://h/b"})
        self.assertAlmostEqual(cfg["conf_threshold"], 0.4)
        self.assertEqual(cfg["car_class_id"], 2)
        self.assertEqual(cfg["brand_max_batch"], 8)
        self.assertEqual(cfg["muxer_w"], 320)
        self.assertEqual(cfg["coco_classes"], ["person", "car", "truck"])

    def test_parses_unknown_brand_idx(self):
        import main as _main, tempfile, os, textwrap
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            fh.write(textwrap.dedent("""
                [sources]
                A=rtsp://h/a

                [app]
                coco-classes=person,car
                unknown-brand-idx=7
            """).lstrip())
            path = fh.name
        try:
            cfg = _main.load_runtime_config(path)
            self.assertEqual(cfg["unknown_brand_idx"], 7)
        finally:
            os.unlink(path)

    def test_unknown_brand_idx_default_is_22(self):
        import main as _main, tempfile, os, textwrap
        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            fh.write(textwrap.dedent("""
                [sources]
                A=rtsp://h/a

                [app]
                coco-classes=person,car
            """).lstrip())
            path = fh.name
        try:
            cfg = _main.load_runtime_config(path)
            self.assertEqual(cfg["unknown_brand_idx"], 22)
        finally:
            os.unlink(path)

    def test_missing_sources_section_raises(self):
        import main as _main

        with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
            fh.write("[app]\ncoco-classes=person\n")
            path = fh.name
        try:
            with self.assertRaises(RuntimeError):
                _main.load_runtime_config(path)
        finally:
            os.unlink(path)


# ────────────────────────────────────────────────────────────────────
# 3. Static-allocation tests for BrandClassifier
#     Skipped if the engine file isn't available.
# ────────────────────────────────────────────────────────────────────
ENGINE_CANDIDATES = ["model/classy.engine"]


def _find_engine():
    for p in ENGINE_CANDIDATES:
        full = os.path.join(HERE, p)
        if os.path.exists(full):
            return full
    return None


@unittest.skipIf(_find_engine() is None, "no brand engine on disk — skipping")
class BrandClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import main as _main

        cls.main = _main
        cls.engine = _find_engine()
        cls.clf = _main.BrandClassifier(cls.engine, max_batch=4)

    def test_preallocated_buffers_exist(self):
        clf = self.clf
        self.assertTrue(hasattr(clf, "_resize_buf"))
        self.assertTrue(hasattr(clf, "_scratch_hwc"))
        self.assertTrue(hasattr(clf, "_batch_f32"))
        self.assertTrue(hasattr(clf, "_batch_typed"))
        self.assertEqual(clf._resize_buf.shape, (224, 224, 3))
        self.assertEqual(clf._scratch_hwc.shape, (224, 224, 3))
        self.assertEqual(clf._batch_f32.shape, (4, 3, 224, 224))
        self.assertEqual(clf._batch_typed.shape, (4, 3, 224, 224))

    def test_buffers_reused_across_classify_calls(self):
        """Same numpy buffer object id across calls — proves no realloc."""
        clf = self.clf
        ids_before = (
            id(clf._resize_buf),
            id(clf._scratch_hwc),
            id(clf._batch_f32),
            id(clf._batch_typed),
            id(clf.h_in),
            id(clf.h_out),
        )
        dummy = [
            np.random.randint(0, 255, (50, 80, 3), dtype=np.uint8) for _ in range(2)
        ]
        for _ in range(5):
            out = clf.classify(dummy)
            self.assertEqual(len(out), 2)
        ids_after = (
            id(clf._resize_buf),
            id(clf._scratch_hwc),
            id(clf._batch_f32),
            id(clf._batch_typed),
            id(clf.h_in),
            id(clf.h_out),
        )
        self.assertEqual(
            ids_before, ids_after, "BrandClassifier must not reallocate scratch buffers"
        )

    def test_classify_returns_argmax_in_class_range(self):
        clf = self.clf
        dummy = [
            np.random.randint(0, 255, (60, 90, 3), dtype=np.uint8) for _ in range(3)
        ]
        result = clf.classify(dummy)
        self.assertEqual(len(result), 3)
        for b in result:
            self.assertIsInstance(b, int)
            self.assertGreaterEqual(b, 0)
            self.assertLess(b, 26)

    def test_classify_empty_input(self):
        self.assertEqual(self.clf.classify([]), [])


# ────────────────────────────────────────────────────────────────────
# 4. Bounded-queue test on the actual GStreamer pipeline
# ────────────────────────────────────────────────────────────────────
def _gst_available():
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(_gst_available(), "GStreamer not importable in this env")
class BoundedQueueTests(unittest.TestCase):
    def test_pipeline_has_leaky_queue_with_max_buffers(self):
        import main as _main

        pipeline = _main.build_pipeline()

        queues = []
        it = pipeline.iterate_elements()
        while True:
            ok, elem = it.next()
            if int(ok) != 1:
                break
            if elem.get_factory().get_name() == "queue":
                queues.append(elem)

        self.assertGreaterEqual(len(queues), 1, "no queue element found in pipeline")
        leaky_caps = [
            (q.get_property("max-size-buffers"), int(q.get_property("leaky")))
            for q in queues
        ]
        self.assertTrue(
            any(b > 0 and b <= 16 and l == 2 for b, l in leaky_caps),
            f"no bounded leaky=downstream queue found: {leaky_caps}",
        )

        from gi.repository import Gst

        pipeline.set_state(Gst.State.NULL)



@unittest.skipIf(_find_engine() is None, "no brand engine on disk — skipping")
class BrandClassifierUnknownMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import main as _main
        cls.main = _main
        cls.engine = _find_engine()

    def test_default_mask_idx_excludes_unknown(self):
        """With unknown_idx=0, predictions must never equal 0."""
        clf = self.main.BrandClassifier(self.engine, max_batch=4, unknown_idx=0)
        crops = [np.random.randint(0, 255, (60, 90, 3), dtype=np.uint8)
                 for _ in range(4)]
        for _ in range(3):
            preds = clf.classify(crops)
            self.assertTrue(all(p != 0 for p in preds),
                            f"Unknown idx 0 leaked through: {preds}")

    def test_unknown_mask_idx_22(self):
        clf = self.main.BrandClassifier(self.engine, max_batch=4, unknown_idx=22)
        crops = [np.random.randint(0, 255, (60, 90, 3), dtype=np.uint8)
                 for _ in range(4)]
        preds = clf.classify(crops)
        self.assertTrue(all(0 <= p < 26 and p != 22 for p in preds),
                        f"Unknown idx 22 leaked through: {preds}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
