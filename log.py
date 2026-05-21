"""
log.py
======
CSV logging for detection events and per-camera FPS reports.
"""

import csv


class DetectionLogger:
    def __init__(self, log_csv, fps_log, flush_every_rows=60):
        self._flush_every = flush_every_rows
        self._rows_since_flush = 0

        self._csv_file = open(log_csv, "w", newline="", buffering=1)
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ["timestamp", "cam", "cam_frame", "class", "confidence", "count_this_frame", "brand_id"]
        )

        self._fps_file = open(fps_log, "w", newline="", buffering=1)
        self._fps_writer = csv.writer(self._fps_file)
        self._fps_writer.writerow(["timestamp", "cam", "frames_delta", "fps", "elapsed_s"])

    def write_detection_rows(self, rows):
        for row in rows:
            self._csv_writer.writerow(row)
            self._rows_since_flush += 1
        if self._rows_since_flush >= self._flush_every:
            self._csv_file.flush()
            self._rows_since_flush = 0

    def write_fps_row(self, ts, cam, delta, fps, elapsed):
        if elapsed > 0:
            self._fps_writer.writerow([ts, cam, delta, f"{fps:.3f}", f"{elapsed:.3f}"])

    def flush_fps(self):
        try:
            self._fps_file.flush()
        except Exception:
            pass

    def flush_detection(self):
        try:
            self._csv_file.flush()
        except Exception:
            pass

    def close(self):
        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass
        try:
            self._fps_file.flush()
            self._fps_file.close()
        except Exception:
            pass
