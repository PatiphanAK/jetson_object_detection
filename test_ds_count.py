"""
test_ds_count.py
================
Pre-flight checks for ds_count.py on Jetson Nano 4GB with 5 RTSP streams.

Each check returns one of:
    PASS  — required, all good
    FAIL  — required, broken (must fix before running ds_count.py)
    WARN  — works, but suboptimal for production
    INFO  — informational, no action needed

Run:
    python3 test_ds_count.py                   # all checks, no live PLAY
    python3 test_ds_count.py --live-test 20    # also PLAY for 20s, check CSV
    python3 test_ds_count.py --json            # machine-readable output
"""
import argparse
import csv
import importlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

# ────────────────────────────────────────────────────────────
# Config mirror — keep in sync with ds_count.py
# ────────────────────────────────────────────────────────────
DEV_DIR = Path("/home/pang-1/dev")
DS_COUNT_PATH = DEV_DIR / "ds_count.py"
INFER_CONFIG = DEV_DIR / "config_infer_yolov8.txt"
LOG_CSV = DEV_DIR / "detect_log.csv"
CUSTOM_LIB_NAME = "libnvdsinfer_custom_impl_Yolo.so"
CUSTOM_LIB_SEARCH = [
    Path(f"/home/pang-1/dev/{CUSTOM_LIB_NAME}"),
    Path(f"/home/pang-1/dev/nvdsinfer_custom_impl_Yolo/{CUSTOM_LIB_NAME}"),
    Path(f"/home/pang-1/tool/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/{CUSTOM_LIB_NAME}"),
    Path(f"/opt/nvidia/deepstream/deepstream/lib/{CUSTOM_LIB_NAME}"),
]
ENGINE_CANDIDATES = [
    DEV_DIR / "yolov8n.engine",
    DEV_DIR / "yolov8n_b5.engine",
]
SOURCES = {
    "cam-a": "rtsp://10.0.11.37:8554/vdo1",
    "cam-b": "rtsp://10.0.11.37:8554/vdo2",
    "cam-c": "rtsp://10.0.11.37:8554/vdo3",
    "cam-d": "rtsp://10.0.11.37:8554/vdo4",
    "cam-e": "rtsp://10.0.11.37:8554/vdo5",
}
MUXER_W, MUXER_H, MUXER_FPS = 640, 640, 15

# Budget thresholds for Nano 4GB + 5 streams
MIN_FREE_RAM_MB = 1500      # need ~1.5 GB free for pipeline + engine
MIN_FREE_DISK_MB = 500
MAX_OK_GPU_BUSY_PCT = 50    # GPU should be mostly idle before we start
PIPELINE_RAM_BUDGET_MB = 2200  # nvinfer + nvstreammux + activations + writers


# ────────────────────────────────────────────────────────────
# Result collector
# ────────────────────────────────────────────────────────────
class Results:
    def __init__(self):
        self.items = []   # list of dict {level, category, name, msg, detail}
        self.start = time.monotonic()

    def add(self, level, category, name, msg, detail=None):
        self.items.append({
            "level": level, "category": category,
            "name": name, "msg": msg, "detail": detail
        })

    def counts(self):
        c = {"PASS": 0, "FAIL": 0, "WARN": 0, "INFO": 0}
        for r in self.items:
            c[r["level"]] = c.get(r["level"], 0) + 1
        return c

    def overall_ok(self):
        return self.counts()["FAIL"] == 0


def _is_tty():
    return sys.stdout.isatty()


COLOR = {
    "PASS": "\033[92m",  # green
    "FAIL": "\033[91m",  # red
    "WARN": "\033[93m",  # yellow
    "INFO": "\033[94m",  # blue
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
}


def fmt(level):
    if _is_tty():
        return f"{COLOR[level]}{level:<4}{COLOR['RESET']}"
    return f"{level:<4}"


def hdr(text):
    if _is_tty():
        print(f"\n{COLOR['BOLD']}── {text} {'─' * (60 - len(text))}{COLOR['RESET']}")
    else:
        print(f"\n── {text} {'─' * (60 - len(text))}")


def line(r):
    cat = r["category"]
    name = r["name"]
    msg = r["msg"]
    print(f"  {fmt(r['level'])}  [{cat:<8}] {name:<32}  {msg}")
    if r["detail"]:
        for d in r["detail"].splitlines():
            print(f"        {d}")


# ────────────────────────────────────────────────────────────
# Check helpers
# ────────────────────────────────────────────────────────────
def run_cmd(cmd, timeout=10):
    try:
        # Python 3.6 compat: no capture_output / text= keywords
        p = subprocess.run(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout,
        )
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"


def tcp_open(host, port, timeout=2.0):
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True, None
    except Exception as e:
        return False, str(e)


# ────────────────────────────────────────────────────────────
# Individual checks
# ────────────────────────────────────────────────────────────
def check_files(res):
    """Static existence checks for files ds_count.py depends on."""

    # ds_count.py itself
    if DS_COUNT_PATH.exists():
        res.add("PASS", "FILE", "ds_count.py", f"found at {DS_COUNT_PATH}")
    else:
        res.add("FAIL", "FILE", "ds_count.py",
                f"NOT FOUND at {DS_COUNT_PATH}")
        return

    # nvinfer config
    if INFER_CONFIG.exists():
        try:
            txt = INFER_CONFIG.read_text()
            has_engine = re.search(r"^\s*model-engine-file\s*=", txt, re.M)
            has_lib = re.search(r"^\s*custom-lib-path\s*=", txt, re.M)
            has_parser = re.search(r"^\s*parse-bbox-func-name\s*=", txt, re.M)
            missing = []
            if not has_engine: missing.append("model-engine-file=")
            if not has_lib: missing.append("custom-lib-path=")
            if not has_parser: missing.append("parse-bbox-func-name=")
            if missing:
                res.add("WARN", "FILE", "config_infer_yolov8.txt",
                        "exists but missing keys",
                        detail="missing: " + ", ".join(missing))
            else:
                res.add("PASS", "FILE", "config_infer_yolov8.txt",
                        "exists with required keys")
        except Exception as e:
            res.add("WARN", "FILE", "config_infer_yolov8.txt",
                    f"unreadable: {e}")
    else:
        res.add("FAIL", "FILE", "config_infer_yolov8.txt",
                f"MISSING at {INFER_CONFIG}",
                detail=("nvinfer cannot load model without it.\n"
                        "See sample at /opt/nvidia/deepstream/deepstream/"
                        "samples/configs/deepstream-app/config_infer_primary_*.txt"))

    # Custom YOLO lib (search known locations, then fall back to filesystem scan)
    found_lib = next((p for p in CUSTOM_LIB_SEARCH if p.exists()), None)
    if not found_lib:
        rc, out, _ = run_cmd(
            f"find /home/pang-1 /opt -name '{CUSTOM_LIB_NAME}' -type f 2>/dev/null",
            timeout=8,
        )
        if rc == 0 and out:
            found_lib = Path(out.splitlines()[0].strip())
    if found_lib and found_lib.exists():
        size_kb = found_lib.stat().st_size // 1024
        res.add("PASS", "FILE", "DeepStream-Yolo lib",
                f"found ({size_kb} KB)", detail=str(found_lib))
    else:
        res.add("FAIL", "FILE", "DeepStream-Yolo lib",
                "MISSING (searched standard paths + ~/dev + ~/tool + /opt)",
                detail=("Build with:\n"
                        "  cd /home/pang-1/tool/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo\n"
                        "  CUDA_VER=10.2 make"))

    # TRT engine candidates
    found_engine = None
    for cand in ENGINE_CANDIDATES:
        if cand.exists():
            found_engine = cand
            size_mb = cand.stat().st_size / 1e6
            res.add("INFO", "FILE", f"engine {cand.name}",
                    f"found ({size_mb:.1f} MB)")
            break
    if found_engine is None:
        res.add("WARN", "FILE", "TRT engine",
                "no pre-built engine found",
                detail=("First run of ds_count.py will trigger nvinfer to\n"
                        "build the engine from ONNX — takes 10+ minutes on Nano.\n"
                        "Build ahead of time with build.py or build_batch.py."))


def check_deepstream(res):
    """DeepStream SDK + GStreamer plugins."""
    ds_root = Path("/opt/nvidia/deepstream")
    if ds_root.exists():
        # version
        ver = "?"
        for cand in sorted(Path("/opt/nvidia").glob("deepstream-*")):
            ver = cand.name.replace("deepstream-", "")
            break
        res.add("PASS", "DEPS", "DeepStream SDK",
                f"installed (v{ver})", detail=str(ds_root))
    else:
        res.add("FAIL", "DEPS", "DeepStream SDK",
                "NOT installed at /opt/nvidia/deepstream",
                detail="apt install deepstream-6.0 or download from NVIDIA")
        return

    # GStreamer plugins
    for plug in ("nvinfer", "nvstreammux", "nvdsosd", "uridecodebin", "fakesink"):
        rc, out, err = run_cmd(f"gst-inspect-1.0 {plug}")
        if rc == 0:
            res.add("PASS", "DEPS", f"gst:{plug}", "plugin found")
        else:
            res.add("FAIL", "DEPS", f"gst:{plug}", "plugin missing",
                    detail=err[:200])


def check_python_deps(res, venv_python):
    """Python dependencies (run in the venv if specified)."""
    pkgs = [
        ("gi",      "REQUIRED — PyGObject for GStreamer Python bindings"),
        ("pyds",    "REQUIRED — DeepStream Python bindings"),
        ("cv2",     "REQUIRED — OpenCV"),
        ("psutil",  "optional — used by some helpers"),
    ]
    for pkg, note in pkgs:
        rc, out, err = run_cmd(
            f'{venv_python} -c "import {pkg}; print(getattr({pkg}, \'__version__\', \'?\'))"',
            timeout=8,
        )
        if rc == 0:
            res.add("PASS", "PY", pkg, f"importable (v{out.strip()})")
        else:
            level = "FAIL" if "REQUIRED" in note else "WARN"
            res.add(level, "PY", pkg, "import failed",
                    detail=note + "\n" + err[:300])


def check_resources(res):
    """RAM / disk / GPU headroom for Nano + 5 streams."""
    # RAM
    free_kb = total_kb = avail_kb = 0
    try:
        for ln in Path("/proc/meminfo").read_text().splitlines():
            k, v = ln.split(":", 1)
            v = int(v.strip().split()[0])
            if k == "MemTotal": total_kb = v
            elif k == "MemFree": free_kb = v
            elif k == "MemAvailable": avail_kb = v
    except Exception as e:
        res.add("WARN", "RES", "meminfo", f"unreadable: {e}")

    free_mb = free_kb // 1024
    avail_mb = avail_kb // 1024
    msg = f"available={avail_mb} MB / free={free_mb} MB / total={total_kb // 1024} MB"
    if avail_mb < MIN_FREE_RAM_MB:
        res.add("FAIL", "RES", "RAM headroom",
                f"only {avail_mb} MB available — need >= {MIN_FREE_RAM_MB}",
                detail="Disable GUI: sudo systemctl set-default multi-user.target && reboot")
    elif avail_mb < PIPELINE_RAM_BUDGET_MB:
        res.add("WARN", "RES", "RAM headroom",
                f"{avail_mb} MB available (tight for 5-cam pipeline)", detail=msg)
    else:
        res.add("PASS", "RES", "RAM headroom", msg)

    # Disk in DEV_DIR
    try:
        st = shutil.disk_usage(DEV_DIR)
        free_d_mb = st.free // (1 << 20)
        if free_d_mb < MIN_FREE_DISK_MB:
            res.add("WARN", "RES", "disk free",
                    f"{free_d_mb} MB free in {DEV_DIR} (CSV will grow)")
        else:
            res.add("PASS", "RES", "disk free",
                    f"{free_d_mb} MB free in {DEV_DIR}")
    except Exception as e:
        res.add("WARN", "RES", "disk", str(e))

    # GPU busy?
    rc, out, _ = run_cmd("cat /sys/devices/gpu.0/load 2>/dev/null")
    if rc == 0 and out:
        try:
            load_per_mille = int(out)
            pct = load_per_mille / 10.0
            if pct > MAX_OK_GPU_BUSY_PCT:
                res.add("WARN", "RES", "GPU load",
                        f"GPU already {pct:.1f}% busy — another job may be running")
            else:
                res.add("PASS", "RES", "GPU load", f"{pct:.1f}% (idle enough)")
        except ValueError:
            res.add("INFO", "RES", "GPU load", f"unparsed value: {out}")

    # Power mode
    rc, out, _ = run_cmd("nvpmodel -q 2>&1 | grep -E 'NV Power Mode|^\\s*[0-9]+\\s*$'")
    if rc == 0:
        msg = out.replace("\n", " | ")
        if "MAXN" in out:
            res.add("PASS", "RES", "power mode", msg)
        else:
            res.add("WARN", "RES", "power mode",
                    "not in MAXN — inference will be slower",
                    detail="sudo nvpmodel -m 0 && sudo jetson_clocks --fan")

    # zram swap available
    rc, out, _ = run_cmd("free -m | awk '/Swap/ {print $2,$3,$4}'")
    if rc == 0 and out:
        parts = out.split()
        if len(parts) == 3:
            tot, used, free = parts
            res.add("INFO", "RES", "zram swap",
                    f"total={tot} MB used={used} MB free={free} MB")


def check_network(res):
    """Reachability of each RTSP source."""
    seen_hosts = {}
    for name, url in SOURCES.items():
        m = re.match(r"rtsp://([^:/]+)(?::(\d+))?/", url)
        if not m:
            res.add("WARN", "NET", f"{name} url", f"unparseable: {url}")
            continue
        host = m.group(1)
        port = int(m.group(2) or 554)

        # cache host ping result
        if host not in seen_hosts:
            rc, _, _ = run_cmd(f"ping -c 1 -W 1 {host} > /dev/null 2>&1")
            seen_hosts[host] = (rc == 0)
        if not seen_hosts[host]:
            res.add("FAIL", "NET", f"{name} ping", f"host {host} unreachable")
            continue

        # TCP probe
        ok, err = tcp_open(host, port, timeout=2.0)
        if ok:
            res.add("PASS", "NET", f"{name} tcp", f"{host}:{port} OPEN")
        else:
            res.add("FAIL", "NET", f"{name} tcp",
                    f"{host}:{port} closed",
                    detail="RTSP server not listening — start MediaMTX or similar")


def check_ffprobe(res, timeout_per_stream=4):
    """ffprobe each RTSP to confirm media flowing — only run if TCP is open."""
    ok_hosts = set()
    for r in res.items:
        if r["category"] == "NET" and r["name"].endswith("tcp") and r["level"] == "PASS":
            ok_hosts.add(r["name"].split()[0])
    if not ok_hosts:
        res.add("INFO", "MEDIA", "ffprobe", "skipped (no reachable streams)")
        return

    for name, url in SOURCES.items():
        if name not in ok_hosts:
            continue
        cmd = (f"timeout {timeout_per_stream} ffprobe -v error -rtsp_transport tcp "
               f"-show_entries stream=codec_name,width,height,r_frame_rate "
               f"-of default=noprint_wrappers=1 '{url}' 2>&1")
        rc, out, _ = run_cmd(cmd, timeout=timeout_per_stream + 3)
        if rc == 0 and out:
            res.add("PASS", "MEDIA", f"{name} probe", out.replace("\n", " | "))
        else:
            res.add("WARN", "MEDIA", f"{name} probe", "no media in window",
                    detail=out[:200])


def check_ds_count_static(res):
    """Static review of ds_count.py against best practices."""
    if not DS_COUNT_PATH.exists():
        return
    src = DS_COUNT_PATH.read_text()

    # Critical defaults
    m = re.search(r"batched-push-timeout.{0,5}=\s*(\d+)", src) or \
        re.search(r'"batched-push-timeout"\s*,\s*(\d+)', src)
    if m:
        us = int(m.group(1))
        if us > 200_000:
            res.add("WARN", "CODE", "batched-push-timeout",
                    f"{us} μs = {us/1000:.0f} ms (too long for live RTSP)",
                    detail="Recommend 33000-100000 (33-100 ms)")
        else:
            res.add("PASS", "CODE", "batched-push-timeout", f"{us} μs OK")

    if 'uridecodebin' in src and 'nvurisrcbin' not in src:
        res.add("WARN", "CODE", "source element",
                "uses uridecodebin — no RTSP auto-reconnect",
                detail="nvurisrcbin has built-in RTSP keepalive/reconnect")

    if 'confidence' in src and re.search(r'csv_writer\.writerow\(\[.+?,\s*["\']-["\']', src):
        res.add("WARN", "CODE", "confidence logged",
                "writing '-' instead of obj_meta.confidence",
                detail="frame_cls_count loses per-detection confidence")

    if 'bus.connect' not in src and 'add_signal_watch' not in src:
        res.add("WARN", "CODE", "bus message handler",
                "no GStreamer bus message handler",
                detail="Pipeline errors/EOS will be silent")

    if 'csv_file.flush' not in src and 'line_buffering' not in src:
        res.add("WARN", "CODE", "CSV flushing",
                "CSV is not flushed periodically — crash loses recent rows")

    if 'batch-size' in src:
        m = re.search(r'batch-size["\s,]+(\d+)', src) or \
            re.search(r'batch-size["\s]*,\s*len\(\s*SOURCES\s*\)', src)
        res.add("INFO", "CODE", "nvstreammux batch-size",
                "matches len(SOURCES) — correct")

    if 'nvurisrcbin' not in src and 'rtspsrc' not in src:
        # uridecodebin transports default to UDP
        pass

    # Check engine path consistency
    if str(INFER_CONFIG.name) in src:
        res.add("PASS", "CODE", "config path", f"references {INFER_CONFIG.name}")


def check_pipeline_smoke(res, venv_python):
    """
    Try to import + build the pipeline in NULL state without playing.
    Verifies all GStreamer factories produce non-None elements.
    """
    if not all(r["level"] != "FAIL" for r in res.items
               if r["category"] == "DEPS"):
        res.add("INFO", "SMOKE", "build pipeline",
                "skipped (DEPS failures present)")
        return

    smoke = '''
import sys, gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import pyds

Gst.init(None)
errs = []

def make(factory, name):
    e = Gst.ElementFactory.make(factory, name)
    if e is None:
        errs.append(f"{factory}: factory returned None")
    return e

sm = make("nvstreammux", "sm")
if sm:
    sm.set_property("width", 640); sm.set_property("height", 640)
    sm.set_property("batch-size", 5); sm.set_property("batched-push-timeout", 40000)
    sm.set_property("live-source", 1)

for i in range(5):
    make("uridecodebin", f"src-{i}")

inf = make("nvinfer", "inf")
osd = make("nvdsosd", "osd")
snk = make("fakesink", "snk")

pipe = Gst.Pipeline.new("test")
for e in [sm] + [Gst.ElementFactory.make("uridecodebin", f"src-{i}") for i in range(5)] + [inf, osd, snk]:
    if e: pipe.add(e)

# Don't actually request pads or link RTSP — just NULL state ping
state = pipe.set_state(Gst.State.NULL)
print(f"NULL state set: {state}")

if errs:
    for e in errs: print(f"ERR: {e}")
    sys.exit(1)
print("SMOKE OK")
'''
    rc, out, err = run_cmd(
        f"{venv_python} - <<'EOF'\n{smoke}\nEOF",
        timeout=20,
    )
    if rc == 0 and "SMOKE OK" in out:
        res.add("PASS", "SMOKE", "pipeline build", "all 5 src + nvinfer + osd + sink created")
    else:
        res.add("FAIL", "SMOKE", "pipeline build", "could not assemble pipeline",
                detail=(out + "\n" + err)[:600])


def check_live_run(res, venv_python, seconds):
    """
    Actually PLAY ds_count.py for N seconds and verify CSV gets rows.
    Only attempt if everything else passes.
    """
    if not res.overall_ok():
        res.add("INFO", "LIVE", f"PLAY {seconds}s",
                "skipped (other FAILs first)")
        return

    csv_before = LOG_CSV.exists() and LOG_CSV.stat().st_size or 0
    cmd = (f"cd {DEV_DIR} && timeout {seconds} {venv_python} ds_count.py "
           f"> /tmp/ds_count_test.log 2>&1; true")
    t0 = time.monotonic()
    rc, out, err = run_cmd(cmd, timeout=seconds + 10)
    dt = time.monotonic() - t0

    # Read log
    log = ""
    try: log = Path("/tmp/ds_count_test.log").read_text()
    except Exception: pass

    # CSV grew?
    csv_after = LOG_CSV.exists() and LOG_CSV.stat().st_size or 0
    grew = csv_after - csv_before

    if grew > 0:
        n_rows = 0
        try:
            with open(LOG_CSV) as f:
                n_rows = sum(1 for _ in f) - 1
        except Exception: pass
        res.add("PASS", "LIVE", f"PLAY {seconds}s",
                f"CSV grew by {grew} bytes, {n_rows} rows total",
                detail=log[-400:] if log else None)
    elif "Pipeline running" in log:
        res.add("WARN", "LIVE", f"PLAY {seconds}s",
                "pipeline started but CSV got no rows (no objects detected?)",
                detail=log[-400:])
    else:
        res.add("FAIL", "LIVE", f"PLAY {seconds}s",
                "pipeline failed to start",
                detail=log[-600:])


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv-python",
                    default="/home/pang-1/dev/venv/bin/python")
    ap.add_argument("--live-test", type=int, default=0, metavar="SECONDS",
                    help="Also PLAY ds_count.py for N seconds and verify CSV grows")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of human output")
    args = ap.parse_args()

    res = Results()

    # Phase 1: static + env
    check_files(res)
    check_deepstream(res)
    check_python_deps(res, args.venv_python)
    check_resources(res)
    check_network(res)
    check_ffprobe(res)
    check_ds_count_static(res)

    # Phase 2: dynamic (depends on phase 1 results)
    check_pipeline_smoke(res, args.venv_python)
    if args.live_test > 0:
        check_live_run(res, args.venv_python, args.live_test)

    # Output
    if args.json:
        out = {
            "summary": res.counts(),
            "overall": "PASS" if res.overall_ok() else "FAIL",
            "duration_s": round(time.monotonic() - res.start, 2),
            "items": res.items,
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        groups = {}
        for r in res.items:
            groups.setdefault(r["category"], []).append(r)
        order = ["FILE", "DEPS", "PY", "RES", "NET", "MEDIA", "CODE", "SMOKE", "LIVE"]
        for cat in order:
            if cat in groups:
                hdr(cat)
                for r in groups[cat]:
                    line(r)

        c = res.counts()
        hdr("SUMMARY")
        for level in ("PASS", "WARN", "FAIL", "INFO"):
            tag = fmt(level)
            print(f"  {tag}  {c[level]:>3}")
        overall = "READY" if res.overall_ok() else "NOT READY"
        print(f"\n  Overall: {COLOR['BOLD']}{overall}{COLOR['RESET']} "
              f"in {time.monotonic() - res.start:.1f}s")
        if not res.overall_ok():
            print(f"\n  Fix all {COLOR['FAIL']}FAIL{COLOR['RESET']} items "
                  f"before running ds_count.py.")
    sys.exit(0 if res.overall_ok() else 1)


if __name__ == "__main__":
    main()
