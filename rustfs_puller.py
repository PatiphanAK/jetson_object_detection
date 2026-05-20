"""
rustfs_puller.py
================
Poll a RustFS (S3-compatible) bucket every 15s, download any new or
changed files into the local `transform/` directory, and auto-trigger
transform.py whenever a new .onnx file lands.

Dependencies:
    pip install boto3

Run:
    cd /home/pang-1/dev
    ./venv/bin/python rustfs_puller.py
    ./venv/bin/python rustfs_puller.py --once        # pull ครั้งเดียวแล้วออก
    ./venv/bin/python rustfs_puller.py --no-transform  # pull อย่างเดียว ไม่ build
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ============================================================
#  ⚠️  EDIT THESE CONSTANTS  ⚠️
# ============================================================
RUSTFS_ENDPOINT = "http://10.0.11.214:9001"
RUSTFS_ACCESS_KEY = "admin"
RUSTFS_SECRET_KEY = "pang1234"
RUSTFS_REGION = "us-east-1"

RUSTFS_BUCKET = "test"
RUSTFS_PREFIX = ""  # "" = pull ทั้ง bucket

LOCAL_DIR = Path("/home/pang-1/dev/transform")
POLL_INTERVAL_S = 15

RUN_TRANSFORM_ON_NEW_ONNX = True
TRANSFORM_CMD = [
    "/home/pang-1/dev/venv/bin/python",
    "/home/pang-1/dev/transform.py",
    "--onnx",
    "{file}",
]

STATE_FILE = LOCAL_DIR / ".puller_state.json"
# ============================================================


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as e:
            log(f"WARN state file unreadable ({e}), starting fresh")
    return {}


def save_state(path: Path, state: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def make_client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=RUSTFS_ENDPOINT,
        aws_access_key_id=RUSTFS_ACCESS_KEY,
        aws_secret_access_key=RUSTFS_SECRET_KEY,
        region_name=RUSTFS_REGION,
        config=Config(signature_version="s3v4"),
    )


def list_objects(client, bucket, prefix):
    """List all objects, handle pagination."""
    paginator = client.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket}
    if prefix:
        kwargs["Prefix"] = prefix
    objects = []
    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents", []):
            objects.append(obj)
    return objects


def trigger_transform(onnx_path: Path) -> None:
    if not RUN_TRANSFORM_ON_NEW_ONNX:
        return
    cmd = [arg.format(file=str(onnx_path)) for arg in TRANSFORM_CMD]
    log(f"  ↳ NEW ONNX — running: {' '.join(cmd)}")
    try:
        rc = subprocess.call(cmd)
        log(f"  ↳ transform.py {'OK (rc=0)' if rc == 0 else f'FAILED (rc={rc})'}")
    except FileNotFoundError as e:
        log(f"  ↳ spawn failed: {e}")


def pull_once(client, bucket, prefix, dest_dir, state, transform_queue):
    n_dl = n_skip = n_err = 0
    try:
        objects = list_objects(client, bucket, prefix)
    except Exception as e:
        log(f"  list_objects failed: {e}")
        return 0, 0, 1

    seen_keys = set()
    for obj in objects:
        key = obj["Key"]
        etag = obj.get("ETag", "").strip('"')
        size = obj.get("Size", 0)
        seen_keys.add(key)

        if key.endswith("/"):
            continue

        prev = state.get(key, {})
        prev_etag = prev.get("etag")
        local_path = dest_dir / key
        local_exists = local_path.exists()

        if local_exists and prev_etag == etag:
            n_skip += 1
            continue

        # Download
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = local_path.with_suffix(local_path.suffix + ".part")
        try:
            client.download_file(bucket, key, str(tmp))
            tmp.replace(local_path)
            n_dl += 1
            state[key] = {
                "etag": etag,
                "size": size,
                "downloaded_at": time.time(),
                "sha256": sha256_file(local_path),
            }
            kind = "new" if not local_exists else "changed"
            log(f"  ↓ {kind:<7} {key}  ({size:,} bytes, etag {etag[:8]}…)")

            if local_path.suffix.lower() == ".onnx":
                transform_queue.append(local_path)
        except Exception as e:
            n_err += 1
            log(f"  ✗ failed {key}: {e}")
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    removed = set(state.keys()) - seen_keys
    for k in sorted(removed):
        log(f"  (remote removed {k} — keeping local copy)")

    return n_dl, n_skip, n_err


def main():
    p = argparse.ArgumentParser(description="Poll RustFS bucket → transform/")
    p.add_argument("--once", action="store_true", help="Pull once and exit")
    p.add_argument("--interval", type=int, default=None)
    p.add_argument("--no-transform", action="store_true", help="Skip auto transform.py")
    args = p.parse_args()

    global RUN_TRANSFORM_ON_NEW_ONNX
    if args.no_transform:
        RUN_TRANSFORM_ON_NEW_ONNX = False

    interval = args.interval or POLL_INTERVAL_S

    try:
        import boto3  # noqa
    except ImportError:
        print("ERROR: boto3 not installed.", file=sys.stderr)
        print("       pip install boto3", file=sys.stderr)
        sys.exit(2)

    log("=" * 60)
    log("RustFS puller starting")
    log(f"  endpoint  : {RUSTFS_ENDPOINT}")
    log(f"  bucket    : {RUSTFS_BUCKET}")
    log(f"  prefix    : {RUSTFS_PREFIX or '(whole bucket)'}")
    log(f"  dest dir  : {LOCAL_DIR}")
    log(f"  interval  : {interval}s  {'(once)' if args.once else ''}")
    log(f"  on new    : {'transform.py' if RUN_TRANSFORM_ON_NEW_ONNX else 'log only'}")
    log("=" * 60)

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state(STATE_FILE)
    client = make_client()

    # Quick connectivity check
    try:
        client.list_buckets()
        log("connection OK")
    except Exception as e:
        log(f"FATAL cannot reach RustFS: {e}")
        sys.exit(3)

    stop = {"flag": False}

    def _sig(*_):
        log("stop signal")
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    pass_count = 0
    try:
        while not stop["flag"]:
            pass_count += 1
            t0 = time.monotonic()
            transform_queue = []
            n_dl, n_skip, n_err = pull_once(
                client, RUSTFS_BUCKET, RUSTFS_PREFIX, LOCAL_DIR, state, transform_queue
            )
            save_state(STATE_FILE, state)
            dt = time.monotonic() - t0
            log(
                f"pass #{pass_count}: downloaded={n_dl} skipped={n_skip} "
                f"errors={n_err} ({dt:.2f}s)"
            )

            for onnx in transform_queue:
                trigger_transform(onnx)

            if args.once or stop["flag"]:
                break

            elapsed = 0.0
            while elapsed < interval and not stop["flag"]:
                time.sleep(0.5)
                elapsed += 0.5
    finally:
        save_state(STATE_FILE, state)
        log("bye.")


if __name__ == "__main__":
    main()
