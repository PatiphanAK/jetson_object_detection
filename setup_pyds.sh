#!/usr/bin/env bash
# setup_pyds.sh — install pyds (DeepStream Python bindings) for DeepStream 6.0 + JP 4.6.1 + Python 3.6
#
# Steps:
#   1. Quick check: is pyds already importable?
#   2. Look for an existing wheel in known local paths
#   3. Try downloading from NVIDIA-AI-IOT/deepstream_python_apps releases
#   4. pip install + verify
#
# Note: pang-1's venv has include-system-site-packages=true, so installing
# to system pip also makes pyds visible inside the venv automatically.

set -u

CYAN='\033[1;36m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'; RED='\033[1;31m'; OFF='\033[0m'

log()  { echo -e "${CYAN}==>${OFF} $*"; }
ok()   { echo -e "${GREEN}OK ${OFF} $*"; }
warn() { echo -e "${YELLOW}WARN${OFF} $*"; }
err()  { echo -e "${RED}ERR ${OFF} $*"; }

# ---------------------------------------------------------------
# Detect target Python (prefer the dev venv, fallback to system)
# ---------------------------------------------------------------
VENV_PY="/home/pang-1/dev/venv/bin/python"
VENV_PIP="/home/pang-1/dev/venv/bin/pip"

if [ -x "$VENV_PY" ] && [ -x "$VENV_PIP" ]; then
    PY="$VENV_PY"
    PIP="$VENV_PIP"
    log "Using venv: $PY"
elif command -v pip3 >/dev/null 2>&1; then
    PY="python3"
    PIP="pip3"
    log "Using system pip3"
elif python3 -m pip --version >/dev/null 2>&1; then
    PY="python3"
    PIP="python3 -m pip"
    log "Using 'python3 -m pip'"
else
    err "No pip found. Install with:"
    err "  sudo apt install -y python3-pip"
    exit 5
fi

# ---------------------------------------------------------------
# 1) Already installed?
# ---------------------------------------------------------------
log "Checking if pyds is already importable in target Python"
if $PY -c "import pyds; print(getattr(pyds, '__version__', 'unknown'))" 2>/dev/null; then
    ok "pyds already installed — nothing to do"
    exit 0
fi

# ---------------------------------------------------------------
# 2) Search for a wheel locally
# ---------------------------------------------------------------
log "Looking for a pre-downloaded pyds wheel"
CAND=$(find /home/pang-1 /opt /tmp -maxdepth 6 -name "pyds*.whl" 2>/dev/null | head -1)
if [ -n "$CAND" ]; then
    ok "Found wheel: $CAND"
else
    # ---------------------------------------------------------------
    # 3) Download from NVIDIA GitHub release
    # ---------------------------------------------------------------
    log "No local wheel; trying to download from GitHub"
    DOWNLOAD_DIR="$HOME/dev/dependencies"
    mkdir -p "$DOWNLOAD_DIR"

    # For DS 6.0 + Python 3.6 + Jetson aarch64, pyds-1.1.1 is the correct one.
    # We'll try a few versions in order of preference.
    BASE="https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download"
    URLS=(
        "$BASE/v1.1.1/pyds-1.1.1-py3-none-linux_aarch64.whl"
        "$BASE/v1.1.0/pyds-1.1.0-py3-none-linux_aarch64.whl"
    )

    for URL in "${URLS[@]}"; do
        FILE="$DOWNLOAD_DIR/$(basename "$URL")"
        log "Trying $URL"
        if curl -fL --connect-timeout 8 --max-time 60 -o "$FILE" "$URL" 2>/dev/null; then
            ok "Downloaded $(basename "$URL") ($(du -h "$FILE" | cut -f1))"
            CAND="$FILE"
            break
        else
            warn "Download failed for $(basename "$URL")"
            rm -f "$FILE"
        fi
    done

    if [ -z "${CAND:-}" ]; then
        err "Could not download any pyds wheel."
        cat <<EOF

Manual install: from a machine with internet, download:
  https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.1/pyds-1.1.1-py3-none-linux_aarch64.whl

Then SCP it to this Nano and run:
  pip3 install --user <path-to-pyds.whl>
EOF
        exit 2
    fi
fi

# ---------------------------------------------------------------
# 4) Install + verify
# ---------------------------------------------------------------
log "Installing $CAND with $PIP"
if $PIP install --no-deps "$CAND"; then
    ok "pip install completed"
else
    err "pip install failed"
    exit 3
fi

log "Verifying import with $PY"
if $PY -c "import pyds; print('pyds version:', getattr(pyds, '__version__', 'unknown'))"; then
    ok "pyds is importable — setup complete!"
else
    err "pyds still not importable after install"
    exit 4
fi

echo
ok "Done. Run with:"
ok "  cd /home/pang-1/dev && $PY ds_count_v2.py"
