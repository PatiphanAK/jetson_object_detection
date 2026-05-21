# Makefile — DeepStream pipeline ops on Jetson Nano
# Run `make help` to list targets.

SHELL    := /bin/bash
PYTHON   := ./venv/bin/python
PIP      := ./venv/bin/pip
MAIN     := main.py
TESTS    := test_main
CONFIG   := config.conf
LOG_CSV  := v4_brandy.log
RUN_LOG  := /tmp/run.log
PROCESS  := venv/bin/python -u main.py
PIDFILE  := /tmp/main.pid

# ANSI colours for help output
C_BOLD   := \033[1m
C_DIM    := \033[2m
C_GREEN  := \033[32m
C_YELLOW := \033[33m
C_RESET  := \033[0m

.PHONY: help run run-fg stop restart status logs logs-tail report test \
        check-env install clean clean-logs clean-cache distclean \
        rebuild-engine jetson-clocks power-max temp lint git-status \
        commit-check perf-snapshot

.DEFAULT_GOAL := help

help:  ## Show this help
	@printf "$(C_BOLD)DeepStream Pipeline — Makefile targets$(C_RESET)\n"
	@printf "$(C_DIM)Usage: make <target>$(C_RESET)\n\n"
	@printf "$(C_BOLD)Run / control$(C_RESET)\n"
	@awk 'BEGIN {FS = ":.*?## "} /^(run|run-fg|stop|restart|status):.*?## / {printf "  $(C_GREEN)%-18s$(C_RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\n$(C_BOLD)Logs / monitoring$(C_RESET)\n"
	@awk 'BEGIN {FS = ":.*?## "} /^(logs|logs-tail|report|perf-snapshot|temp):.*?## / {printf "  $(C_GREEN)%-18s$(C_RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\n$(C_BOLD)Dev / test$(C_RESET)\n"
	@awk 'BEGIN {FS = ":.*?## "} /^(test|check-env|install|lint|git-status|commit-check):.*?## / {printf "  $(C_GREEN)%-18s$(C_RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\n$(C_BOLD)Perf / hardware$(C_RESET)\n"
	@awk 'BEGIN {FS = ":.*?## "} /^(jetson-clocks|power-max|rebuild-engine):.*?## / {printf "  $(C_GREEN)%-18s$(C_RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\n$(C_BOLD)Clean$(C_RESET)\n"
	@awk 'BEGIN {FS = ":.*?## "} /^(clean|clean-logs|clean-cache|distclean):.*?## / {printf "  $(C_GREEN)%-18s$(C_RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ── Run / control ────────────────────────────────────────────────────
run: stop clean-logs  ## Start pipeline in background, log to /tmp/run.log
	@nohup $(PYTHON) -u $(MAIN) > $(RUN_LOG) 2>&1 & \
	  echo $$! > $(PIDFILE); \
	  echo "Started pid=$$(cat $(PIDFILE)) — log: $(RUN_LOG)"
	@sleep 2
	@if ! kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
	  printf "$(C_YELLOW)Process died immediately — last log:$(C_RESET)\n"; \
	  tail -20 $(RUN_LOG); exit 1; \
	fi

run-fg:  ## Start pipeline in foreground (Ctrl+C to stop)
	$(PYTHON) -u $(MAIN)

stop:  ## Send SIGINT to running pipeline (graceful shutdown)
	@# Use pidfile when available (clean shutdown). Fall back to pgrep
	@# filtered by exact comm=python3 so we never kill the shell running
	@# the recipe (which has main.py in its argv).
	@if [ -f $(PIDFILE) ] && kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
	  pid=$$(cat $(PIDFILE)); \
	  kill -INT $$pid && echo "Sent SIGINT to pid $$pid"; \
	  for i in 1 2 3 4 5; do kill -0 $$pid 2>/dev/null || break; sleep 1; done; \
	  kill -KILL $$pid 2>/dev/null && echo "Force-killed (timed out)" || true; \
	  rm -f $(PIDFILE); \
	else \
	  pids=$$(pgrep -x python3 -f "$(MAIN)" 2>/dev/null); \
	  if [ -n "$$pids" ]; then \
	    echo "Sending SIGINT to: $$pids"; \
	    kill -INT $$pids; \
	    sleep 3; \
	    kill -KILL $$pids 2>/dev/null && echo "Force-killed" || true; \
	  else \
	    echo "Not running"; \
	  fi; \
	  rm -f $(PIDFILE); \
	fi

restart: stop run  ## Stop then start

status:  ## Show running PID, uptime, RAM/CPU
	@printf "$(C_BOLD)Process:$(C_RESET)\n"
	@pgrep -af "$(PROCESS)" | grep -v grep || echo "  not running"
	@printf "\n$(C_BOLD)Memory (RSS):$(C_RESET)\n"
	@ps -eo rss,pid,cmd --sort=-rss | grep -E "$(PROCESS)" | grep -v grep | head -3 || echo "  n/a"
	@printf "\n$(C_BOLD)Last batch report:$(C_RESET)\n"
	@grep "\[REPORT\]" $(RUN_LOG) 2>/dev/null | tail -1 || echo "  no report yet"

# ── Logs ────────────────────────────────────────────────────────────
logs:  ## Print last 50 lines of run log
	@tail -50 $(RUN_LOG)

logs-tail:  ## tail -f run log (Ctrl+C to exit)
	@tail -f $(RUN_LOG)

report:  ## Print last full [REPORT] block from run log
	@awk '/^=+$$/{block=""; in_block=1; next} in_block{block=block $$0 "\n"} /\[REPORT\]/{in_block=1; block=$$0 "\n"} END{print block}' \
	  $(RUN_LOG) 2>/dev/null || echo "no log"

perf-snapshot:  ## Single tegrastats line + tail of latest log
	@printf "$(C_BOLD)tegrastats:$(C_RESET) "
	@sudo -n tegrastats --interval 100 2>/dev/null | head -1 || \
	  cat /proc/loadavg | awk '{print "load:", $$1, $$2, $$3}'
	@latest=$$(ls -1t [0-9]*_v4_brandy*.log 2>/dev/null | head -1); [ -n "$$latest" ] && echo "log: $$latest"; tail -3 "$$latest" 2>/dev/null || echo "no log yet"

temp:  ## Show all thermal zones
	@for z in /sys/devices/virtual/thermal/thermal_zone*; do \
	  type=$$(cat $$z/type); \
	  t=$$(cat $$z/temp); \
	  printf "  %-18s %5.1f°C\n" "$$type" "$$(echo "scale=1; $$t/1000" | bc)"; \
	done

# ── Dev / test ──────────────────────────────────────────────────────
test:  ## Run unittest suite
	$(PYTHON) -m unittest -v $(TESTS)

check-env:  ## Verify imports + model files + venv
	@printf "$(C_BOLD)Python:$(C_RESET) "; $(PYTHON) --version
	@printf "$(C_BOLD)Required modules:$(C_RESET)\n"
	@$(PYTHON) -c "import sys; sys.path.insert(0,'.'); \
	mods=['cv2','numpy','tensorrt','pycuda.driver','gi','pyds','car_color']; \
	[print(f'  {m}: '+('OK' if __import__(m) else 'FAIL')) for m in mods]"
	@printf "$(C_BOLD)Model files:$(C_RESET)\n"
	@for f in model/detector.engine model/classy.engine; do \
	  if [ -f $$f ]; then \
	    printf "  %s ($$(du -h $$f | cut -f1))\n" "$$f"; \
	  else \
	    printf "  $(C_YELLOW)%s MISSING$(C_RESET)\n" "$$f"; \
	  fi; \
	done
	@printf "$(C_BOLD)Config:$(C_RESET) $(CONFIG) ($$(grep -c '^CCTV' $(CONFIG)) cameras)\n"

install:  ## Recreate venv + install requirements
	@if [ ! -f requirements.txt ]; then echo "no requirements.txt"; exit 1; fi
	@rm -rf venv
	python3 -m venv --system-site-packages venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@$(MAKE) -s check-env

lint:  ## Quick syntax + import check
	@$(PYTHON) -c "import ast, glob; [ast.parse(open(f).read()) for f in glob.glob('*.py')]; print('syntax OK')"
	@$(PYTHON) -c "import sys; sys.path.insert(0,'.'); import main, car_color, test_main; print('imports OK')"

git-status:  ## git status -sb + unpushed commits
	@git status -sb
	@printf "\n$(C_BOLD)Commits ahead of origin:$(C_RESET)\n"
	@git log @{u}..HEAD --oneline 2>/dev/null || echo "  none"

commit-check: lint test  ## Pre-commit verification: lint + test
	@printf "$(C_GREEN)✓ ready to commit$(C_RESET)\n"

# ── Perf / hardware ─────────────────────────────────────────────────
jetson-clocks:  ## Lock CPU/GPU/EMC at max clocks (needs sudo)
	@sudo -n jetson_clocks --show 2>/dev/null || sudo jetson_clocks --show
	@sudo -n jetson_clocks 2>/dev/null || sudo jetson_clocks
	@echo "max clocks locked"

power-max:  ## Set nvpmodel to MAXN (mode 0)
	@sudo -n nvpmodel -m 0 2>/dev/null || sudo nvpmodel -m 0
	@sudo -n nvpmodel -q 2>/dev/null | head -3

rebuild-engine:  ## Rebuild TensorRT engine from ONNX (DON'T copy .engine across devices)
	@if [ ! -f transform.py ]; then echo "transform.py not found"; exit 1; fi
	@if [ -z "$(ONNX)" ]; then echo "Usage: make rebuild-engine ONNX=model/path.onnx OUT=model/out.engine"; exit 1; fi
	$(PYTHON) transform.py --onnx $(ONNX) --output $(OUT)

# ── Clean ───────────────────────────────────────────────────────────
clean: clean-logs clean-cache  ## Remove logs + caches

clean-logs:  ## Remove CSV log files (timestamped + run.log)
	@rm -f [0-9]*_v4_brandy*.log [0-9]*_v4_fps*.log $(LOG_CSV) v4_fps.log $(RUN_LOG)
	@echo "logs cleared"

clean-cache:  ## Remove Python __pycache__
	@find . -name "__pycache__" -not -path "./venv/*" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -not -path "./venv/*" -delete 2>/dev/null || true
	@echo "cache cleared"

distclean: clean  ## Plus remove venv + .bak files
	@rm -rf venv
	@rm -f *.bak.* main.py.bak* config.conf.bak*
	@echo "venv + backups removed"
