FILE = video.mp4
MODEL = yolov8n.onnx

venv:
	python3 -m venv venv --system-site-packages

play:
	python infer.py --source $(FILE) --model $(MODEL) --save output.mp4

RTSP_HOST  := 10.0.11.176

CAM_A      := rtsp://$(RTSP_HOST):8554/vdo1
CAM_B      := rtsp://$(RTSP_HOST):8555/vdo2
CAM_C      := rtsp://$(RTSP_HOST):8556/vdo3
CAM_D      := rtsp://$(RTSP_HOST):8557/vdo4
CAM_E      := rtsp://$(RTSP_HOST):8558/vdo5

MODEL      := yolov8n_b5.engine
SAVE_DIR   := ./out
VRAM_GB    := 3.0

SOURCES    := cam-a=$(CAM_A),cam-b=$(CAM_B),cam-c=$(CAM_C),cam-d=$(CAM_D),cam-e=$(CAM_E)

run:
	python main.py \
		--sources $(SOURCES) \
		--model $(MODEL) \
		--save-dir $(SAVE_DIR) \
		--vram-gb $(VRAM_GB)
