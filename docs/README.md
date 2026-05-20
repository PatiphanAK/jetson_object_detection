# 🚗 Jetson Car Detection App

Real-time object detection on **NVIDIA Jetson** using YOLOv8 + TensorRT, written in **C++**.  
รับ stream จาก RTSP IP Camera และ render bounding box ผ่าน OpenCV.

---

## 📋 Requirements

| Dependency | Version | หมายเหตุ |
|---|---|---|
| JetPack | 4.x / 5.x | ติดตั้งมากับ Jetson |
| CUDA | 10.2 / 11.x | ขึ้นกับ JetPack |
| TensorRT | 7.x / 8.x | ขึ้นกับ JetPack |
| OpenCV | ≥ 4.1.1 | ต้องมี FFMPEG + GStreamer |
| CMake | ≥ 3.10 | |
| GCC / G++ | ≥ 7.4 | |

---

## 📁 Project Structure

```
jetson-car-detection/
├── CMakeLists.txt          # Build configuration
├── README.md
├── include/
│   ├── engine.hpp          # TensorRT engine loader & buffer management
│   ├── preprocess.hpp      # Image preprocessing (resize, pad, normalize)
│   ├── postprocess.hpp     # NMS, bbox decode, confidence filter
│   ├── draw.hpp            # Bounding box & label rendering
│   ├── rtsp.hpp            # RTSP capture helper + reconnect logic
│   └── logger.hpp          # Frame stats logger (FPS, RAM, objects)
├── src/
│   ├── main.cpp            # Entry point & main loop
│   ├── engine.cpp
│   ├── preprocess.cpp
│   ├── postprocess.cpp
│   ├── draw.cpp
│   ├── rtsp.cpp
│   └── logger.cpp
├── models/
│   └── yolov8n.engine      # TensorRT engine file (generate ก่อนใช้)
└── build/                  # CMake build output (auto-generated)
```

---

## ⚙️ Build

```bash
# 1. Clone repo
git clone https://github.com/yourname/jetson-car-detection.git
cd jetson-car-detection

# 2. สร้าง build directory
mkdir build && cd build

# 3. Configure
cmake ..

# 4. Compile
make -j$(nproc)
```

> **หมายเหตุ** : `make -j$(nproc)` ใช้ทุก core ที่มี — บน Jetson Nano อาจช้า แนะนำ `make -j2`

---

## 🔧 CMakeLists.txt (ตัวอย่าง)

```cmake
cmake_minimum_required(VERSION 3.10)
project(jetson_car_detection)

set(CMAKE_CXX_STANDARD 14)

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorRT (แก้ path ให้ตรงกับ JetPack ที่ติดตั้ง)
set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu)
set(TENSORRT_LIBRARIES    nvinfer nvonnxparser)
include_directories(${TENSORRT_INCLUDE_DIRS})

# Headers
include_directories(include)

# Source files
file(GLOB SOURCES "src/*.cpp")

add_executable(car_detection ${SOURCES})

target_link_libraries(car_detection
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARIES}
    cuda
)
```

---

## 🏗️ Generate TensorRT Engine

ต้อง export model จาก YOLOv8 → ONNX → TensorRT engine ก่อน:

```bash
# บนเครื่อง PC หรือ Jetson ก็ได้
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640

# แปลง ONNX → TensorRT engine (รันบน Jetson)
/usr/src/tensorrt/bin/trtexec \
    --onnx=yolov8n.onnx \
    --saveEngine=models/yolov8n.engine \
    --fp16
```

---

## 🚀 Run

```bash
cd build

# RTSP stream (default)
./car_detection --source rtsp://10.0.11.37:8554/vdo1 --model ../models/yolov8n.engine

# บันทึก output video ด้วย
./car_detection --source rtsp://10.0.11.37:8554/vdo1 \
                --model ../models/yolov8n.engine \
                --save output.mp4

# Webcam
./car_detection --source 0 --model ../models/yolov8n.engine
```

**Arguments:**

| Flag | Default | คำอธิบาย |
|---|---|---|
| `--source` | `rtsp://...` | RTSP URL / file path / webcam index |
| `--model` | `yolov8n.engine` | Path ของ TensorRT engine |
| `--save` | _(none)_ | บันทึก output เป็น .mp4 |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.45` | IoU threshold สำหรับ NMS |
| `--log-every` | `30` | Log stats ทุก N frames |

**กด `q`** เพื่อหยุดโปรแกรม

---

## 📄 Log Output

โปรแกรมจะบันทึก `infer_log.txt` ทุก N frames:

```
frame=0   infer_ms=12.3  objects=3  ram_used_mb=1024  ram_percent=52.1  ram_available_mb=941
frame=30  infer_ms=11.8  objects=2  ram_used_mb=1031  ram_percent=52.5  ram_available_mb=934
```

---

## 🔌 RTSP Constants

แก้ค่าใน `include/rtsp.hpp`:

```cpp
// ── RTSP Constants ──────────────────────────────────
const std::string RTSP_URL           = "rtsp://10.0.11.37:8554/vdo1";
const std::string RTSP_TRANSPORT     = "udp";   // "tcp" | "udp"
const int         RTSP_BUFFER_SIZE   = 1;       // frames
const int         RTSP_RECONNECT_SEC = 3;       // วินาทีรอก่อน reconnect
```

---

## 🛠️ Troubleshooting

| ปัญหา | สาเหตุ | วิธีแก้ |
|---|---|---|
| `461 Unsupported transport` | RTSP server ไม่รับ TCP | เปลี่ยน `RTSP_TRANSPORT = "udp"` |
| `Cannot open source` | URL ผิด / network ไม่ถึง | `nc -zv <ip> 8554` เช็ค port |
| `Cannot connect to X server` | รันบน headless server | ใช้ `ssh -X` หรือปิด imshow |
| Engine load failed | Engine ไม่ตรงกับ TensorRT version | Re-generate engine บน Jetson ตัวเดิม |
| Low FPS | Model ใหญ่เกินไป | ใช้ `yolov8n` หรือลด `imgsz` |

---

## 📌 Tested On

- NVIDIA Jetson Xavier NX — JetPack 5.1.2
- OpenCV 4.1.1 (FFMPEG: YES, GStreamer: YES)
- TensorRT 8.5
