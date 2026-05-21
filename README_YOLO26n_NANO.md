# YOLO26n บน Jetson Nano 4GB — สรุปการแก้ไข

เป้าหมาย: รัน DeepStream inference 5 กล้อง (RTSP) ด้วยโมเดล **YOLO26n end2end** บน Jetson Nano 4GB / JetPack 4.x / **TensorRT 8.2** ผ่าน `ds_count_v3_nice.py`

**ผลลัพธ์:** ✅ 5 กล้องรันได้ที่ ~**17 FPS รวม** (≈ 3.3-3.7 FPS/กล้อง), CPU ~80%, RAM ~1.5 GB

---

## ปัญหาที่เจอ

### 1. Flow เดิมแยกขั้น Colab → Nano

โครงเดิมตามคอมเมนต์ใน `config_infer_yolo26.txt`:

```
yolo26n.pt
   │  [Colab] to_onnx.py + ONNX_Exporter_YOLO26n_BATCH5.py
   ▼
yolo26n_wpost_b5.onnx     (post Mod-surgery + Div(255), dynamic batch)
   │  scp -> Nano
   ▼
Nano: transform.py --onnx yolo26n_wpost_b5.onnx --input-name images_raw
   ▼
yolo26n_b5.engine
```

แต่บน Nano ไม่มีไฟล์ `yolo26n_wpost_b5.onnx` และไม่มี surgery script เลย — `ds_count_v3_nice.py` เลย fail ที่ `nvinfer` ตอน build engine fallback:

```
ERROR: Cannot access ONNX file '/home/pang-1/dev/yolo26n_wpost_b5.onnx'
[FATAL] Unable to set pipeline to PLAYING
```

### 2. Export ใน docker บน Nano แล้ว — TRT 8.2 reject ตอน build

หลัง pull `ultralytics/ultralytics:latest-jetson-jetpack4` และ run surgery script เวอร์ชันแรก (dynamic=True) → `yolo26n_wpost_b5.onnx` สำเร็จ. แต่ตอน `transform.py` build engine:

```
In node 438 (importRange): UNSUPPORTED_NODE:
Assertion failed: inputs.at(0).isInt32() &&
"For range operator with dynamic inputs, this version of TensorRT only supports INT32!"
```

**สาเหตุ:** TRT 8.2 รับ `Range` op ที่ input เป็น dynamic ได้แค่ INT32. YOLO26 end2end head ใช้ `torch.arange(batch_size)` ที่ออกมาเป็น INT64 + dynamic dim → TRT auto-cast ทำได้แค่กับ constants/weights ไม่ใช่ผลของ Shape op.

นี่คือผลข้างเคียงของ `dynamic=True` (สคริปต์ต้นฉบับใน Colab notebook ใช้ `dynamic=False` แล้วเปลี่ยนมาเป็น `True` เพื่อรองรับ batch 1..5).

---

## ทางแก้ที่ใช้ — static batch=5 ตั้งแต่ export

แทนที่จะใช้ dynamic batch แล้วไป fix Range op ทีหลัง → **export เป็น batch=5 ตายตัวเลย**.
nvstreammux มี `batch=5` อยู่แล้ว ดังนั้น engine batch fix ที่ 5 ใช้งานพอดี ไม่ต้อง padding.

```
yolo26n.pt
   │  Docker: ONNX_Exporter_YOLO26n_STATIC_B5.py
   │   ├─ ultralytics.YOLO('yolo26n.pt').export(
   │   │     format='onnx', opset=12, end2end=True,
   │   │     simplify=True, dynamic=False, batch=5)
   │   └─ surgery (Mod->Sub trick + prepend Div(255))
   ▼
yolo26n_wpost_b5.onnx     [5, 3, 640, 640] static
   │
   │  Nano venv: build_engine_static_b5.py  (no optimization profile)
   ▼
yolo26n_b5.engine          11.39 MB, FP16, batch=5 fixed
```

ข้อดี:
- Range op ทุกตัวกลายเป็น constant → TRT 8.2 build ผ่าน
- ไม่ต้อง optimization profile → builder code สั้นลงมาก
- inference latency คงที่ (batch padding ไม่มี variability)

ข้อเสีย:
- engine ใช้ได้ที่ batch=5 เท่านั้น (rebuild ถ้าจะใช้ batch อื่น)

---

## ไฟล์ใหม่ + ที่เปลี่ยน บน Nano (`/home/pang-1/dev/`)

| ไฟล์ | สถานะ | คำอธิบาย |
|---|---|---|
| `ONNX_Exporter_YOLO26n_STATIC_B5.py` | **ใหม่** | Export + surgery แบบ static batch=5 (รันใน docker) |
| `build_engine_static_b5.py` | **ใหม่** | TRT 8.2 engine builder สำหรับ static-batch ONNX (no profile) |
| `yolo26n_std.onnx` | gen | ONNX ดิบหลัง export (batch=5, dynamic=False) — ก่อน surgery |
| `yolo26n_wpost_b5.onnx` | gen | หลัง surgery: Mod-free + Div(255) prepend + input ชื่อ `images_raw` |
| `yolo26n_b5.engine` | gen | TRT engine 11.39 MB, FP16, batch=5 |
| `config_infer_yolo26.txt` | ไม่แก้ | ใช้ตามเดิม (`batch-size=5`, `onnx-file=yolo26n_wpost_b5.onnx`, `model-engine-file=yolo26n_b5.engine`) |
| `ds_count_v3_nice.py` | ไม่แก้ | ใช้ตามเดิม |

**ไฟล์ที่ flow เดิมต้องใช้แต่ตอนนี้ไม่ใช้แล้ว:**
- `to_onnx.py` (export ฝั่ง laptop) — ใช้ docker บน Nano ครบขั้นแทน
- `transform.py` (dynamic-batch engine builder) — ใช้ `build_engine_static_b5.py` แทน
- `ONNX_Exporter_YOLO26n_BATCH5.py` (dynamic version) — superseded โดย `STATIC_B5`

---

## วิธี reproduce ตั้งแต่ต้นบน Nano

```bash
cd /home/pang-1/dev

# 1) Pull image (ครั้งเดียว ~1.2 GB)
sudo docker pull ultralytics/ultralytics:latest-jetson-jetpack4

# 2) Export + surgery ใน docker -> yolo26n_wpost_b5.onnx (ใช้เวลา ~50s)
sudo docker run --rm --runtime nvidia \
  -v /home/pang-1/dev:/work -w /work \
  ultralytics/ultralytics:latest-jetson-jetpack4 \
  python3 ONNX_Exporter_YOLO26n_STATIC_B5.py

# 3) Build TRT engine บน Nano (~12 นาที, FP16, 1024MB workspace)
./venv/bin/python build_engine_static_b5.py

# 4) รัน DeepStream 5-cam pipeline
./venv/bin/python ds_count_v3_nice.py
```

---

## รายละเอียด surgery ที่ทำใน `ONNX_Exporter_YOLO26n_STATIC_B5.py`

### (a) แทน `Mod` op ที่ TRT 8.2 ไม่รองรับ

YOLO end2end head ใช้ `index // nc` และ `index % nc` (Python modulo) → ออกมาเป็น `Mod` op ใน ONNX. TRT 8.2 parse ไม่ได้.

```
Mod(a, b)         ←  replace with:
  if INT  :   a − (a / b) * b
  if FLOAT:   a − Floor(a / b) * b
```

จำนวนที่แทน: **1 Mod (int)** — ออกมาเป็น 3 nodes (`Div + Mul + Sub`).

### (b) Prepend `Div(255)` ก่อน input

DeepStream ส่ง pixel [0, 255] เข้า. โมเดลต้องการ [0, 1]. แทนที่จะใช้ `net-scale-factor=1/255` ของ nvinfer (อาจมี precision loss ตอน FP16), bake การหารใน graph เลย:

```
Input 'images_raw' [5,3,640,640]
  │  Div(/255.0)
  ▼
Original 'images' input  →  rest of network
```

แล้ว `config_infer_yolo26.txt` ตั้ง `net-scale-factor=1.0` + `--input-name images_raw` ตอน build engine ก็พอ.

### (c) Force static batch=5

```python
YOLO('yolo26n.pt').export(
    format='onnx', opset=12, imgsz=640,
    end2end=True, simplify=True,
    dynamic=False, batch=5,        # ★ key difference vs dynamic version
    half=False,
)
```

ผลลัพธ์ ONNX: input `[5, 3, 640, 640]`, output `[5, 300, 6]` (ทุก dim เป็นค่าตายตัว — ไม่มี `dim_param`).

---

## หมายเหตุ + caveat

- **`simplify=True` ใน ultralytics 8.4.52** มี warning `simplifier failure: FLOAT8E4M3FN` — `onnx-simplifier` รุ่นนี้เจอ dtype ที่ไม่รู้จัก แต่ export สำเร็จเพราะ ultralytics fall back ไปใช้ ONNX ที่ไม่ผ่าน simplify. ไม่กระทบ functionality.
- **INT64 weights warning** ตอน TRT parse — TRT 8.2 auto-cast เป็น INT32 ให้เอง (สำหรับ static weights/inputs); ปลอดภัยที่จะ ignore เมื่อ batch เป็น static.
- **DeepStream `enable-perf-measurement`** ไม่ได้เปิด → ต้องคำนวณ FPS เองจาก `cam_frame_count` ที่ `ds_count_v3_nice.py` log ลง CSV. ถ้าจะดู realtime PERF ให้เพิ่ม `[application] enable-perf-measurement=1` ใน config.
- **Power mode** ของ Nano: เพื่อให้ inference เร็วที่สุด แนะนำ `sudo nvpmodel -m 0` (MAXN) + `sudo jetson_clocks` ก่อนรัน (ยังไม่ได้ทำในเซสชันนี้ — FPS ปัจจุบันคือ default mode).

---

## Performance ที่วัดได้ (May 21, 2026)

```
=== 30s window ===
cam-a: 3.63 FPS    cam-b: 3.57 FPS    cam-c: 3.60 FPS
cam-d: 3.17 FPS    cam-e: 3.83 FPS
aggregate: 17.79 FPS

=== since start (437s) ===
cam-a: 3.31 FPS    cam-b: 3.51 FPS    cam-c: 3.09 FPS
cam-d: 3.56 FPS    cam-e: 3.36 FPS
aggregate: 16.85 FPS
```

Engine batch ≈ 60ms/iter on Maxwell GPU (FP16) → ~17 FPS aggregate ตรงตามขีดของ Nano 4GB.

ถ้าต้องการเพิ่ม FPS:
1. `nvpmodel -m 0 && jetson_clocks` → +10-20%
2. ลด `imgsz` 640 → 320 (rebuild engine) → ~3-4x แต่ลด accuracy
3. ลด `interval` หรือเพิ่ม `drop-frame-interval` ใน nvinfer
