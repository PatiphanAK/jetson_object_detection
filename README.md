#Jetson Car Detection App

project structure
CMAKEList.txt
add and .hpp


```# webcam
python main.py --source 0 --model yolov8n.onnx

# video file
python main.py --source video.mp4 --model yolov8n.onnx

# บันทึก output ด้วย
python main.py --source video.mp4 --model yolov8n.onnx --save output.mp4

# ปรับ threshold
python main.py --source 0 --model yolov8n.onnx --conf 0.4 --iou 0.5
```

ก่อนใช้ DeepStream มี prerequire
```
sudo apt install \
libssl1.1 \
libssl-dev \
libcurl4-openssl-dev \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstreamer-plugins-base1.0-dev \
libgstrtspserver-1.0-0 \
libjansson4 \
libyaml-cpp-dev \
libmosquitto1
```

```
sudo apt install deepstream-6.0
```


คำสั่ง version tensorRT
```python new_infer.py --source video.mp4 --model yolov8n.engine --save output.mp4```
