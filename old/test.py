# test_rtsp.py
import cv2
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|buffer_size;1048576"

cap = cv2.VideoCapture("rtsp://10.0.11.37:8554/vdo1", cv2.CAP_FFMPEG)
print("Opened:", cap.isOpened())

ret, frame = cap.read()
print("Read:", ret)
if ret:
    print("Frame shape:", frame.shape)

cap.release()
