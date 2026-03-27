##streaming from pi to laptop
import socket
import struct
import cv2
from picamera2 import Picamera2

SERVER_IP = "192.168.137.1"  # windows laptop IP
SERVER_PORT = 5000

picam2 = Picamera2()
picam2.preview_configuration.main.size = (320, 240)  
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_IP, SERVER_PORT))

try:
    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # JPEG encode
        ret, encoded = cv2.imencode(".jpg", frame_bgr)
        if not ret:
            continue

        data = encoded.tobytes()
        # Pack length (unsigned long long) + data [web:11]
        sock.sendall(struct.pack(">Q", len(data)) + data)
finally:
    sock.close()
    picam2.stop()
    picam2.close()
