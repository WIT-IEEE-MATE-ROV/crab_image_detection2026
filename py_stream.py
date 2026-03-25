import socket
import struct
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


HOST = "0.0.0.0"
PORT = 5000

model_path = Path(r"C:\Users\gviei\OneDrive\Documents\rov-vision\runs\detect\crabv42\weights\best.pt")
model = YOLO(model_path)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print("Listening on", (HOST, PORT))

conn, addr = server.accept()
print("Connection from", addr)

data = b""
payload_size = struct.calcsize(">Q")

print("Entering recv loop...")

while True:
    while len(data) < payload_size:
        packet = conn.recv(4096)
        if not packet:
            break
        data += packet
    if len(data) < payload_size:
        break

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">Q", packed_msg_size)[0]
    print("Receiving frame of size:", msg_size)

    while len(data) < msg_size:
        packet = conn.recv(4096)
        if not packet:
            break
        data += packet
    if len(data) < msg_size:
        break

    frame_data = data[:msg_size]
    data = data[msg_size:]

    np_array = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, verbose=False, conf = .9)
    annotated = results[0].plot()

    if frame_rgb is None:
        print("Frame decode failed (None)")
        continue

    cv2.imshow("Pi Stream", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

conn.close()
server.close()
cv2.destroyAllWindows()
