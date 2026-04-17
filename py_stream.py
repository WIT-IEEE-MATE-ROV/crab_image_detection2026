import socket
import struct
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import Counter

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

def count_classes(result, class_names):
    counts = {name: 0 for name in class_names.values()}

    if result.boxes is None or result.boxes.cls is None:
        return counts

    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    detected = Counter(class_ids)

    for cls_id, n in detected.items():
        counts[class_names[cls_id]] = n

    return counts

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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if rgb_frame is None:
        print("Frame decode failed (None)")
        continue

    results = model(rgb_frame, verbose=False, conf=0.9)
    result = results[0]

    counts = count_classes(result, model.names)
    annotated = result.plot()
    
    cv2.imshow("Pi Stream", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("quitting....")
        break
    elif key == ord('c'):
        print("Counts in current frame:")
        for name, n in counts.items():
            print(f"  {name}: {n}")

conn.close()
server.close()
cv2.destroyAllWindows()
