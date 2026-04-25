import time
from ultralytics import YOLO

model = YOLO("runs/classify/runs/waldo_yolo/weights/best.pt")

times = []
for _ in range(100):
    start = time.perf_counter()
    model.predict("data_roboflow/waldo/0_w0.jpg", verbose=False)
    times.append((time.perf_counter() - start) * 1000)

print(f"YOLO avg inference: {sum(times)/len(times):.1f} ms per patch")