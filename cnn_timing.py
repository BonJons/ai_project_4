import torch
import time
from PIL import Image
from waldo_cnn import WaldoCNN, VAL_TRANSFORM

# CNN timing
model = WaldoCNN()
model.load_state_dict(torch.load("waldo_cnn_best.pth", map_location="cpu"))
model.eval()

img = Image.open("data_roboflow/waldo/0_w0.jpg").convert("RGB")
tensor = VAL_TRANSFORM(img).unsqueeze(0)

# Warm up
with torch.no_grad():
    model(tensor)

# Time it
times = []
for _ in range(100):
    start = time.perf_counter()
    with torch.no_grad():
        model(tensor)
    times.append((time.perf_counter() - start) * 1000)

print(f"CNN avg inference: {sum(times)/len(times):.1f} ms per patch")