"""
Roboflow Waldo Patch Extractor
================================
Reads images + YOLO labels from the ait_project4 Roboflow dataset,
crops 256×256 patches around each Waldo (class 1) bounding box,
and samples random non-Waldo patches from the same images.

Output:
    data_roboflow/waldo/       — cropped Waldo patches
    data_roboflow/notwaldo/    — random non-Waldo patches

Run:
    python extract_roboflow_patches.py
"""

import os
import random
from PIL import Image

SRC_IMAGES = "/home/jon_s/ait_project4/data/raw/dataset/images"
SRC_LABELS = "/home/jon_s/ait_project4/data/raw/dataset/labels"
OUT_WALDO    = "data_roboflow/waldo"
OUT_NOTWALDO = "data_roboflow/notwaldo"
PATCH_SIZE   = 256
WALDO_CLASS  = 1
NOTWALDO_PER_IMAGE = 3  # random non-Waldo crops per image
SEED = 42

random.seed(SEED)


def yolo_to_pixels(cx, cy, bw, bh, img_w, img_h):
    """Convert YOLO normalized coords to pixel (x1, y1, x2, y2)."""
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return x1, y1, x2, y2


def crop_centered(image, cx_px, cy_px, patch_size, img_w, img_h):
    """Crop a patch_size square centered on (cx_px, cy_px), clamped to image bounds."""
    half = patch_size // 2
    x1 = max(0, cx_px - half)
    y1 = max(0, cy_px - half)
    x2 = x1 + patch_size
    y2 = y1 + patch_size
    if x2 > img_w:
        x2 = img_w
        x1 = max(0, x2 - patch_size)
    if y2 > img_h:
        y2 = img_h
        y1 = max(0, y2 - patch_size)
    return image.crop((x1, y1, x2, y2))


def random_notwaldo_crop(image, waldo_boxes, patch_size, img_w, img_h, max_tries=50):
    """Sample a random patch that doesn't overlap any Waldo box."""
    for _ in range(max_tries):
        x1 = random.randint(0, max(0, img_w - patch_size))
        y1 = random.randint(0, max(0, img_h - patch_size))
        x2, y2 = x1 + patch_size, y1 + patch_size

        overlap = False
        for bx1, by1, bx2, by2 in waldo_boxes:
            if x1 < bx2 and x2 > bx1 and y1 < by2 and y2 > by1:
                overlap = True
                break

        if not overlap:
            return image.crop((x1, y1, x2, y2))
    return None


def extract(src_images=SRC_IMAGES, src_labels=SRC_LABELS,
            out_waldo=OUT_WALDO, out_notwaldo=OUT_NOTWALDO):

    os.makedirs(out_waldo, exist_ok=True)
    os.makedirs(out_notwaldo, exist_ok=True)

    image_files = [f for f in os.listdir(src_images) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    waldo_count    = 0
    notwaldo_count = 0

    print(f"\n{'='*60}")
    print(f"  Roboflow Waldo Patch Extractor")
    print(f"  Source : {src_images}")
    print(f"  Output : {out_waldo} / {out_notwaldo}")
    print(f"{'='*60}")

    for fname in sorted(image_files):
        stem      = os.path.splitext(fname)[0]
        img_path  = os.path.join(src_images, fname)
        lbl_path  = os.path.join(src_labels, stem + ".txt")

        if not os.path.exists(lbl_path):
            continue

        image  = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        waldo_boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                if cls == WALDO_CLASS:
                    x1, y1, x2, y2 = yolo_to_pixels(cx, cy, bw, bh, img_w, img_h)
                    waldo_boxes.append((x1, y1, x2, y2))

                    cx_px = int(cx * img_w)
                    cy_px = int(cy * img_h)
                    patch = crop_centered(image, cx_px, cy_px, PATCH_SIZE, img_w, img_h)
                    patch = patch.resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)
                    patch.save(os.path.join(out_waldo, f"{stem}_w{waldo_count}.jpg"))
                    waldo_count += 1

        for i in range(NOTWALDO_PER_IMAGE):
            patch = random_notwaldo_crop(image, waldo_boxes, PATCH_SIZE, img_w, img_h)
            if patch:
                patch = patch.resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)
                patch.save(os.path.join(out_notwaldo, f"{stem}_nw{i}.jpg"))
                notwaldo_count += 1

    print(f"  Waldo patches    : {waldo_count}")
    print(f"  NotWaldo patches : {notwaldo_count}")
    print(f"  Done.")


if __name__ == "__main__":
    extract()
