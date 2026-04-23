"""
Where's Waldo? — Sliding Window Bounding Box Localization
==========================================================
Reuses the trained WaldoCNN classifier to locate Waldo in a full scene image
by sliding a window across the image at multiple scales.

Run:
    python waldo_sliding_window.py --image scene.jpg
    python waldo_sliding_window.py --image scene.jpg --threshold 0.8 --output result.jpg
"""

import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from waldo_cnn import WaldoCNN, VAL_TRANSFORM


# =============================================
# NON-MAXIMUM SUPPRESSION
# =============================================

def compute_iou(box_a, box_b):
    """Compute Intersection over Union for two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / (union + 1e-6)


def nms(detections, iou_threshold=0.3):
    """
    Non-Maximum Suppression — removes overlapping boxes, keeps highest confidence.

    Args:
        detections : list of (x1, y1, x2, y2, confidence)
        iou_threshold : boxes with IoU above this are suppressed

    Returns:
        list of kept detections
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d[4], reverse=True)
    kept = []

    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [d for d in detections if compute_iou(best[:4], d[:4]) < iou_threshold]

    return kept


# =============================================
# SLIDING WINDOW
# =============================================

def sliding_window(image, model, device, threshold=0.7, step_ratio=0.25, scales=(1.0, 0.75, 0.5)):
    """
    Slides a 256×256 window across the image at multiple scales.

    Args:
        image      : PIL Image (full scene)
        model      : loaded WaldoCNN
        device     : torch device
        threshold  : minimum confidence to keep a detection
        step_ratio : stride as a fraction of window size (0.25 = 25% overlap)
        scales     : image scale factors to search at

    Returns:
        list of (x1, y1, x2, y2, confidence) in original image coordinates
    """
    orig_w, orig_h = image.size
    window_size = 256
    detections = []

    for scale in scales:
        scaled_w = int(orig_w * scale)
        scaled_h = int(orig_h * scale)
        scaled_img = image.resize((scaled_w, scaled_h), Image.BILINEAR)

        step = int(window_size * step_ratio)
        step = max(step, 1)

        for y in range(0, scaled_h - window_size + 1, step):
            for x in range(0, scaled_w - window_size + 1, step):
                patch = scaled_img.crop((x, y, x + window_size, y + window_size))
                tensor = VAL_TRANSFORM(patch).unsqueeze(0).to(device)

                with torch.no_grad():
                    prob = torch.sigmoid(model(tensor)).item()

                if prob >= threshold:
                    # Map coordinates back to original image size
                    x1 = int(x / scale)
                    y1 = int(y / scale)
                    x2 = int((x + window_size) / scale)
                    y2 = int((y + window_size) / scale)
                    detections.append((x1, y1, x2, y2, prob))

    return detections


# =============================================
# DRAW RESULTS
# =============================================

def draw_boxes(image, detections, output_path):
    """Draws bounding boxes on the image and saves it."""
    draw = ImageDraw.Draw(image)

    for x1, y1, x2, y2, conf in detections:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        label = f"Waldo {conf:.0%}"
        draw.rectangle([x1, y1 - 20, x1 + len(label) * 8, y1], fill="red")
        draw.text((x1 + 2, y1 - 18), label, fill="white")

    image.save(output_path)
    print(f"  Saved: {output_path}")


# =============================================
# MAIN
# =============================================

def locate_waldo(
    image_path,
    model_path = "waldo_cnn_best.pth",
    threshold  = 0.75,
    iou_threshold = 0.3,
    step_ratio = 0.25,
    scales     = (1.0, 0.75, 0.5),
    output_path = None,
    device     = None,
):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"\n{'='*60}")
    print(f"  Waldo Sliding Window Localization")
    print(f"  Image     : {image_path}")
    print(f"  Threshold : {threshold}")
    print(f"  Scales    : {scales}")
    print(f"  Device    : {device}")
    print(f"{'='*60}")

    model = WaldoCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    image = Image.open(image_path).convert("RGB")
    print(f"  Image size: {image.size[0]}×{image.size[1]}")

    print("  Running sliding window...")
    detections = sliding_window(image, model, device, threshold, step_ratio, scales)
    print(f"  Raw detections : {len(detections)}")

    detections = nms(detections, iou_threshold)
    print(f"  After NMS      : {len(detections)}")

    if not detections:
        print("  Waldo not found. Try lowering --threshold.")
        return []

    for i, (x1, y1, x2, y2, conf) in enumerate(detections):
        print(f"  [{i+1}] Box: ({x1}, {y1}) -> ({x2}, {y2})  Confidence: {conf:.1%}")

    if output_path is None:
        base = image_path.rsplit(".", 1)[0]
        output_path = f"{base}_waldo_detected.jpg"

    draw_boxes(image.copy(), detections, output_path)
    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     type=str, required=True,          help="Path to full scene image")
    parser.add_argument("--model",     type=str, default="waldo_cnn_best.pth")
    parser.add_argument("--threshold", type=float, default=0.75,         help="Minimum confidence (0-1)")
    parser.add_argument("--iou",       type=float, default=0.3,          help="NMS IoU threshold")
    parser.add_argument("--step",      type=float, default=0.25,         help="Sliding window step ratio")
    parser.add_argument("--output",    type=str, default=None,           help="Output image path")
    args = parser.parse_args()

    locate_waldo(
        image_path    = args.image,
        model_path    = args.model,
        threshold     = args.threshold,
        iou_threshold = args.iou,
        step_ratio    = args.step,
        output_path   = args.output,
    )
