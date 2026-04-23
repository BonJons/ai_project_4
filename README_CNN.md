# Waldo CNN Classifier — Complete Guide

This guide walks through training and running a CNN-based Waldo classifier from scratch, including downloading Roboflow data, training, and inference on full scene images.

---

## Table of Contents

1. [Setup](#setup)
2. [Download Roboflow Data](#download-roboflow-data)
3. [Extract Training Patches](#extract-training-patches)
4. [Train the CNN](#train-the-cnn)
5. [Run Inference (Sliding Window Localization)](#run-inference-sliding-window-localization)
6. [Troubleshooting](#troubleshooting)

---

## Setup

### 1. Install Python & Dependencies

Make sure you have **Python 3.8+** installed:

```bash
python3 --version
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

This installs:
- `torch` & `torchvision` — PyTorch for neural networks
- `Pillow` — image processing
- `scikit-learn` — train/val/test splitting
- `tqdm` — progress bars
- `ultralytics` — YOLOv8 (optional, for comparison)

---

## Download Roboflow Data

### 1. Get Roboflow API Key

1. Go to [Roboflow](https://roboflow.com/)
2. Create an account or log in
3. Find your dataset (e.g., `ait_project4`)
4. Navigate to **Versions** → latest version
5. Click **Export** → **Roboflow API**
6. Copy the download command

### 2. Download Dataset

Run the Roboflow download command. It will extract to a local directory, typically:

```
~/ait_project4/data/raw/dataset/
```

The directory structure should be:

```
~/ait_project4/data/raw/dataset/
├── images/          # All original images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/          # YOLO format labels (one .txt per image)
    ├── image1.txt
    ├── image2.txt
    └── ...
```

Each `.txt` file has lines: `<class> <cx> <cy> <width> <height>` (normalized YOLO format)

---

## Extract Training Patches

The script `extract_roboflow_patches.py` reads the raw Roboflow images and labels, then crops:
- **Waldo patches**: 256×256 crops centered on each Waldo bounding box
- **Non-Waldo patches**: random 256×256 crops that don't overlap Waldo boxes

### Path Configuration

Edit the paths in `extract_roboflow_patches.py` if your Roboflow data is in a different location:

```python
SRC_IMAGES = "/home/jon_s/ait_project4/data/raw/dataset/images"
SRC_LABELS = "/home/jon_s/ait_project4/data/raw/dataset/labels"
```

### Run Extraction

```bash
./venv/bin/python extract_roboflow_patches.py
```

**Output:**
- `data_roboflow/waldo/`    — cropped Waldo patches
- `data_roboflow/notwaldo/` — random non-Waldo patches

---

## Train the CNN

The `waldo_cnn.py` script trains a 5-block CNN classifier from scratch.

### Default Configuration

- **Data source**: `data_roboflow/` (or edit the `data_root` parameter)
- **Epochs**: 50
- **Batch size**: 16
- **Learning rate**: 1e-3
- **Loss**: BCEWithLogitsLoss (binary classification)
- **Device**: Auto-detect (CUDA if available, else CPU)

### Run Training

```bash
./venv/bin/python waldo_cnn.py
```

**Expected output:**

```
============================================================
  Waldo CNN — Classifier Training
  Epochs: 50
============================================================

[Splits]
  Train : 447 samples  (359 waldo)
  Val   : 97 samples   (80 waldo)
  Test  : 96 samples   (77 waldo)

Ep 001/050 | Train: 0.6821 | Val: 0.5234 | Accuracy: 0.876 | Precision: 0.891 | Recall: 0.834 | F1: 0.862
Ep 002/050 | Train: 0.4321 | Val: 0.3891 | Accuracy: 0.901 | Precision: 0.912 | Recall: 0.887 | F1: 0.899
...
  Saved best model (F1=0.9234)

============================================================
  Final Test Set Evaluation
============================================================
  Accuracy: 0.927 | Precision: 0.934 | Recall: 0.921 | F1: 0.927
```

### Output

- **Model weights**: `waldo_cnn_best.pth` — saved whenever F1 score improves
- **Training log**: printed to console

---

## Run Inference (Sliding Window Localization)

Once the model is trained, use `waldo_sliding_window.py` to locate Waldo in full scene images.

### How It Works

1. Load the trained CNN model
2. Slide a 256×256 window across the scene image at multiple scales (1.0, 0.75, 0.5)
3. Test each window with the classifier
4. Keep detections above the confidence threshold
5. Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
6. Draw bounding boxes on the output image

### Basic Usage

```bash
./venv/bin/python waldo_sliding_window.py --image scenes/scene_1.jpg
```

**Output:**

```
============================================================
  Waldo Sliding Window Localization
  Image     : scenes/scene_1.jpg
  Threshold : 0.75
  Scales    : (1.0, 0.75, 0.5)
  Device    : cpu
============================================================
  Image size: 1024×1024
  Running sliding window...
  Raw detections : 3
  After NMS      : 2
  [1] Box: (128, 256) -> (384, 512)  Confidence: 89.2%
  [2] Box: (640, 512) -> (896, 768)  Confidence: 76.5%
  Saved: scenes/scene_1_waldo_detected.jpg
```

### Advanced Options

```bash
# Lower threshold for more detections (higher false positives)
./venv/bin/python waldo_sliding_window.py --image scenes/scene_1.jpg --threshold 0.5

# Specify model path
./venv/bin/python waldo_sliding_window.py --image scenes/scene_1.jpg --model waldo_cnn_best.pth

# Specify output image path
./venv/bin/python waldo_sliding_window.py --image scenes/scene_1.jpg --output result.jpg

# Adjust NMS overlap threshold (default 0.3)
./venv/bin/python waldo_sliding_window.py --image scenes/scene_1.jpg --iou 0.5

# Adjust step ratio (0.25 = 25% overlap, default)
./venv/bin/python waldo_sliding_window.py --image scenes/scene_1.jpg --step 0.1
```

### Output

- **Detected image with boxes**: `scenes/scene_{id}_waldo_detected.jpg`
- **Console log**: detections, confidence scores, coordinates

---

## Troubleshooting

### No detections found?

**Problem**: `Raw detections: 0, After NMS: 0`

**Causes & Solutions**:

1. **Threshold too high**  
   The model may not be confident enough. Try a lower threshold:
   ```bash
   ./venv/bin/python waldo_sliding_window.py --image scenes/scene_1.jpg --threshold 0.4
   ```

2. **Model not trained on this data**  
   If the training data (`data_roboflow/`) is different from the inference images, the model may not recognize them. Retrain using the correct data source.

3. **Model is untrained**  
   Ensure `waldo_cnn_best.pth` exists and was created by the training script.

### Bounding boxes too large?

The sliding-window approach uses fixed 256×256 patches. Each detection box represents the full patch where the classifier found Waldo, not a precise object boundary. This is expected behavior.

### Out of memory?

Reduce `batch_size` in `waldo_cnn.py` (e.g., `batch_size=8`) or the step ratio in `waldo_sliding_window.py` (e.g., `--step 0.5`).

### GPU not detected?

Check CUDA installation:
```bash
./venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, training will fall back to CPU (slower but still works).

---

## Complete Workflow (Quick Reference)

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Download Roboflow data (manually or via API)
# => ~/ait_project4/data/raw/dataset/

# 3. Extract patches
./venv/bin/python extract_roboflow_patches.py

# 4. Train CNN
./venv/bin/python waldo_cnn.py

# 5. Test on scenes
./venv/bin/python waldo_sliding_window.py --image scenes/scene_1.jpg --threshold 0.5

# Output: scenes/scene_1_waldo_detected.jpg
```

---

## Model Architecture

The CNN consists of 5 convolutional blocks with spatial reduction:

```
Input (256×256) 
  ↓
Block 1: Conv → Conv → MaxPool(2) ... 128×128, 32 channels
Block 2: Conv → Conv → MaxPool(2) ... 64×64, 64 channels
Block 3: Conv → Conv → MaxPool(2) ... 32×32, 128 channels
Block 4: Conv → Conv → MaxPool(2) ... 16×16, 256 channels
Block 5: Conv → Conv → MaxPool(2) ... 8×8, 256 channels
  ↓
Global Average Pooling → 256-dim
  ↓
FC(256 → 128) → ReLU → Dropout → FC(128 → 1)
  ↓
Output: raw logit → sigmoid for probability [0, 1]
```

- **Total params**: ~1.9M
- **Loss**: Binary Cross-Entropy with Logits
- **Training strategy**: Weighted sampling to balance Waldo/non-Waldo class imbalance

---

## Files Reference

| File | Purpose |
|------|---------|
| `waldo_cnn.py` | Main training script |
| `extract_roboflow_patches.py` | Converts Roboflow data to patches |
| `waldo_sliding_window.py` | Localization on full scenes |
| `waldo_cnn_best.pth` | Trained model weights |
| `requirements.txt` | Python dependencies |
| `data_roboflow/` | Extracted training patches |
| `scenes/` | Output from stitching; input to sliding window |

---

## References

- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [Roboflow Documentation](https://docs.roboflow.com/)
- [ImageNet Normalization](https://pytorch.org/vision/stable/models.html) — used for `VAL_TRANSFORM`
