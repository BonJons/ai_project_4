# Waldo? — Binary Classifier

This project trains a binary image classifier to detect Waldo in 256×256 image patches using two approaches: a custom CNN built from scratch, and a fine-tuned YOLOv8 classifier.

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
- `ultralytics` — YOLOv8

---

## Dataset

The dataset should already be organized into two folders under `data_roboflow/`:

```
data_roboflow/
├── waldo/        # 256×256 patches containing Waldo
│   ├── img1.jpg
│   └── ...
└── notwaldo/     # 256×256 patches not containing Waldo
    ├── img1.jpg
    └── ...
```

Both training scripts read directly from this structure.

---

## Train the CNN

`waldo_cnn.py` trains a 5-block CNN from scratch on the dataset.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_root` | `data_roboflow/` | Path to dataset |
| `epochs` | `50` | Number of training epochs |
| `batch_size` | `16` | Batch size |
| `lr` | `1e-3` | Learning rate |
| `save_path` | `waldo_cnn_best.pth` | Where to save best model weights |

### Run Training

```bash
python waldo_cnn.py
```

### Expected Output

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
...
  Saved best model (F1=0.9234)

============================================================
  Final Test Set Evaluation
============================================================
  Accuracy: 0.927 | Precision: 0.934 | Recall: 0.921 | F1: 0.927
```

### Output

- **Model weights**: `waldo_cnn_best.pth` — saved whenever validation F1 improves

### Model Architecture

The CNN uses 5 convolutional blocks with progressive spatial reduction:

```
Input (256×256)
  ↓
Block 1: Conv → Conv → MaxPool(2) ... 128×128, 32 channels
Block 2: Conv → Conv → MaxPool(2) ... 64×64,  64 channels
Block 3: Conv → Conv → MaxPool(2) ... 32×32,  128 channels
Block 4: Conv → Conv → MaxPool(2) ... 16×16,  256 channels
Block 5: Conv → Conv → MaxPool(2) ... 8×8,    256 channels
  ↓
Global Average Pooling → 256-dim
  ↓
FC(256 → 128) → ReLU → Dropout → FC(128 → 1)
  ↓
Output: raw logit → sigmoid → probability [0, 1]
```

- **Total params**: ~2.38M
- **Loss**: Binary Cross-Entropy with Logits
- **Class balancing**: Weighted random sampler (automatically oversamples whichever class is the minority)

---

## Train the YOLO Classifier

`waldo_yolo.py` fine-tunes a YOLOv8n classification model on the same dataset.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_ROOT` | `data_roboflow/` | Path to source dataset |
| `YOLO_DIR` | `yolo_data/` | Where prepared YOLO data is written |
| `epochs` | `30` | Number of training epochs |
| `batch` | `16` | Batch size |
| `imgsz` | `256` | Input image size |

### Run Training

```bash
python waldo_yolo.py
```

This will automatically:
1. Reformat the dataset into the folder structure YOLOv8 expects (`yolo_data/`)
2. Oversample the minority class to balance training
3. Train the model
4. Evaluate on the test set and print metrics

### Expected Output

```
[Prep] Building YOLO dataset -> yolo_data/
  waldo     : train=X | val=X | test=X
  notwaldo  : train=X | val=X | test=X
  Oversampled notwaldo train: X -> X (matches X majority class)

============================================================
  Waldo YOLO - Classifier Training
  Epochs: 30
============================================================
...

============================================================
  Final Test Set Evaluation
============================================================
  Acc: 0.XXX | Precision: 0.XXX | Recall: 0.XXX | F1: 0.XXX
```

### Output

- **Model weights**: `runs/classify/waldo_yolo/weights/best.pt`

---

## Files Reference

| File | Purpose |
|------|---------|
| `waldo_cnn.py` | CNN training and evaluation |
| `waldo_yolo.py` | YOLOv8 training and evaluation |
| `requirements.txt` | Python dependencies |
| `data_roboflow/` | Dataset (waldo/ and notwaldo/ folders) |
| `waldo_cnn_best.pth` | Best CNN model weights (created after training) |
| `runs/` | YOLO training output and weights (created after training) |
