"""
Where's Waldo? — YOLOv8 Classifier
=====================================
Task    : Classify 256×256 patches as Waldo or Not Waldo
Metrics : Accuracy, Precision, Recall, F1

Run:
    python waldo_yolo.py
    python waldo_yolo.py --predict --image patch.jpg
"""

import os
import shutil
import argparse
from sklearn.model_selection import train_test_split


# =====================================
# DATASET PREPARATION
# =====================================

def prepare_dataset(
    data_root = "data/",
    out_dir   = "yolo_data/",
    val_size  = 0.15,
    test_size = 0.15,
    seed      = 42        # same seed as CNN scripts — identical splits
):
    """
    Converts waldo/ and notwaldo/ into the train/val/test folder structure expected by YOLOv8 classification.

    Uses the same seed and split sizes as CNN
    """
    print(f"\n[Prep] Building YOLO dataset -> {out_dir}")

    for split in ["train", "val", "test"]:
        for cls in ["waldo", "notwaldo"]:
            os.makedirs(os.path.join(out_dir, split, cls), exist_ok=True)

    for cls in ["waldo", "notwaldo"]:
        src_dir = os.path.join(data_root, cls)
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(f"Folder not found: {src_dir}")

        files = [f for f in os.listdir(src_dir)]

        train_f, test_f = train_test_split(
            files, test_size=test_size, random_state=seed)
        val_ratio = val_size / (1 - test_size)
        train_f, val_f = train_test_split(
            train_f, test_size=val_ratio, random_state=seed)

        for split, file_list in [("train", train_f),
                                  ("val",   val_f),
                                  ("test",  test_f)]:
            for fname in file_list:
                shutil.copy(
                    os.path.join(src_dir, fname),
                    os.path.join(out_dir, split, cls, fname)
                )

        print(f"  {cls:10s}: train={len(train_f)} | "
              f"val={len(val_f)} | test={len(test_f)}")

    # Oversample Waldo training patches to match Not Waldo count.
    # YOLO has no WeightedRandomSampler — oversampling is the equivalent fix.
    # Val and test are NOT oversampled — evaluation must reflect real distribution.
    waldo_train_dir   = os.path.join(out_dir, "train", "waldo")
    notwaldo_train_dir = os.path.join(out_dir, "train", "notwaldo")
    waldo_files   = [f for f in os.listdir(waldo_train_dir)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    notwaldo_count = len([f for f in os.listdir(notwaldo_train_dir)
                          if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    # Duplicate Waldo images until count matches Not Waldo
    import random
    random.seed(42)
    copies_needed = notwaldo_count - len(waldo_files)
    for i in range(copies_needed):
        src_fname = random.choice(waldo_files)
        src_path  = os.path.join(waldo_train_dir, src_fname)
        stem, ext = os.path.splitext(src_fname)
        dst_path  = os.path.join(waldo_train_dir, f"{stem}_aug{i}{ext}")
        shutil.copy(src_path, dst_path)

    final_waldo_count = len([f for f in os.listdir(waldo_train_dir)
                              if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"  Oversampled waldo train: {len(waldo_files)} -> {final_waldo_count} "
          f"(matches {notwaldo_count} notwaldo)")
    print(f"  Dataset ready at: {out_dir}")
    return out_dir


# =====================================
# TRAINING
# =====================================

def train(
    data_dir   = "yolo_data/",
    model_size = "yolov8n",
    epochs     = 30,
    imgsz      = 256,
    batch      = 16,
    project    = "runs",
    name       = "waldo_yolo",
):
    """
    Fine-tunes YOLOv8n-cls on the Waldo dataset. Augmentation mirrors the CNN for fair comparison
    """
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"  Waldo YOLO Classifier Training")
    print(f"  Epochs: {epochs}")
    print(f"{'='*60}")

    model = YOLO(f"{model_size}-cls.pt")

    model.train(
        data      = data_dir,
        epochs    = epochs,
        imgsz     = imgsz,
        batch     = batch,
        project   = project,
        name      = name,
        # Augmentation — mirrors CNN transforms
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        fliplr    = 0.5,
        flipud    = 0.2,
        degrees   = 10,
        translate = 0.08,
        scale     = 0.1,
        # Regularisation
        dropout   = 0.4,
        patience  = 15,       # early stopping
        verbose   = True,
    )

    # Ask YOLO directly where it saved — avoids path construction errors
    best_path = str(model.trainer.best)
    return best_path


# =====================================
# EVALUATION
# =====================================

# Evaluates test set and reports Accuracy, Precision, Recall, F1.
def evaluate(model_path, test_dir="yolo_data/test"):
    
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"  Final Test Set Evaluation")
    print(f"{'='*60}")

    model = YOLO(model_path)
    tp = fp = fn = tn = 0

    for cls in ["waldo", "notwaldo"]:
        cls_dir    = os.path.join(test_dir, cls)
        true_label = 1 if cls == "waldo" else 0

        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            results    = model.predict(
                os.path.join(cls_dir, fname), verbose=False)
            pred_name  = results[0].names[results[0].probs.top1]
            pred_label = 1 if pred_name == "waldo" else 0

            if   pred_label == 1 and true_label == 1: tp += 1
            elif pred_label == 1 and true_label == 0: fp += 1
            elif pred_label == 0 and true_label == 1: fn += 1
            else:                                      tn += 1

    acc  = (tp + tn) / (tp + fp + fn + tn + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)

    print(f"  Acc: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# =====================================
# INFERENCE
# =====================================

def predict(image_path, model_path, threshold=0.5):
    """Run classifier on a single 256×256 image"""
    from ultralytics import YOLO

    model   = YOLO(model_path)
    results = model.predict(image_path, verbose=False)[0]

    pred_name = results.names[results.probs.top1]
    prob      = float(results.probs.top1conf)

    if pred_name == "notwaldo":
        prob = 1 - prob

    is_waldo = pred_name == "waldo" and prob >= threshold
    return {
        "is_waldo" : is_waldo,
        "prob"     : prob,
        "label"    : "Waldo" if is_waldo else "Not Waldo"
    }


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--image",   type=str, default=None)
    parser.add_argument("--model",   type=str, default="runs/classify/waldo_yolo/weights/best.pt")
    args = parser.parse_args()

    DATA_ROOT = "data/"
    YOLO_DIR  = "yolo_data/"

    if args.predict:
        if not args.image:
            print("Error: --image required with --predict")
            exit(1)
        result = predict(args.image, args.model)
        print(f"\nPrediction: {result['label']} "
              f"(confidence: {result['prob']:.1%})")
    else:
        prepare_dataset(data_root=DATA_ROOT, out_dir=YOLO_DIR)

        best_model = train(
            data_dir   = YOLO_DIR,
            model_size = "yolov8n",
            epochs     = 30,
            imgsz      = 256,
            batch      = 16,
        )

        evaluate(best_model, test_dir=os.path.join(YOLO_DIR, "test"))
