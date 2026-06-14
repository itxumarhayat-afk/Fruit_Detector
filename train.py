"""
train.py
--------
Trains an image classifier on YOUR OWN food photos using transfer
learning (MobileNetV3-Large pretrained on ImageNet).

HOW TO USE
1. Put ALL your photos in one folder:

       dataset/images/
           pizza_0001.jpg
           pizza_0002.jpg
           burger_0001.jpg
           burger_0002.jpg
           sushi_0001.jpg
           ...

   You can have as many images and as many different food items as you want.

2. Labels (which food is in each photo) come from one of two places:

   OPTION A - labels.csv (recommended, works with any filenames)
       Put a file dataset/labels.csv with two columns:

           filename,label
           pizza_0001.jpg,pizza
           pizza_0002.jpg,pizza
           burger_0001.jpg,burger
           sushi_0001.jpg,sushi

       Many Kaggle datasets already come with a CSV like this - just
       rename/copy it to dataset/labels.csv.

   OPTION B - filename prefix (no CSV needed)
       If dataset/labels.csv does NOT exist, the label is taken from the
       start of the filename, up to the first '_', '-', '.', or digit.

           cat_0001.jpg   -> "cat"
           dog.045.png    -> "dog"
           lion-12.jpg    -> "lion"
           Tiger23.jpg    -> "tiger"

3. Run this script:
       python train.py

   Optional arguments:
       python train.py --epochs 15 --batch-size 32

4. When training finishes, the model is saved to:
       models/food_model.pth
       models/class_names.json

   The web app (app.py) will automatically use this trained model.
"""

import os
import re
import csv
import json
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_CSV = os.path.join(DATASET_DIR, "labels.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

IMG_SIZE = 224
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Labeling helpers
# ---------------------------------------------------------------------------

def label_from_filename(filename):
    """
    Extract a label from a filename when no labels.csv is provided.

    Takes everything before the first '_', '-', '.', or digit, and
    lowercases it.

        "pizza_0001.jpg"  -> "pizza"
        "Burger-12.png"    -> "burger"
        "sushi.045.jpg"  -> "sushi"
        "Tacos23.jpg"   -> "tacos"
    """
    name = os.path.splitext(filename)[0]
    match = re.match(r"^([A-Za-z]+)", name)
    if not match:
        return None
    return match.group(1).lower()


def load_labels():
    """
    Build a dict {filename: label} for every image in dataset/images/.

    Uses dataset/labels.csv if present, otherwise falls back to
    filename-prefix labeling.

    Returns (labels_dict, used_csv: bool)
    """
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(
            f"Could not find '{IMAGES_DIR}'. Put all your photos inside dataset/images/."
        )

    all_files = [f for f in os.listdir(IMAGES_DIR)
                  if f.lower().endswith(IMAGE_EXTENSIONS)]

    if len(all_files) == 0:
        raise FileNotFoundError(f"No images found in '{IMAGES_DIR}'.")

    labels = {}

    if os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # normalize column names (case-insensitive)
            fieldmap = {k.lower().strip(): k for k in reader.fieldnames or []}
            fname_col = fieldmap.get("filename") or fieldmap.get("file") or fieldmap.get("image")
            label_col = fieldmap.get("label") or fieldmap.get("class") or fieldmap.get("food")

            if not fname_col or not label_col:
                raise ValueError(
                    "labels.csv must have a 'filename' column and a 'label' column "
                    f"(found columns: {reader.fieldnames})"
                )

            for row in reader:
                fname = row[fname_col].strip()
                label = row[label_col].strip().lower()
                if fname and label:
                    labels[fname] = label

        # Only keep entries for files that actually exist
        labels = {f: l for f, l in labels.items() if f in set(all_files)}
        if len(labels) == 0:
            raise ValueError(
                "labels.csv was found but none of its filenames match files in dataset/images/."
            )
        return labels, True

    # Fallback: derive label from filename prefix
    for fname in all_files:
        label = label_from_filename(fname)
        if label:
            labels[fname] = label

    if len(labels) == 0:
        raise ValueError(
            "Could not determine labels from filenames. Either rename your files so "
            "they start with the food name (e.g. pizza_0001.jpg), or add a "
            "dataset/labels.csv file with 'filename,label' columns."
        )

    return labels, False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FoodDataset(Dataset):
    """Dataset that reads images from dataset/images/ using a {filename: label} map."""

    def __init__(self, items, class_to_idx, transform):
        # items: list of (filename, label)
        self.items = items
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        filename, label = self.items[idx]
        path = os.path.join(IMAGES_DIR, filename)
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, self.class_to_idx[label]


def get_dataset_classes():
    """
    Return a sorted list of {"name": label, "count": n} for the current
    dataset/images/ + labels (csv or filename-based).

    Returns an empty list if the dataset isn't set up yet.
    """
    try:
        labels, _ = load_labels()
    except Exception:
        return []

    counts = {}
    for label in labels.values():
        counts[label] = counts.get(label, 0) + 1

    return [{"name": name, "count": counts[name]} for name in sorted(counts.keys())]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes):
    """Load pretrained MobileNetV3-Large and replace the final classifier layer."""
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

    # Freeze the feature extractor (transfer learning)
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the final classifier layer with one matching our number of classes
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run(epochs=10, batch_size=16, learning_rate=0.001, progress_callback=None):
    """
    Train the model on dataset/images/ + labels and save it to models/.

    progress_callback(percent, message) is called periodically so a web UI
    can show live progress. percent is 0-100.
    """

    def log(percent, message):
        print(f"[{percent:5.1f}%] {message}")
        if progress_callback:
            progress_callback(percent, message)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(1, f"Using device: {device}")

    # ---- 1. Load labels ----
    labels_map, used_csv = load_labels()
    source = "labels.csv" if used_csv else "filenames"
    log(2, f"Loaded {len(labels_map)} labeled images (labels from {source})")

    class_names = sorted(set(labels_map.values()))
    if len(class_names) < 2:
        raise ValueError(
            f"Need at least 2 different food classes, but only found: {class_names}. "
            "Add more photos of other foods to dataset/images/."
        )

    class_to_idx = {name: i for i, name in enumerate(class_names)}

    counts = {}
    for label in labels_map.values():
        counts[label] = counts.get(label, 0) + 1
    log(3, f"Found {len(class_names)} classes: "
            + ", ".join(f"{name} ({counts[name]} imgs)" for name in class_names))

    items = list(labels_map.items())  # [(filename, label), ...]

    # ---- 2. Split train/val ----
    val_fraction = 0.15
    val_size = max(1, int(len(items) * val_fraction))
    train_size = len(items) - val_size

    if train_size <= 0:
        raise ValueError("Not enough images to train. Add more photos.")

    train_split, val_split = random_split(items, [train_size, val_size])
    train_items = [items[i] for i in train_split.indices]
    val_items = [items[i] for i in val_split.indices]

    train_dataset = FoodDataset(train_items, class_to_idx, TRAIN_TRANSFORM)
    val_dataset = FoodDataset(val_items, class_to_idx, VAL_TRANSFORM)

    log(5, f"Training images: {len(train_dataset)}, Validation images: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ---- 3. Build model ----
    log(8, "Building model (MobileNetV3-Large, pretrained)...")
    model = build_model(num_classes=len(class_names))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # ---- 4. Training loop ----
    best_val_acc = 0.0
    os.makedirs(MODELS_DIR, exist_ok=True)
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == targets).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        # ---- Validation ----
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += images.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        percent = 10 + int(85 * (epoch + 1) / epochs)
        log(percent,
            f"Epoch {epoch + 1}/{epochs} - "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.2%}, "
            f"val_acc: {val_acc:.2%}")

        # Save the best model based on validation accuracy
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "food_model.pth"))

    # ---- 5. Save class names ----
    with open(os.path.join(MODELS_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    elapsed = time.time() - start_time
    log(100, f"Done! Best validation accuracy: {best_val_acc:.2%} "
             f"(trained in {elapsed/60:.1f} min). Model saved to models/food_model.pth")

    return {
        "classes": class_names,
        "best_val_acc": best_val_acc,
        "epochs": epochs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train your custom food detector")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    args = parser.parse_args()

    run(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
