# 🐾 Food Detector - Train Your Own Model

A web app that lets you train an AI image classifier on **your own food
photos** (e.g. downloaded from Kaggle) and then use it to detect foods in
new images.

Unlike the original version, this app does **not** rely on a generic
pretrained ImageNet model. Instead, it trains a fresh classifier on the
exact food classes and photos that **you** provide - so it can recognize
any foods you want, not just the ~100 ImageNet categories.

---

## 🚀 Quick Start

### 1. Install & run

**Windows:**
- Double-click `setup_and_run.bat`

**Mac/Linux:**
```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

The first run installs Python libraries (Flask, PyTorch, etc.) - this can
take a few minutes. Once you see:

```
Running on http://127.0.0.1:5000
```

open **http://localhost:5000** in your browser. Keep the terminal window
open while using the app.

---

## 📁 2. Add Your Dataset (no folder-per-food needed)

Put **all** your photos into a single folder:

```
food_detector/
└── dataset/
    └── images/
        ├── cat_0001.jpg
        ├── cat_0002.jpg
        ├── dog_0001.jpg
        ├── dog_0002.jpg
        ├── lion_0001.jpg
        └── ...
```

You can add **as many images and as many different foods as you want** -
there's no limit. This is perfect for large Kaggle datasets where all images
are dumped into one folder.

### How the app knows which food is in each photo

There are two options:

**Option A - `labels.csv` (recommended for Kaggle datasets)**

If your Kaggle dataset comes with a CSV mapping filenames to labels, copy or
rename it to `dataset/labels.csv` with at least these two columns:

```csv
filename,label
cat_0001.jpg,cat
cat_0002.jpg,cat
dog_0001.jpg,dog
lion_0001.jpg,lion
```

(Column names `file`/`image` and `class`/`food` are also accepted.)

**Option B - filename prefix (no CSV needed)**

If `dataset/labels.csv` does not exist, the app figures out the label from
the start of each filename - everything before the first `_`, `-`, `.`, or
digit:

| Filename          | Detected label |
|-------------------|----------------|
| `cat_0001.jpg`    | `cat`          |
| `Dog-12.png`      | `dog`          |
| `lion.045.jpg`    | `lion`         |
| `Tiger23.jpg`     | `tiger`        |

So if your Kaggle images are already named like `cat.123.jpg`, `dog.456.jpg`,
etc., you can just drop them straight into `dataset/images/` with no extra
setup.

---

## 🧠 3. Train Your Model

1. Open the app at http://localhost:5000
2. Go to the **"Train Model"** tab
3. Check the **Dataset Summary** - it should show all your food classes and
   how many images each has
4. (Optional) Adjust **Epochs** and **Batch size**:
   - More epochs = better accuracy but longer training time
   - Larger batch size = faster training but needs more memory
   - Defaults (10 epochs, batch size 16) work well for most datasets
5. Click **"🚀 Start Training"**

Training runs in the background and shows live progress. When it finishes,
the model is saved to:

```
models/food_model.pth
models/class_names.json
```

You can also train from the command line instead:

```bash
python train.py --epochs 15 --batch-size 32
```

---

## 🔍 4. Detect Foods

Once training is complete, go to the **"Detect"** tab:

1. Upload or drag-and-drop an image
2. Click **"🔍 Detect"**
3. The app shows the predicted food, confidence %, and the top predictions
   from your trained classes

---

## 🔁 Re-training

You can add more photos to `dataset/images/` (and update `labels.csv` if
you're using it) at any time, then click **"Start Training"** again. The new
model will replace the old one.

---

## 🛠️ How It Works (Technical Notes)

- Uses **transfer learning** on **MobileNetV3-Large** (pretrained on
  ImageNet) - only the final classification layer is retrained, which is
  fast even on a normal CPU.
- Images are automatically split into training (85%) and validation (15%)
  sets.
- Data augmentation (random crops, flips, color jitter) is applied during
  training to improve generalization.
- The best model (highest validation accuracy) is saved automatically.

---

## 📦 Project Structure

```
food_detector/
├── app.py                 # Flask web server
├── train.py                # Training script (transfer learning)
├── detector.py              # Loads trained model & runs predictions
├── requirements.txt
├── setup_and_run.bat        # Windows setup/run script
├── setup_and_run.sh          # Mac/Linux setup/run script
├── dataset/
│   ├── images/              # <-- put ALL your photos here
│   └── labels.csv            # optional: filename,label
├── models/                   # trained model saved here automatically
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/main.js
```

---

## ❓ Troubleshooting

- **"Need at least 2 different food classes"** - make sure
  `dataset/images/` has photos of at least 2 different foods, and that
  labels are detected correctly (check the Dataset Summary).
- **Training is slow** - this is expected on a CPU, especially with large
  datasets. Reduce epochs or batch size if needed, or be patient - it only
  needs to be done once.
- **"No trained model found yet"** when detecting - train a model first
  using the "Train Model" tab.
