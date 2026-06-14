"""
detector.py
-----------
Loads the model trained by train.py (models/food_model.pth +
models/class_names.json) and runs predictions on uploaded images.
"""

import os
import json

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "food_model.pth")
CLASSES_PATH = os.path.join(MODELS_DIR, "class_names.json")

IMG_SIZE = 224

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model = None
_class_names = None


def model_is_trained():
    """Check whether a trained model exists on disk."""
    return os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH)


def _build_model(num_classes):
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def load_model():
    """Load the trained model + class names (cached after first call)."""
    global _model, _class_names

    if _model is not None:
        return _model, _class_names

    if not model_is_trained():
        raise FileNotFoundError(
            "No trained model found. Please train your model first "
            "(see the 'Train Model' tab in the app, or run 'python train.py')."
        )

    with open(CLASSES_PATH, "r") as f:
        _class_names = json.load(f)

    _model = _build_model(num_classes=len(_class_names))
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    _model.load_state_dict(state_dict)
    _model.eval()

    return _model, _class_names


def reload_model():
    """Force a reload of the model (e.g. after training finishes)."""
    global _model, _class_names
    _model = None
    _class_names = None
    return load_model()


def get_emoji(label):
    """Best-effort emoji lookup for nicer UI - falls back to a fork & knife."""
    label = label.lower()
    emojis = {
        "pizza": "🍕", "burger": "🍔", "hamburger": "🍔", "sushi": "🍣",
        "taco": "🌮", "tacos": "🌮", "burrito": "🌯", "fries": "🍟",
        "hotdog": "🌭", "sandwich": "🥪", "salad": "🥗", "pasta": "🍝",
        "spaghetti": "🍝", "noodle": "🍜", "ramen": "🍜", "rice": "🍚",
        "curry": "🍛", "soup": "🍲", "steak": "🥩", "meat": "🥩",
        "chicken": "🍗", "egg": "🍳", "bacon": "🥓", "bread": "🍞",
        "croissant": "🥐", "bagel": "🥯", "pancake": "🥞", "waffle": "🧇",
        "cheese": "🧀", "donut": "🍩", "doughnut": "🍩", "cookie": "🍪",
        "cake": "🍰", "cupcake": "🧁", "pie": "🥧", "chocolate": "🍫",
        "icecream": "🍦", "ice_cream": "🍦",
        "popcorn": "🍿", "pretzel": "🥨", "apple": "🍎", "banana": "🍌",
        "grape": "🍇", "orange": "🍊", "strawberry": "🍓", "watermelon": "🍉",
        "pineapple": "🍍", "mango": "🥭", "peach": "🍑", "cherry": "🍒",
        "lemon": "🍋", "avocado": "🥑", "tomato": "🍅", "carrot": "🥕",
        "corn": "🌽", "broccoli": "🥦", "potato": "🥔", "pepper": "🌶️",
        "onion": "🧅", "garlic": "🧄", "cucumber": "🥒", "mushroom": "🍄",
        "shrimp": "🍤", "fish": "🐟", "crab": "🦀", "lobster": "🦞",
        "dumpling": "🥟", "bento": "🍱",
        "coffee": "☕", "tea": "🍵", "juice": "🧃", "milk": "🥛",
        "wine": "🍷", "beer": "🍺", "cocktail": "🍹",
    }
    for key, emoji in emojis.items():
        if key in label:
            return emoji
    return "🍽️"


def predict_image(image_path, top_k=5):
    """
    Run the trained model on an image and return predictions.

    Returns a dict:
        {
            "success": True,
            "top_prediction": "cat",
            "confidence": 92.4,
            "emoji": "🐈",
            "all_predictions": [
                {"label": "cat", "confidence": 92.4},
                {"label": "dog", "confidence": 5.1},
                ...
            ]
        }
    """
    model, class_names = load_model()

    img = Image.open(image_path).convert("RGB")
    img_tensor = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    k = min(top_k, len(class_names))
    top_probs, top_idx = torch.topk(probabilities, k)

    predictions = []
    for i in range(k):
        idx = top_idx[i].item()
        prob = top_probs[i].item()
        predictions.append({
            "label": class_names[idx],
            "confidence": round(prob * 100, 2),
        })

    return {
        "success": True,
        "top_prediction": predictions[0]["label"],
        "confidence": predictions[0]["confidence"],
        "emoji": get_emoji(predictions[0]["label"]),
        "all_predictions": predictions,
    }
