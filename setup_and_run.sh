#!/bin/bash
echo "============================================"
echo "   Food Detector - Setup and Run Script"
echo "============================================"
echo ""

echo "[1/3] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo ""
echo "[2/3] Installing libraries (this may take a while the first time)..."
pip install flask pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "[3/3] Starting the app..."
echo ""
echo "============================================"
echo " Open this link in your browser:"
echo " http://localhost:5000"
echo "============================================"
echo ""
echo " TIP: To train your own model first, put your photos in"
echo " dataset/images/ and use the 'Train Model' tab in the app"
echo " (or run: python train.py)"
echo ""

python app.py
