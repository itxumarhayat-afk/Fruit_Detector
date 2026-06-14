import os
import uuid
import threading

from flask import Flask, render_template, request, jsonify

import detector
import train as train_module

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

# Global state for the background training job
training_state = {
    "running": False,
    "percent": 0,
    "message": "",
    "done": False,
    "error": None,
    "result": None,
}
training_lock = threading.Lock()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dataset-info")
def dataset_info():
    """Return the list of classes (folders) found in dataset/ and their image counts."""
    classes = train_module.get_dataset_classes()
    return jsonify({
        "classes": classes,
        "total_images": sum(c["count"] for c in classes),
        "model_trained": detector.model_is_trained(),
    })


@app.route("/train", methods=["POST"])
def start_training():
    """Start training in a background thread so the request returns immediately."""
    with training_lock:
        if training_state["running"]:
            return jsonify({"error": "Training is already running."}), 400

        classes = train_module.get_dataset_classes()
        if len(classes) < 2:
            return jsonify({
                "error": "You need at least 2 class folders inside 'dataset/', "
                         "each containing images. "
                         f"Currently found: {[c['name'] for c in classes]}"
            }), 400

        body = request.get_json(silent=True) or {}
        epochs = int(body.get("epochs", 10))
        batch_size = int(body.get("batch_size", 16))

        training_state.update({
            "running": True,
            "percent": 0,
            "message": "Starting...",
            "done": False,
            "error": None,
            "result": None,
        })

    def progress_callback(percent, message):
        with training_lock:
            training_state["percent"] = percent
            training_state["message"] = message

    def run_training():
        try:
            result = train_module.run(
                epochs=epochs,
                batch_size=batch_size,
                progress_callback=progress_callback,
            )
            detector.reload_model()
            with training_lock:
                training_state["running"] = False
                training_state["done"] = True
                training_state["result"] = result
        except Exception as e:
            with training_lock:
                training_state["running"] = False
                training_state["error"] = str(e)
                training_state["message"] = f"Error: {e}"

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return jsonify({"started": True})


@app.route("/train-status")
def train_status():
    with training_lock:
        return jsonify(dict(training_state))


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image was uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Allowed formats: PNG, JPG, JPEG, GIF, BMP, WEBP"}), 400

    if not detector.model_is_trained():
        return jsonify({
            "success": False,
            "error": "No trained model found yet. Please train your model first "
                     "in the 'Train Model' tab."
        }), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        result = detector.predict_image(filepath)
        result["image_url"] = f"/static/uploads/{filename}"
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    app.run(debug=True, port=5000)
