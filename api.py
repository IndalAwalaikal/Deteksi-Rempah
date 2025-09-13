# api.py

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

MODEL_PATH = "model/rempah_detection_model.h5"
CLASS_NAMES = [
    "adas", "andaliman", "asam jawa", "bawang bombai", "bawang merah",
    "bawang putih", "biji ketumbar", "bukan rempah", "bunga lawang",
    "cengkeh", "daun jeruk", "daun kemangi", "daun ketumbar", "daun salam",
    "jahe", "jinten", "kapulaga", "kayu manis", "kayu secang", "kemiri",
    "kemukus", "kencur", "kluwek", "kunyit", "lada", "lengkuas", "pala",
    "saffron", "serai", "vanili", "wijen"
]
IMG_SIZE = (224, 224)

# --- Load Model Saat Startup ---
print("Memuat model... (pastikan sudah dilatih sebelumnya)")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model tidak ditemukan! Jalankan dulu: python train_model.py\nFile yang dicari: {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"[✓] Model berhasil dimuat dari {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Gagal memuat model: {e}")

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada gambar yang diunggah"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Tidak ada file yang dipilih"}), 400
    try:
        img = Image.open(file.stream).convert("RGB")
        processed = preprocess_image(img)
        predictions = model.predict(processed)[0]
        class_idx = np.argmax(predictions)
        confidence = float(predictions[class_idx]) * 100
        return jsonify({
            "className": CLASS_NAMES[class_idx],
            "confidence": round(confidence, 2),
            "predictions": predictions.tolist()
        })
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan saat memproses gambar: {str(e)}"}), 500

@app.route("/")
def home():
    return send_from_directory("", "index.html")

@app.route("/<path:path>")
def static_files(path):
    if os.path.exists(path):
        return send_from_directory("", path)
    return "File tidak ditemukan", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)