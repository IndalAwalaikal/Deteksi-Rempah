# train_model.py

import tensorflow as tf
import os
import numpy as np
from PIL import Image

# --- Konfigurasi ---
MODEL_PATH = "model/rempah_detection_model.h5"
DATA_DIR = "data"
CLASS_NAMES = [
    "adas", "andaliman", "asam jawa", "bawang bombai", "bawang merah",
    "bawang putih", "biji ketumbar", "bukan rempah", "bunga lawang",
    "cengkeh", "daun jeruk", "daun kemangi", "daun ketumbar", "daun salam",
    "jahe", "jinten", "kapulaga", "kayu manis", "kayu secang", "kemiri",
    "kemukus", "kencur", "kluwek", "kunyit", "lada", "lengkuas", "pala",
    "saffron", "serai", "vanili", "wijen"
]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

os.makedirs("model", exist_ok=True)


def is_dataset_valid():
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    if not os.path.exists(train_dir):
        print(f"[!] Folder tidak ditemukan: {train_dir}")
        return False
    if not os.path.exists(val_dir):
        print(f"[!] Folder tidak ditemukan: {val_dir}")
        return False
    train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    val_classes = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    if len(train_classes) == 0:
        print("[!] Tidak ada kelas di data/train/")
        return False
    if len(val_classes) == 0:
        print("[!] Tidak ada kelas di data/val/")
        return False
    return True


def is_tf_readable(filepath):
    try:
        raw = tf.io.read_file(filepath)
        tf.io.decode_image(raw, channels=3)
        return True
    except:
        return False


def convert_and_filter_images(folder):
    for root, _, files in os.walk(folder):
        for fname in files:
            filepath = os.path.join(root, fname)
            name, ext = os.path.splitext(fname.lower())
            if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    os.remove(filepath)
                    print(f"Non-image format dihapus: {filepath}")
                except:
                    pass
                continue
            if not is_tf_readable(filepath):
                try:
                    with Image.open(filepath) as img:
                        rgb_img = img.convert("RGB")
                        new_path = os.path.join(root, name + ".jpg")
                        rgb_img.save(new_path, "JPEG", quality=95)
                        os.remove(filepath)
                        print(f"Perbaiki: {filepath} → {new_path}")
                except Exception as e:
                    print(f"Gagal perbaiki {filepath}: {e}")
                    if os.path.exists(filepath):
                        os.remove(filepath)


def train_model():
    if not is_dataset_valid():
        print("[x] Dataset tidak valid. Pelatihan dibatalkan.")
        return

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    print("Memperbaiki dan membersihkan gambar...")
    convert_and_filter_images(train_dir)
    convert_and_filter_images(val_dir)

    print("Memuat dataset...")
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            batch_size=BATCH_SIZE,
            image_size=IMG_SIZE,
            shuffle=True,
            seed=42
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            batch_size=BATCH_SIZE,
            image_size=IMG_SIZE,
            shuffle=False,
            seed=42
        )
    except Exception as e:
        print(f"Gagal muat dataset: {e}")
        return

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    print("Membangun model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Uji coba satu batch...")
    try:
        for _ in train_ds.take(1):
            pass
    except Exception as e:
        print(f"Error saat iterasi batch: {e}")
        return

    print("Mulai pelatihan...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

    model.save(MODEL_PATH, save_format='h5')
    print(f"[✓] Model berhasil disimpan: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()