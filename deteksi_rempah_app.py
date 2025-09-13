# deteksi_rempah_app.py - DIPERBAIKI UNTUK KONSISTENSI & KEAMANAN

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

# --- Fungsi: Cek Validitas Model ---
def is_model_valid(model_path):
    if not os.path.exists(model_path):
        return False
    try:
        tf.keras.models.load_model(model_path, compile=False)
        return True
    except Exception as e:
        st.warning(f"Model tidak bisa dimuat: {e}")
        return False

# --- Fungsi: Latih dari Dataset Lokal ---
def train_from_local_data():
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    if not (os.path.exists(train_dir) and os.path.exists(val_dir)):
        st.error(f"Folder dataset tidak lengkap! Harus ada:\n- `{train_dir}`\n- `{val_dir}`")
        return None

    st.info(f"Dataset ditemukan. Melatih dari: `{train_dir}`")

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

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

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

    with st.spinner("Sedang melatih model dari dataset lokal..."):
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

    model.save(MODEL_PATH)
    st.success(f"Model berhasil dilatih dan disimpan di: `{MODEL_PATH}`")
    return model

# --- Fungsi: Load Model (dengan fix opsional jika perlu) ---
def load_model_with_fix():
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.warning(f"Gagal muat model standar: {e}. Mencoba dengan custom objects...")
        try:
            class SafeInputLayer(tf.keras.layers.InputLayer):
                def __init__(self, *args, **kwargs):
                    kwargs.pop('batch_shape', None)
                    kwargs.pop('dtype', None)
                    super().__init__(*args, **kwargs)

            custom_objects = {
                'InputLayer': SafeInputLayer,
                'DTypePolicy': type('DTypePolicy', (), {})
            }
            return tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
        except Exception as e2:
            st.error(f"Gagal total muat model: {e2}")
            return None

# --- Fungsi Utama: Load atau Latih Model ---
@st.cache_resource
def load_or_train_model():
    model = train_from_local_data()
    if model is not None:
        return model

    if is_model_valid(MODEL_PATH):
        st.info("Memuat model yang sudah dilatih sebelumnya...")
        return load_model_with_fix()

    st.error("""
    Tidak dapat menjalankan aplikasi karena:
    - Dataset tidak ditemukan, DAN
    - Model `.h5` tidak ada atau rusak.
    
    Silakan pastikan:
    1. Ada folder `data/train/` dan `data/val/`, atau
    2. File `model/rempah_detection_model.h5` ada dan valid.
    """)
    st.stop()

# --- Preprocessing & Prediksi ---
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def predict(model, img):
    return model.predict(img)

# --- UI Streamlit ---
st.set_page_config(page_title="Deteksi Rempah Indonesia", page_icon="🌿", layout="centered")
st.title("Deteksi Rempah Indonesia")
st.write("Unggah gambar rempah, dan AI akan mengenali jenisnya.")

uploaded_file = st.file_uploader("Pilih gambar rempah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    with st.spinner("Menganalisis..."):
        model = load_or_train_model()
        processed = preprocess_image(image)
        preds = predict(model, processed)

    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]

    st.success(f"Hasil: **{CLASS_NAMES[class_idx]}**")
    st.metric("Kepercayaan", f"{confidence * 100:.2f}%")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=CLASS_NAMES, y=preds[0], ax=ax, palette="viridis")
    ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=8)
    ax.set_ylabel("Kepercayaan")
    ax.set_title("Prediksi untuk Setiap Kelas")
    plt.tight_layout()
    st.pyplot(fig)

    st.info("""
    Catatan:
    - Aplikasi ini akan melatih dari dataset lokal jika tersedia.
    - Jika tidak, akan coba muat model `.h5`.
    - Pastikan struktur folder benar:
      ```
      project/
      ├── data/
      │   ├── train/
      │   └── val/
      └── model/
          └── rempah_detection_model.h5
      ```
    """)
