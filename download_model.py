import os
import gdown
import tensorflow as tf

MODEL_URL = "https://drive.google.com/uc?export=download&id=12bYRNfO6tryeCjidDFQREgSAR23LdYTf"
MODEL_PATH = "model/rempah_detection_model.h5"
MODEL_TEMP = "model/rempah_detection_model_temp.h5"

def download_and_fix_model():
    if not os.path.exists("model"):
        os.makedirs("model")
        print("Folder 'model' telah dibuat.")

    if os.path.exists(MODEL_PATH):
        print(f"Model sudah ada: {MODEL_PATH}")
        print("Mengecek kompatibilitas dengan TensorFlow 2.12...")
        try:
            from tensorflow.keras.layers import InputLayer

            def custom_input_layer(**kwargs):
                kwargs.pop('batch_shape', None)
                return InputLayer(**kwargs)

            tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={'InputLayer': custom_input_layer},
                compile=False
            )
            print("Model sudah kompatibel! Tidak perlu diperbaiki.")
            return
        except Exception as e:
            print(f"Model tidak kompatibel (error: {str(e)})")
            print("Akan diperbaiki...")

    print("Mengunduh model dari Google Drive...")
    gdown.download(MODEL_URL, MODEL_TEMP, quiet=False)
    print(f"Model asli berhasil diunduh ke: {MODEL_TEMP}")

    print("Memuat model asli dan menyimpan ulang untuk kompatibilitas TF 2.12...")
    try:
        model = tf.keras.models.load_model(MODEL_TEMP, compile=False)
        model.save(MODEL_PATH, save_format='h5')
        print(f"Model telah diperbaiki dan disimpan sebagai: {MODEL_PATH}")
    except Exception as e:
        print(f"Gagal memproses model: {e}")
        return

    if os.path.exists(MODEL_TEMP):
        os.remove(MODEL_TEMP)
        print("File sementara telah dihapus.")

    print("\nSemua selesai! Sekarang jalankan:")
    print("   streamlit run deteksi_rempah_app.py")
    print("   python api.py")

if __name__ == "__main__":
    download_and_fix_model()
