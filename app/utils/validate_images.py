# validate_images.py
import os
from PIL import Image

def validate_images_in_directory(root_dir):
    print(f"Memeriksa folder: {root_dir}")
    corrupted_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            filepath = os.path.join(dirpath, fname)
            try:
                img = Image.open(filepath)
                img.verify()  # Verifikasi struktur file
                img.close()
            except Exception as e:
                print(f"❌ Rusak: {filepath} → {e}")
                corrupted_files.append(filepath)

    return corrupted_files

if __name__ == "__main__":
    train_dir = "data/train"
    val_dir = "data/val"

    bad_files = []
    if os.path.exists(train_dir):
        bad_files += validate_images_in_directory(train_dir)
    else:
        print(f"[!] Folder {train_dir} tidak ditemukan")

    if os.path.exists(val_dir):
        bad_files += validate_images_in_directory(val_dir)
    else:
        print(f"[!] Folder {val_dir} tidak ditemukan")

    print("\n📋 Ringkasan:")
    if bad_files:
        print(f"🔴 Ditemukan {len(bad_files)} file gambar rusak!")
        for f in bad_files:
            print(f"   - {f}")
        print("\n💡 Silakan hapus atau perbaiki file-file di atas.")
    else:
        print("✅ Semua file gambar valid!")