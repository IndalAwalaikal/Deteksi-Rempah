# 🐳 Panduan Docker - Indonesian Spice Detection System

## Prasyarat

Pastikan Anda telah menginstal:
- **Docker** (versi 20.10 atau lebih baru)
- **Docker Compose** (versi 2.0 atau lebih baru)

### Cek Instalasi
```bash
docker --version
docker compose version
```

## Quick Start

### 1. Clone Repository (jika belum)
```bash
git clone <repository-url>
cd indonesian-spice-detection
```

### 2. Konfigurasi Environment (Opsional)
```bash
cp .env.example .env
# Edit .env sesuai kebutuhan Anda
```

### 3. Jalankan Semua Service
```bash
docker compose up -d
```

Ini akan menjalankan 3 service:
- **API Backend** (Flask) - Port 5000
- **Streamlit UI** - Port 8501
- **Frontend Nginx** - Port 80

### 4. Akses Aplikasi

| Service | URL | Deskripsi |
|---------|-----|-----------|
| Frontend Web | http://localhost | Interface HTML modern |
| API Backend | http://localhost:5000 | REST API |
| Streamlit UI | http://localhost:8501 | Interface Streamlit |
| API Docs | http://localhost:5000/docs | Dokumentasi API |

### 5. Monitoring Logs
```bash
# Lihat semua logs
docker compose logs -f

# Lihat log service tertentu
docker compose logs -f api
docker compose logs -f streamlit
docker compose logs -f frontend
```

## Perintah Docker Compose Umum

### Menjalankan Aplikasi
```bash
# Jalankan di background (recommended)
docker compose up -d

# Jalankan di foreground (untuk debugging)
docker compose up
```

### Menghentikan Aplikasi
```bash
# Hentikan semua service
docker compose down

# Hentikan dan hapus volumes (hati-hati: data akan hilang!)
docker compose down -v
```

### Restart Service
```bash
# Restart semua service
docker compose restart

# Restart service tertentu
docker compose restart api
```

### Rebuild Image (setelah perubahan kode)
```bash
# Build ulang dan restart
docker compose up -d --build

# Force rebuild tanpa cache
docker compose build --no-cache
```

### Melihat Status
```bash
# Lihat status semua container
docker compose ps

# Lihat resource usage
docker stats
```

### Masuk ke Container
```bash
# Masuk ke container API
docker compose exec api bash

# Masuk ke container Streamlit
docker compose exec streamlit bash

# Masuk ke container Nginx
docker compose exec frontend sh
```

## Struktur Service

### 1. API Service (`api`)
- **Framework**: Flask + Gunicorn
- **Port**: 5000
- **Fungsi**: REST API untuk deteksi rempah
- **Endpoints**:
  - `GET /health` - Health check
  - `POST /api/predict` - Prediksi gambar
  - `GET /api/classes` - Daftar kelas rempah

### 2. Streamlit Service (`streamlit`)
- **Framework**: Streamlit
- **Port**: 8501
- **Fungsi**: UI interaktif untuk deteksi
- **Fitur**: Upload gambar, visualisasi hasil, confidence score

### 3. Frontend Service (`frontend`)
- **Web Server**: Nginx
- **Port**: 80
- **Fungsi**: Serve static HTML/CSS/JS
- **Fitur**: Proxy ke API, compression, security headers

## Volume & Persistensi

Docker Compose ini menggunakan bind mounts untuk:
- `./model` - Menyimpan model trained
- `./data` - Dataset untuk training
- `./logs` - Log aplikasi

**Catatan**: Model dan data akan tersimpan di host machine, sehingga tidak hilang saat container direstart.

## Troubleshooting

### Container Tidak Start
```bash
# Cek logs untuk error
docker compose logs api

# Pastikan port tidak digunakan oleh aplikasi lain
lsof -i :5000
lsof -i :8501
lsof -i :80
```

### Model Tidak Ditemukan
Model akan otomatis diunduh saat pertama kali berjalan. Jika gagal:
```bash
# Download manual
docker compose exec api python app/utils/download_model.py
```

### Out of Memory
TensorFlow membutuhkan RAM besar. Solusi:
```bash
# Batasi memory di docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Performance Lambat
- Pastikan menggunakan GPU jika tersedia (butuh nvidia-docker)
- Reduce batch size di `.env`
- Gunakan model yang sudah trained

## Development Mode

Untuk development dengan hot reload:

```yaml
# docker-compose.dev.yml (buat file baru)
services:
  api:
    volumes:
      - ./app:/app/app
    command: flask run --host=0.0.0.0 --port=5000 --debug
  
  streamlit:
    volumes:
      - ./app:/app/app
    command: streamlit run app/deteksi_rempah_app.py --server.headless=true
```

Jalankan dengan:
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## Security Best Practices

1. **Jangan commit .env** - File ini berisi secret keys
2. **Ganti SECRET_KEY** - Di production, gunakan key yang kuat
3. **Enable HTTPS** - Gunakan reverse proxy dengan SSL
4. **Update Image** - Selalu gunakan versi terbaru
5. **Scan Vulnerabilities**:
   ```bash
   docker scan spice-detection-api
   ```

## Production Deployment

Untuk production, pertimbangkan:
- Gunakan Docker Swarm atau Kubernetes
- Enable SSL/TLS dengan Let's Encrypt
- Setup monitoring (Prometheus + Grafana)
- Configure backup untuk model dan data
- Use secrets management (Docker Secrets)

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Indonesian Spice Detection README](README.md)

## Support

Jika mengalami masalah:
1. Cek logs: `docker compose logs -f`
2. Baca dokumentasi: `README.md`
3. Buka issue di repository

---
**Version**: 2.0.0  
**Last Updated**: 2024
