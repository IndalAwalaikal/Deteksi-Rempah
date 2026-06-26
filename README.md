# 🌿 Indonesian Spice Detection System

## Overview

An advanced AI-powered system for detecting and classifying 31 types of Indonesian spices using deep learning. Built with production-ready practices including transfer learning, comprehensive security measures, and modern web interfaces.

## ✨ Key Features

### Model Architecture
- **Transfer Learning**: Uses MobileNetV2 pre-trained on ImageNet for superior accuracy
- **Two-Phase Training**: Initial training with frozen base + fine-tuning for optimal performance
- **Data Augmentation**: Real-time augmentation for better generalization
- **Early Stopping**: Prevents overfitting with automatic training termination
- **Top-K Accuracy**: Tracks both standard and top-3 accuracy metrics

### API Features
- **RESTful Design**: Clean, versioned API endpoints (`/api/v1/`)
- **Rate Limiting**: Protection against abuse (20 requests/minute default)
- **Input Validation**: Comprehensive file type, size, and dimension checks
- **CORS Support**: Configurable cross-origin resource sharing
- **Health Checks**: `/health` endpoint for monitoring
- **Structured Logging**: Loguru-based logging with rotation

### Security
- Input validation for all uploads
- File size limits (10MB default)
- Image dimension validation
- Rate limiting per IP
- Environment-based configuration
- No hardcoded credentials

## 📁 Project Structure

```
/workspace/
├── api.py                      # Production Flask API server
├── train_model.py              # Advanced model training script
├── deteksi_rempah_app.py       # Streamlit web application
├── config.py                   # Centralized configuration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
├── index.html                  # Frontend landing page
├── style.css                   # Frontend styles
├── script.js                   # Frontend JavaScript
└── logs/                       # Application logs (auto-created)
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
cp .env.example .env
```

### 2. Prepare Dataset

```
data/
├── train/
│   ├── adas/
│   └── ... (31 classes)
└── val/
    ├── adas/
    └── ...
```

### 3. Train & Run

```bash
python train_model.py
python api.py
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/classes` | Get spice classes |
| POST | `/api/v1/predict` | Predict from image |

## 🔒 Security

- Never commit `.env` files
- Change SECRET_KEY in production
- Enable HTTPS when deploying

---

**Built with ❤️ for preserving Indonesian culinary heritage**
