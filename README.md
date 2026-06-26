# Indonesian Spice Detection System

A modern AI-powered system for identifying Indonesian spices using deep learning. This project combines cutting-edge machine learning with a professional web interface to provide accurate spice recognition.

## Features

- **AI-Powered Detection**: Uses Convolutional Neural Networks (CNN) with 95% accuracy
- **31+ Spice Types**: Recognizes common Indonesian spices including garlic, chili, ginger, turmeric, and more
- **Modern Web Interface**: Clean, professional UI with responsive design
- **Real-time Processing**: Results in under 3 seconds
- **RESTful API**: Production-ready API with rate limiting and security features

## Project Structure

```
/workspace
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── api.py              # Flask REST API
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py           # Configuration management
│   ├── models/                 # Model storage
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── train_model.py      # Model training script
│   │   ├── download_model.py   # Model download utility
│   │   ├── fix_model.py        # Model fixing utility
│   │   └── validate_images.py  # Image validation
│   └── deteksi_rempah_app.py   # Streamlit application
├── frontend/
│   ├── index.html              # Main HTML file
│   ├── css/
│   │   └── style.css           # Modern CSS styles
│   └── js/
│       └── script.js           # Frontend JavaScript
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── .env.example                # Environment variables template
├── .dockerignore               # Docker ignore file
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile                  # Docker image configuration
├── nginx.conf                  # Nginx configuration
├── DOCKER.md                   # Docker documentation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Quick Start

### Option 1: Using Docker (Recommended)

The easiest way to run the application is with Docker Compose:

1. Make sure Docker and Docker Compose are installed:
```bash
docker --version
docker compose version
```

2. Run the application:
```bash
docker compose up -d
```

3. Access the application:
- **Frontend Web**: http://localhost
- **API Backend**: http://localhost:5000
- **Streamlit UI**: http://localhost:8501

See [DOCKER.md](DOCKER.md) for detailed Docker instructions.

### Option 2: Manual Installation

#### Prerequisites

- Python 3.8+
- pip package manager
- Modern web browser

#### Installation

1. Clone the repository:
```bash
cd /workspace
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained model:
```bash
python app/utils/download_model.py
```

4. Run the application:
```bash
# Run Flask API
python app/api/api.py

# Or run Streamlit UI
streamlit run app/deteksi_rempah_app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend application |
| `/api/v1/predict` | POST | Upload image for spice detection |
| `/api/v1/classes` | GET | Get list of supported spice classes |
| `/health` | GET | Health check endpoint |

### Example API Usage

```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -F "image=@path/to/spice/image.jpg"
```

Response:
```json
{
  "className": "bawang putih",
  "confidence": 94.5,
  "topPredictions": [
    {"class": "bawang putih", "confidence": 94.5},
    {"class": "bawang bombai", "confidence": 3.2},
    {"class": "lada", "confidence": 1.1}
  ],
  "processingTime": 1.23
}
```

## Supported Spices

The system recognizes 31 types of Indonesian spices:
- Adas, Andaliman, Asam Jawa
- Bawang Bombai, Bawang Merah, Bawang Putih
- Biji Ketumbar, Bunga Lawang, Cengkeh
- Daun Jeruk, Daun Kemangi, Daun Ketumbar, Daun Salam
- Jahe, Jinten, Kapulaga, Kayu Manis, Kayu Secang
- Kemiri, Kemukus, Kencur, Kluwek, Kunyit, Lada
- Lengkuas, Pala, Saffron, Serai, Vanili, Wijen

## Technology Stack

### Machine Learning
- TensorFlow 2.15+
- Keras
- OpenCV
- NumPy

### Backend
- Python 3.8+
- Flask
- Pydantic Settings
- Loguru

### Frontend
- HTML5, CSS3, JavaScript
- Chart.js for visualization
- Modern responsive design

## Configuration

Environment variables can be set in `.env` file:

```env
DEBUG=false
API_HOST=0.0.0.0
API_PORT=5000
MODEL_PATH=model/rempah_detection_model.h5
MAX_UPLOAD_SIZE=10485760
RATE_LIMIT=20 per minute
```

## Training Your Own Model

To train a custom model:

```bash
python app/utils/train_model.py --data_dir path/to/data --epochs 15
```

## License

This project is open source and available under the MIT License.

## Contact

- Email: info@spicedetection.ai
- Location: Makassar, Indonesia

---

Built with ❤️ for preserving Indonesian culinary heritage
