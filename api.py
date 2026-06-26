"""
Indonesian Spice Detection API - Production Ready Version
Flask-based REST API for spice image classification with enhanced security and performance.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import io
import hashlib
from datetime import datetime
from functools import lru_cache
from loguru import logger
from config import settings, create_directories

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, origins=settings.CORS_ORIGINS)

# Configure rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[settings.RATE_LIMIT],
    storage_uri="memory://"
)

# Configure logging
logger.add(
    settings.LOG_FILE,
    rotation="500 MB",
    retention="10 days",
    level=settings.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Global variables
model = None
model_loaded = False


class ImageValidator:
    """Validates uploaded images for security and quality."""
    
    ALLOWED_EXTENSIONS = set(settings.ALLOWED_EXTENSIONS)
    MAX_SIZE = settings.MAX_UPLOAD_SIZE
    MIN_DIMENSION = 50
    MAX_DIMENSION = 4096
    
    @staticmethod
    def allowed_file(filename: str) -> bool:
        """Check if file extension is allowed."""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ImageValidator.ALLOWED_EXTENSIONS
    
    @staticmethod
    def validate_size(file_size: int) -> bool:
        """Check if file size is within limits."""
        return 0 < file_size <= ImageValidator.MAX_SIZE
    
    @staticmethod
    def validate_image(image: Image.Image) -> tuple[bool, str]:
        """Validate image dimensions and format."""
        width, height = image.size
        
        if width < ImageValidator.MIN_DIMENSION or height < ImageValidator.MIN_DIMENSION:
            return False, f"Image too small. Minimum dimensions: {ImageValidator.MIN_DIMENSION}x{ImageValidator.MIN_DIMENSION}"
        
        if width > ImageValidator.MAX_DIMENSION or height > ImageValidator.MAX_DIMENSION:
            return False, f"Image too large. Maximum dimensions: {ImageValidator.MAX_DIMENSION}x{ImageValidator.MAX_DIMENSION}"
        
        # Check for valid image mode
        if image.mode not in ('RGB', 'RGBA', 'L'):
            return False, f"Unsupported image mode: {image.mode}"
        
        return True, "Valid"


class SpiceClassifier:
    """Handles model loading and prediction."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.class_names = settings.CLASS_NAMES
        self.img_size = settings.IMG_SIZE
    
    def load_model(self) -> bool:
        """Load the TensorFlow model."""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            logger.info(f"Loading model from {self.model_path}...")
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            logger.success("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model input."""
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(self.img_size, Image.Resampling.LANCZOS)
        
        # Normalize pixel values
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        return np.expand_dims(image_array, axis=0)
    
    def predict(self, image: Image.Image) -> dict:
        """Make prediction on an image."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Preprocess
        processed = self.preprocess_image(image)
        
        # Predict
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Get top prediction
        class_idx = int(np.argmax(predictions))
        confidence = float(predictions[class_idx]) * 100
        
        # Get top 5 predictions
        top_indices = np.argsort(predictions)[::-1][:5]
        top_predictions = [
            {
                "class": self.class_names[idx],
                "confidence": round(float(predictions[idx]) * 100, 2)
            }
            for idx in top_indices
        ]
        
        return {
            "className": self.class_names[class_idx],
            "confidence": round(confidence, 2),
            "topPredictions": top_predictions,
            "allPredictions": predictions.tolist()
        }


# Initialize classifier
classifier = SpiceClassifier(settings.MODEL_PATH)


def get_model_cache_key(image_data: bytes) -> str:
    """Generate cache key for image."""
    return hashlib.md5(image_data).hexdigest()


@lru_cache(maxsize=128)
def cached_predict(image_hash: str, image_bytes: bytes) -> dict:
    """Cached prediction function."""
    image = Image.open(io.BytesIO(image_bytes))
    return classifier.predict(image)


@app.before_first_request
def initialize():
    """Initialize application before first request."""
    create_directories()
    
    logger.info("Initializing Spice Detection API...")
    
    if not classifier.load_model():
        logger.error("Failed to initialize. Model not available.")
        raise RuntimeError("Model initialization failed. Please train the model first.")
    
    logger.info("API initialization complete")


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.APP_VERSION,
        "model_loaded": classifier.model is not None
    })


@app.route("/api/v1/predict", methods=["POST"])
@limiter.limit("20 per minute")
def predict_api():
    """
    Predict spice type from uploaded image.
    
    Request:
        - image: File (multipart/form-data)
    
    Response:
        - className: str - Predicted spice class
        - confidence: float - Confidence score (0-100)
        - topPredictions: list - Top 5 predictions
        - processingTime: float - Time taken in seconds
    """
    start_time = datetime.now()
    
    # Validate file presence
    if "image" not in request.files:
        logger.warning("No image file in request")
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    
    # Validate filename
    if file.filename == "":
        logger.warning("Empty filename provided")
        return jsonify({"error": "No file selected"}), 400
    
    if not ImageValidator.allowed_file(file.filename):
        logger.warning(f"Invalid file type: {file.filename}")
        return jsonify({"error": "Invalid file type. Allowed: JPG, PNG, WebP"}), 400
    
    # Read file data
    try:
        file_data = file.read()
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return jsonify({"error": "Failed to read file"}), 500
    
    # Validate file size
    if not ImageValidator.validate_size(len(file_data)):
        logger.warning(f"File size invalid: {len(file_data)} bytes")
        return jsonify({"error": f"File size must be between 0 and {settings.MAX_UPLOAD_SIZE // (1024*1024)}MB"}), 400
    
    # Load and validate image
    try:
        image = Image.open(io.BytesIO(file_data))
        is_valid, message = ImageValidator.validate_image(image)
        if not is_valid:
            logger.warning(f"Image validation failed: {message}")
            return jsonify({"error": message}), 400
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return jsonify({"error": "Invalid or corrupted image file"}), 400
    
    # Make prediction
    try:
        result = classifier.predict(image)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        result["processingTime"] = round(processing_time, 3)
        
        logger.info(f"Prediction: {result['className']} ({result['confidence']}%) in {processing_time:.3f}s")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/v1/classes", methods=["GET"])
def get_classes():
    """Get list of all supported spice classes."""
    return jsonify({
        "classes": settings.CLASS_NAMES,
        "count": len(settings.CLASS_NAMES)
    })


@app.route("/", methods=["GET"])
def home():
    """Serve the main HTML page."""
    return send_from_directory(settings.BASE_DIR, "index.html")


@app.route("/<path:path>", methods=["GET"])
def static_files(path):
    """Serve static files."""
    if os.path.exists(os.path.join(settings.BASE_DIR, path)):
        return send_from_directory(settings.BASE_DIR, path)
    return jsonify({"error": "File not found"}), 404


@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded."""
    logger.warning(f"Rate limit exceeded: {request.remote_addr}")
    return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal error: {e}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Create directories
    create_directories()
    
    # Log startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Running on http://{settings.API_HOST}:{settings.API_PORT}")
    
    # Run with production settings
    app.run(
        host=settings.API_HOST,
        port=settings.API_PORT,
        debug=settings.DEBUG,
        threaded=True
    )