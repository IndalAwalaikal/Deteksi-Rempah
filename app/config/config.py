"""
Configuration settings for Indonesian Spice Detection System.
Centralized configuration management using environment variables.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = Field(default="Indonesian Spice Detection System", description="Application name")
    APP_VERSION: str = Field(default="2.0.0", description="Application version")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # Paths
    BASE_DIR: Path = Field(default=Path(__file__).parent, description="Base directory")
    MODEL_PATH: str = Field(default="model/rempah_detection_model.h5", description="Path to trained model")
    DATA_DIR: str = Field(default="data", description="Directory containing training data")
    
    # Model Configuration
    CLASS_NAMES: List[str] = Field(
        default=[
            "adas", "andaliman", "asam jawa", "bawang bombai", "bawang merah",
            "bawang putih", "biji ketumbar", "bukan rempah", "bunga lawang",
            "cengkeh", "daun jeruk", "daun kemangi", "daun ketumbar", "daun salam",
            "jahe", "jinten", "kapulaga", "kayu manis", "kayu secang", "kemiri",
            "kemukus", "kencur", "kluwek", "kunyit", "lada", "lengkuas", "pala",
            "saffron", "serai", "vanili", "wijen"
        ],
        description="List of spice class names"
    )
    IMG_SIZE: tuple = Field(default=(224, 224), description="Image size for model input")
    BATCH_SIZE: int = Field(default=32, description="Batch size for training")
    EPOCHS: int = Field(default=15, description="Number of training epochs")
    
    # Training Configuration
    LEARNING_RATE: float = Field(default=0.001, description="Learning rate for optimizer")
    DROPOUT_RATE: float = Field(default=0.5, description="Dropout rate for regularization")
    EARLY_STOPPING_PATIENCE: int = Field(default=5, description="Early stopping patience")
    AUGMENTATION_ENABLED: bool = Field(default=True, description="Enable data augmentation")
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=5000, description="API port")
    MAX_UPLOAD_SIZE: int = Field(default=10 * 1024 * 1024, description="Maximum upload size in bytes (10MB)")
    ALLOWED_EXTENSIONS: List[str] = Field(default=["jpg", "jpeg", "png", "webp"], description="Allowed file extensions")
    RATE_LIMIT: str = Field(default="10 per minute", description="API rate limit")
    
    # Security
    SECRET_KEY: str = Field(default_factory=lambda: os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production"))
    CORS_ORIGINS: List[str] = Field(default=["*"], description="Allowed CORS origins")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FILE: str = Field(default="logs/app.log", description="Log file path")
    
    # Performance
    NUM_WORKERS: int = Field(default=4, description="Number of worker processes")
    CACHE_ENABLED: bool = Field(default=True, description="Enable caching")
    CACHE_TTL: int = Field(default=300, description="Cache time-to-live in seconds")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        settings.BASE_DIR / "model",
        settings.BASE_DIR / "logs",
        settings.BASE_DIR / settings.DATA_DIR / "train",
        settings.BASE_DIR / settings.DATA_DIR / "val",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Print current configuration
    print(f"Application: {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Debug Mode: {settings.DEBUG}")
    print(f"Model Path: {settings.MODEL_PATH}")
    print(f"Number of Classes: {len(settings.CLASS_NAMES)}")
    print(f"Image Size: {settings.IMG_SIZE}")
    print(f"API Port: {settings.API_PORT}")
