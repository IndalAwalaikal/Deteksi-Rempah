"""
Indonesian Spice Detection - Advanced Model Training Script
Uses transfer learning with MobileNetV2 for better accuracy and performance.
Includes data augmentation, early stopping, and comprehensive metrics.
"""

import tensorflow as tf
import os
import numpy as np
from PIL import Image
import json
from datetime import datetime
from loguru import logger
from config import settings, create_directories
from tqdm import tqdm


class DataPreprocessor:
    """Handles data validation and preprocessing."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, "train")
        self.val_dir = os.path.join(data_dir, "val")
    
    def is_dataset_valid(self) -> bool:
        """Validate dataset structure."""
        if not os.path.exists(self.train_dir):
            logger.error(f"Training folder not found: {self.train_dir}")
            return False
        if not os.path.exists(self.val_dir):
            logger.error(f"Validation folder not found: {self.val_dir}")
            return False
        
        train_classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        val_classes = [d for d in os.listdir(self.val_dir) if os.path.isdir(os.path.join(self.val_dir, d))]
        
        if len(train_classes) == 0:
            logger.error("No classes found in training directory")
            return False
        if len(val_classes) == 0:
            logger.error("No classes found in validation directory")
            return False
        
        logger.info(f"Dataset valid: {len(train_classes)} classes found")
        return True
    
    def is_tf_readable(self, filepath: str) -> bool:
        """Check if TensorFlow can read the image."""
        try:
            raw = tf.io.read_file(filepath)
            tf.io.decode_image(raw, channels=3)
            return True
        except:
            return False
    
    def convert_and_filter_images(self, folder: str):
        """Clean and convert images to consistent format."""
        logger.info(f"Processing images in {folder}...")
        processed = 0
        removed = 0
        
        for root, _, files in os.walk(folder):
            for fname in tqdm(files, desc=f"Processing {os.path.basename(folder)}"):
                filepath = os.path.join(root, fname)
                name, ext = os.path.splitext(fname.lower())
                
                if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    try:
                        os.remove(filepath)
                        removed += 1
                    except:
                        pass
                    continue
                
                if not self.is_tf_readable(filepath):
                    try:
                        with Image.open(filepath) as img:
                            rgb_img = img.convert("RGB")
                            new_path = os.path.join(root, name + ".jpg")
                            rgb_img.save(new_path, "JPEG", quality=95)
                            os.remove(filepath)
                            processed += 1
                    except Exception as e:
                        logger.warning(f"Failed to process {filepath}: {e}")
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            removed += 1
        
        logger.success(f"Processed: {processed}, Removed: {removed}")


class SpiceModelTrainer:
    """Advanced model trainer with transfer learning."""
    
    def __init__(self):
        self.class_names = settings.CLASS_NAMES
        self.num_classes = len(self.class_names)
        self.img_size = settings.IMG_SIZE
        self.batch_size = settings.BATCH_SIZE
        self.epochs = settings.EPOCHS
        self.learning_rate = settings.LEARNING_RATE
        self.dropout_rate = settings.DROPOUT_RATE
        self.model_path = settings.MODEL_PATH
        self.data_dir = settings.DATA_DIR
        self.model = None
        self.history = None
    
    def create_datasets(self):
        """Create training and validation datasets."""
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")
        
        logger.info("Loading training dataset...")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            batch_size=self.batch_size,
            image_size=self.img_size,
            shuffle=True,
            seed=42,
            interpolation='bilinear'
        )
        
        logger.info("Loading validation dataset...")
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            batch_size=self.batch_size,
            image_size=self.img_size,
            shuffle=False,
            seed=42,
            interpolation='bilinear'
        )
        
        # Cache datasets for performance
        train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds
    
    def create_data_augmentation(self):
        """Create data augmentation layer."""
        if not settings.AUGMENTATION_ENABLED:
            return tf.keras.Sequential([tf.keras.layers.Rescaling(1./255)])
        
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.Rescaling(1./255)
        ])
    
    def build_model(self) -> tf.keras.Model:
        """Build model using transfer learning with MobileNetV2."""
        logger.info("Building model with MobileNetV2 backbone...")
        
        # Create base model from MobileNetV2 (pre-trained on ImageNet)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.img_size + (3,),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Build the complete model
        inputs = tf.keras.Input(shape=self.img_size + (3,))
        
        # Data augmentation
        x = self.create_data_augmentation()(inputs)
        
        # Base model (feature extractor)
        x = base_model(x, training=False)
        
        # Regularization
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
        )
        
        logger.info(f"Model built successfully with {model.count_params():,} parameters")
        return model
    
    def build_fine_tune_model(self, base_model: tf.keras.Model, fine_tune_at: int = 100) -> tf.keras.Model:
        """Unfreeze some layers for fine-tuning."""
        # Unfreeze layers for fine-tuning
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True
        
        # Rebuild model
        inputs = tf.keras.Input(shape=self.img_size + (3,))
        x = self.create_data_augmentation()(inputs)
        x = base_model(x, training=True)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Use lower learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate * 0.1),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
        )
        
        logger.info(f"Fine-tuning model with {model.count_params():,} trainable parameters")
        return model
    
    def train(self):
        """Execute the complete training pipeline."""
        # Validate dataset
        preprocessor = DataPreprocessor(self.data_dir)
        if not preprocessor.is_dataset_valid():
            logger.error("Dataset validation failed. Training aborted.")
            return None
        
        # Clean and preprocess images
        preprocessor.convert_and_filter_images(os.path.join(self.data_dir, "train"))
        preprocessor.convert_and_filter_images(os.path.join(self.data_dir, "val"))
        
        # Create directories
        create_directories()
        
        # Load datasets
        train_ds, val_ds = self.create_datasets()
        
        # Build initial model (transfer learning)
        self.model = self.build_model()
        
        # Setup callbacks
        callbacks = self._get_callbacks()
        
        # Initial training (frozen base)
        logger.info("=" * 50)
        logger.info("Phase 1: Training with frozen base model")
        logger.info("=" * 50)
        
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs // 2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning phase
        logger.info("=" * 50)
        logger.info("Phase 2: Fine-tuning top layers")
        logger.info("=" * 50)
        
        # Get base model
        base_model = self.model.get_layer(index=1)  # MobileNetV2 is the second layer after augmentation
        
        # Build fine-tuning model
        self.model = self.build_fine_tune_model(base_model)
        
        # Continue training with fine-tuning
        fine_tune_history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs // 2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Merge histories
        for key in self.history.history:
            self.history.history[key].extend(fine_tune_history.history[key])
        
        # Save model
        self._save_model()
        
        # Save training history
        self._save_history()
        
        return self.model
    
    def _get_callbacks(self):
        """Create training callbacks."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=settings.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"model/best_model_{timestamp}.h5",
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                f"logs/training_history_{timestamp}.csv",
                append=False
            )
        ]
    
    def _save_model(self):
        """Save the trained model."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path, save_format='h5')
        logger.success(f"Model saved to: {self.model_path}")
        
        # Save class labels
        labels_path = os.path.join(os.path.dirname(self.model_path), "class_labels.json")
        with open(labels_path, 'w') as f:
            json.dump({
                "classes": self.class_names,
                "num_classes": self.num_classes,
                "img_size": self.img_size,
                "created_at": datetime.now().isoformat()
            }, f, indent=2)
        logger.info(f"Class labels saved to: {labels_path}")
    
    def _save_history(self):
        """Save training history."""
        if self.history is None:
            return
        
        history_path = "logs/training_metrics.json"
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        with open(history_path, 'w') as f:
            json.dump({
                "metrics": self.history.history,
                "final_accuracy": float(self.history.history['accuracy'][-1]),
                "final_val_accuracy": float(self.history.history['val_accuracy'][-1]),
                "epochs_trained": len(self.history.history['loss'])
            }, f, indent=2)
        
        logger.info(f"Training history saved to: {history_path}")
        
        # Print final metrics
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Final Training Accuracy: {self.history.history['accuracy'][-1]*100:.2f}%")
        logger.info(f"Final Validation Accuracy: {self.history.history['val_accuracy'][-1]*100:.2f}%")
        logger.info(f"Best Validation Accuracy: {max(self.history.history['val_accuracy'])*100:.2f}%")


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("INDONESIAN SPICE DETECTION - MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Classes: {len(settings.CLASS_NAMES)}")
    logger.info(f"Image Size: {settings.IMG_SIZE}")
    logger.info(f"Batch Size: {settings.BATCH_SIZE}")
    logger.info(f"Epochs: {settings.EPOCHS}")
    logger.info(f"Learning Rate: {settings.LEARNING_RATE}")
    logger.info(f"Data Augmentation: {'Enabled' if settings.AUGMENTATION_ENABLED else 'Disabled'}")
    logger.info("=" * 60)
    
    trainer = SpiceModelTrainer()
    model = trainer.train()
    
    if model is not None:
        logger.success("Training completed successfully!")
    else:
        logger.error("Training failed!")


if __name__ == "__main__":
    main()