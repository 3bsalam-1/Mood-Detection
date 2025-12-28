"""Inference utilities for Mood Detection.

Provides the MoodPredictor class which loads a Keras model and exposes
predict_from_array and predict_from_file helpers.
"""
from __future__ import annotations

import os
from typing import Dict, Any

import cv2
import numpy as np
from tensorflow.keras.models import load_model


class MoodPredictor:
    """Load a trained mood model and run inference.

    Args:
        model_path: Path to a Keras .h5 model file.
    """

    def __init__(self, model_path: str = "models/mood.h5"):
        self.model_path = model_path
        self.model = None

    def load(self) -> None:
        """Load the Keras model from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = load_model(self.model_path)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize and scale image to model input requirements.

        Returns a numpy array of shape (1, 256, 256, 3).
        """
        if image is None:
            raise ValueError("Input image is None")
        # Convert BGR (OpenCV) -> RGB
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = image.astype("float32") / 255.0
        return np.expand_dims(image, 0)

    def predict_from_array(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict mood from a numpy image array.

        Returns dict: {"mood": "Happy"|"Sad", "probability": float}
        """
        if self.model is None:
            self.load()
        x = self.preprocess(image)
        prob = float(self.model.predict(x, verbose=0)[0][0])
        mood = "Happy" if prob <= 0.5 else "Sad"
        confidence = (1 - prob) if prob <= 0.5 else prob
        return {"mood": mood, "probability": confidence}

    def predict_from_file(self, path: str) -> Dict[str, Any]:
        """Load image from file and predict."""
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Unable to read image: {path}")
        return self.predict_from_array(image)
