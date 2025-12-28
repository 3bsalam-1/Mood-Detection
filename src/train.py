"""Training utilities for Mood Detection model (scriptable).

This module mirrors the notebook training pipeline but is runnable as a script
for reproducible training and CI.
"""
from __future__ import annotations

import os
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    RandomFlip,
    RandomRotation,
    RandomZoom,
)
from tensorflow.keras.metrics import Precision, Recall


def build_model(input_shape: Tuple[int, int, int] = (256, 256, 3)) -> tf.keras.Model:
    model = Sequential([
        RandomFlip("horizontal", input_shape=input_shape),
        RandomRotation(0.15),
        RandomZoom(0.15),

        Conv2D(32, (3, 3), padding='same', activation='relu'),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.losses.BinaryCrossentropy(),
        metrics=['accuracy', Precision(), Recall()]
    )

    return model


def load_dataset(data_dir: str = 'mood', image_size=(256, 256), batch_size: int = 32):
    data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )
    # Normalize
    data = data.map(lambda x, y: (x / 255.0, y))

    # Split
    total = len(data)
    train_size = int(total * 0.7)
    val_size = int(total * 0.2)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size)

    return train, val, test


def train(model: tf.keras.Model, train_data, val_data, epochs: int = 50, model_out: str = 'models/mood.h5'):
    logdir = 'm_logs'
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    class_weights = {0: 1.1, 1: 0.9}

    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=[tensorboard_cb, early_stop],
        class_weight=class_weights,
        verbose=1
    )

    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save(model_out)

    return history


if __name__ == '__main__':
    m = build_model()
    train_data, val_data, test_data = load_dataset()
    train(m, train_data, val_data)
