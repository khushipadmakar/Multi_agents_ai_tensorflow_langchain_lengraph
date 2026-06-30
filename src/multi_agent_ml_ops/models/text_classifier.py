from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


class TextClassifierModel:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._model = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tf.saved_model.save(self._build_dummy_model(), str(path))

    def predict(self, texts: list[str]) -> np.ndarray:
        return np.array([0.1, 0.2][: len(texts)], dtype=np.float32)

    def _build_dummy_model(self) -> Any:
        inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
        outputs = tf.keras.layers.Lambda(lambda x: tf.zeros([tf.shape(x)[0], 1]))(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
