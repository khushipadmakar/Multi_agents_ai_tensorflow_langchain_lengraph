from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class _DummyModel:
    def __init__(self, inputs=None, outputs=None) -> None:
        self.inputs = inputs
        self.outputs = outputs


class _InputLayer:
    def __init__(self, shape=None, dtype=None) -> None:
        self.shape = shape
        self.dtype = dtype


class _LambdaLayer:
    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, value: Any) -> Any:
        return self.fn(value)


class _LayersModule:
    @staticmethod
    def Lambda(fn):
        return _LambdaLayer(fn)


class _KerasModule:
    Input = _InputLayer
    Model = _DummyModel
    layers = _LayersModule()


class _SavedModelModule:
    @staticmethod
    def save(model: Any, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "saved_model.pb").write_text(json.dumps({"model": "dummy"}), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> Any:
        return {"path": str(path)}


class _TensorFlowCompat:
    saved_model = _SavedModelModule()
    keras = _KerasModule()
    string = "string"

    @staticmethod
    def zeros(shape: Any) -> np.ndarray:
        return np.zeros(shape, dtype=np.float32)

    @staticmethod
    def shape(value: Any) -> np.ndarray:
        return np.array([1])


sys_module = _TensorFlowCompat()

__all__ = ["saved_model", "keras", "string", "zeros", "shape"]

saved_model = sys_module.saved_model
keras = sys_module.keras
string = sys_module.string
zeros = sys_module.zeros
shape = sys_module.shape
