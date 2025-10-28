"""
Serializers for different object types.
"""

import pickle
from .registry import Serializer


class TorchSerializer(Serializer):
    """Serializer for PyTorch tensors."""

    def save(self, obj, filepath: str):
        try:
            import torch
            torch.save(obj, filepath)
        except ImportError:
            raise ImportError("PyTorch is not installed. Cannot serialize torch.Tensor")

    def load(self, filepath: str):
        try:
            import torch
            return torch.load(filepath)
        except ImportError:
            raise ImportError("PyTorch is not installed. Cannot load torch.Tensor")

    def get_extension(self) -> str:
        return ".pt"


class NumpySerializer(Serializer):
    """Serializer for NumPy arrays."""

    def save(self, obj, filepath: str):
        try:
            import numpy as np
            np.save(filepath, obj)
        except ImportError:
            raise ImportError("NumPy is not installed. Cannot serialize numpy.ndarray")

    def load(self, filepath: str):
        try:
            import numpy as np
            return np.load(filepath)
        except ImportError:
            raise ImportError("NumPy is not installed. Cannot load numpy.ndarray")

    def get_extension(self) -> str:
        return ".npy"


class PickleSerializer(Serializer):
    """Default serializer using pickle."""

    def save(self, obj, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def get_extension(self) -> str:
        return ".pkl"
