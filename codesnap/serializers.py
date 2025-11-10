"""Serializers for different object types.

This module provides concrete implementations of the Serializer interface
for common ML/DL data types:
- TorchSerializer: PyTorch tensors (.pt files)
- NumpySerializer: NumPy arrays (.npy files)
- PickleSerializer: Any Python object (.pkl files, fallback)

Each serializer handles saving and loading objects in its specific format,
with lazy imports to avoid requiring all dependencies.
"""

import pickle
from .registry import Serializer


class TorchSerializer(Serializer):
    """Serializer for PyTorch tensors using torch.save/torch.load.

    Saves tensors to .pt files, preserving device information and metadata.
    Uses lazy import to avoid requiring PyTorch if not needed.

    File format: PyTorch's native format (.pt)
    Requires: PyTorch (torch)

    Examples:
        >>> serializer = TorchSerializer()
        >>> serializer.save(torch.tensor([1, 2, 3]), "data.pt")
        >>> tensor = serializer.load("data.pt")
    """

    def save(self, obj, filepath: str):
        """Save PyTorch tensor to file.

        Args:
            obj: PyTorch tensor to save
            filepath: Destination file path

        Raises:
            ImportError: If PyTorch is not installed
        """
        try:
            import torch
            torch.save(obj, filepath)
        except ImportError:
            raise ImportError("PyTorch is not installed. Cannot serialize torch.Tensor")

    def load(self, filepath: str):
        """Load PyTorch tensor from file.

        Args:
            filepath: Source file path

        Returns:
            PyTorch tensor

        Raises:
            ImportError: If PyTorch is not installed
        """
        try:
            import torch
            return torch.load(filepath)
        except ImportError:
            raise ImportError("PyTorch is not installed. Cannot load torch.Tensor")

    def get_extension(self) -> str:
        """Get file extension for PyTorch files.

        Returns:
            str: ".pt"
        """
        return ".pt"


class NumpySerializer(Serializer):
    """Serializer for NumPy arrays using numpy.save/numpy.load.

    Saves arrays to .npy files in NumPy's binary format.
    Uses lazy import to avoid requiring NumPy if not needed.

    File format: NumPy's binary format (.npy)
    Requires: NumPy (numpy)

    Examples:
        >>> serializer = NumpySerializer()
        >>> serializer.save(np.array([1, 2, 3]), "data.npy")
        >>> array = serializer.load("data.npy")
    """

    def save(self, obj, filepath: str):
        """Save NumPy array to file.

        Args:
            obj: NumPy array to save
            filepath: Destination file path

        Raises:
            ImportError: If NumPy is not installed
        """
        try:
            import numpy as np
            np.save(filepath, obj)
        except ImportError:
            raise ImportError("NumPy is not installed. Cannot serialize numpy.ndarray")

    def load(self, filepath: str):
        """Load NumPy array from file.

        Args:
            filepath: Source file path

        Returns:
            NumPy array

        Raises:
            ImportError: If NumPy is not installed
        """
        try:
            import numpy as np
            return np.load(filepath)
        except ImportError:
            raise ImportError("NumPy is not installed. Cannot load numpy.ndarray")

    def get_extension(self) -> str:
        """Get file extension for NumPy files.

        Returns:
            str: ".npy"
        """
        return ".npy"


class PickleSerializer(Serializer):
    """Default serializer using Python's pickle module.

    Saves any Python object to .pkl files using pickle serialization.
    This is the fallback serializer for objects that don't have specialized handlers.

    File format: Python pickle format (.pkl)
    Requires: No additional dependencies (uses standard library)

    Examples:
        >>> serializer = PickleSerializer()
        >>> serializer.save({"key": "value"}, "data.pkl")
        >>> obj = serializer.load("data.pkl")

    Note:
        Pickle files are not secure and can execute arbitrary code.
        Only load pickle files from trusted sources.
    """

    def save(self, obj, filepath: str):
        """Save Python object to file using pickle.

        Args:
            obj: Any Python object to save
            filepath: Destination file path
        """
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)

    def load(self, filepath: str):
        """Load Python object from file using pickle.

        Args:
            filepath: Source file path

        Returns:
            Deserialized Python object

        Warning:
            Only load pickle files from trusted sources.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def get_extension(self) -> str:
        """Get file extension for pickle files.

        Returns:
            str: ".pkl"
        """
        return ".pkl"
