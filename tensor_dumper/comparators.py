"""
Comparators for different object types.
"""

from .registry import Comparator


class TorchComparator(Comparator):
    """Comparator for PyTorch tensors."""

    def compare(self, a, b, atol=1e-8, rtol=1e-5, **kwargs) -> bool:
        try:
            import torch
            return torch.allclose(a, b, atol=atol, rtol=rtol)
        except ImportError:
            raise ImportError("PyTorch is not installed. Cannot compare torch.Tensor")


class NumpyComparator(Comparator):
    """Comparator for NumPy arrays."""

    def compare(self, a, b, atol=1e-8, rtol=1e-5, **kwargs) -> bool:
        try:
            import numpy as np
            return np.allclose(a, b, atol=atol, rtol=rtol)
        except ImportError:
            raise ImportError("NumPy is not installed. Cannot compare numpy.ndarray")


class DefaultComparator(Comparator):
    """Default comparator using equality."""

    def compare(self, a, b, **kwargs) -> bool:
        return a == b
