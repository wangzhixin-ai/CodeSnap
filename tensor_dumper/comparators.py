"""
Comparators for different object types.
"""

from .registry import Comparator


class TorchComparator(Comparator):
    """Comparator for PyTorch tensors."""

    def compare(self, a, b, atol=1e-8, rtol=1e-5, **kwargs) -> bool:
        try:
            import torch
            # Support cross-type comparison: torch vs numpy
            if not isinstance(b, type(a)):
                # Try to convert b to torch tensor
                if hasattr(b, '__array__'):  # NumPy array or similar
                    b = torch.from_numpy(b) if hasattr(b, 'dtype') else torch.tensor(b)

            # Handle cross-device comparison (e.g., cuda:0 vs cuda:1, or cuda vs cpu)
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                if a.device != b.device:
                    # Move both tensors to CPU for comparison
                    a = a.cpu()
                    b = b.cpu()

            return torch.allclose(a, b, atol=atol, rtol=rtol)
        except ImportError:
            raise ImportError("PyTorch is not installed. Cannot compare torch.Tensor")


class NumpyComparator(Comparator):
    """Comparator for NumPy arrays."""

    def compare(self, a, b, atol=1e-8, rtol=1e-5, **kwargs) -> bool:
        try:
            import numpy as np
            # Support cross-type comparison: numpy vs torch
            if not isinstance(b, type(a)):
                # Try to convert b to numpy array
                if hasattr(b, 'numpy'):  # PyTorch tensor
                    b = b.detach().cpu().numpy() if hasattr(b, 'detach') else b.numpy()
                elif hasattr(b, '__array__'):
                    b = np.asarray(b)
            return np.allclose(a, b, atol=atol, rtol=rtol)
        except ImportError:
            raise ImportError("NumPy is not installed. Cannot compare numpy.ndarray")


class DefaultComparator(Comparator):
    """Default comparator using equality."""

    def compare(self, a, b, **kwargs) -> bool:
        return a == b
