"""Comparators for different object types.

This module provides concrete implementations of the Comparator interface
for common ML/DL data types:
- TorchComparator: PyTorch tensors (with cross-type and cross-device support)
- NumpyComparator: NumPy arrays (with cross-type support)
- DefaultComparator: Any Python object (using equality)

Key features:
- Cross-type comparison (e.g., PyTorch tensor vs NumPy array)
- Cross-device comparison (e.g., CUDA tensor vs CPU tensor)
- Numerical tolerance support (atol, rtol)
"""

from .registry import Comparator


class TorchComparator(Comparator):
    """Comparator for PyTorch tensors using torch.allclose().

    Supports:
    - Standard tensor comparison with numerical tolerance
    - Cross-type comparison (PyTorch tensor vs NumPy array)
    - Cross-device comparison (CUDA vs CPU, or different CUDA devices)

    When comparing different types or devices, automatically converts
    to compatible format for comparison.

    Requires: PyTorch (torch)

    Examples:
        >>> comparator = TorchComparator()
        >>> # Same device
        >>> comparator.compare(tensor1, tensor2, atol=1e-8, rtol=1e-5)
        True
        >>> # Cross-device
        >>> comparator.compare(cuda_tensor, cpu_tensor)
        True
        >>> # Cross-type
        >>> comparator.compare(torch_tensor, numpy_array)
        True
    """

    def compare(self, a, b, atol=1e-8, rtol=1e-5, **kwargs) -> bool:
        """Compare two tensors or tensor-like objects.

        Args:
            a: First tensor (PyTorch tensor)
            b: Second tensor (PyTorch tensor or NumPy array)
            atol: Absolute tolerance (default: 1e-8)
            rtol: Relative tolerance (default: 1e-5)
            **kwargs: Additional arguments (currently unused)

        Returns:
            bool: True if tensors are equal within tolerance

        Raises:
            ImportError: If PyTorch is not installed
        """
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
    """Comparator for NumPy arrays using numpy.allclose().

    Supports:
    - Standard array comparison with numerical tolerance
    - Cross-type comparison (NumPy array vs PyTorch tensor)

    When comparing with PyTorch tensors, automatically converts
    tensor to NumPy array for comparison.

    Requires: NumPy (numpy)

    Examples:
        >>> comparator = NumpyComparator()
        >>> # Same type
        >>> comparator.compare(array1, array2, atol=1e-8, rtol=1e-5)
        True
        >>> # Cross-type
        >>> comparator.compare(numpy_array, torch_tensor)
        True
    """

    def compare(self, a, b, atol=1e-8, rtol=1e-5, **kwargs) -> bool:
        """Compare two arrays or array-like objects.

        Args:
            a: First array (NumPy array)
            b: Second array (NumPy array or PyTorch tensor)
            atol: Absolute tolerance (default: 1e-8)
            rtol: Relative tolerance (default: 1e-5)
            **kwargs: Additional arguments (currently unused)

        Returns:
            bool: True if arrays are equal within tolerance

        Raises:
            ImportError: If NumPy is not installed
        """
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
    """Default comparator using Python's equality operator.

    This is the fallback comparator for objects that don't have specialized handlers.
    Simply uses the == operator for comparison.

    Suitable for:
    - Simple Python objects (int, float, str, etc.)
    - Objects with custom __eq__ methods
    - Non-numerical data

    Examples:
        >>> comparator = DefaultComparator()
        >>> comparator.compare("hello", "hello")
        True
        >>> comparator.compare(42, 42)
        True
        >>> comparator.compare([1, 2, 3], [1, 2, 3])
        True
    """

    def compare(self, a, b, **kwargs) -> bool:
        """Compare two objects using equality.

        Args:
            a: First object
            b: Second object
            **kwargs: Additional arguments (ignored)

        Returns:
            bool: True if a == b, False otherwise
        """
        return a == b
