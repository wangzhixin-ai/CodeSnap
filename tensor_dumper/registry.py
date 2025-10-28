"""
Registry system for serializers and comparators.
"""

from typing import Any, Dict
from abc import ABC, abstractmethod


class Serializer(ABC):
    """Base class for serializers."""

    @abstractmethod
    def save(self, obj: Any, filepath: str):
        """Save object to file."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> Any:
        """Load object from file."""
        pass

    @abstractmethod
    def get_extension(self) -> str:
        """Get file extension for this serializer."""
        pass


class Comparator(ABC):
    """Base class for comparators."""

    @abstractmethod
    def compare(self, a: Any, b: Any, **kwargs) -> bool:
        """Compare two objects."""
        pass


class SerializerRegistry:
    """Registry for serializers and comparators."""

    def __init__(self):
        self._serializers: Dict[str, Serializer] = {}
        self._comparators: Dict[str, Comparator] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default serializers and comparators."""
        from .serializers import (
            TorchSerializer,
            NumpySerializer,
            PickleSerializer
        )
        from .comparators import (
            TorchComparator,
            NumpyComparator,
            DefaultComparator
        )

        # Register serializers
        self.register_serializer('torch.Tensor', TorchSerializer())
        self.register_serializer('numpy.ndarray', NumpySerializer())
        self.register_serializer('default', PickleSerializer())

        # Register comparators
        self.register_comparator('torch.Tensor', TorchComparator())
        self.register_comparator('numpy.ndarray', NumpyComparator())
        self.register_comparator('default', DefaultComparator())

    def register_serializer(self, type_name: str, serializer: Serializer):
        """Register a serializer for a type."""
        self._serializers[type_name] = serializer

    def register_comparator(self, type_name: str, comparator: Comparator):
        """Register a comparator for a type."""
        self._comparators[type_name] = comparator

    def get_serializer(self, obj: Any) -> Serializer:
        """Get serializer for an object."""
        type_name = f"{type(obj).__module__}.{type(obj).__name__}"

        # Try exact match
        if type_name in self._serializers:
            return self._serializers[type_name]

        # Try without module name
        simple_name = type(obj).__name__
        for key in self._serializers:
            if key.endswith(simple_name):
                return self._serializers[key]

        # Return default
        return self._serializers['default']

    def get_comparator(self, a: Any, b: Any) -> Comparator:
        """Get comparator for two objects."""
        type_name = f"{type(a).__module__}.{type(a).__name__}"

        # Try exact match
        if type_name in self._comparators:
            return self._comparators[type_name]

        # Try without module name
        simple_name = type(a).__name__
        for key in self._comparators:
            if key.endswith(simple_name):
                return self._comparators[key]

        # Return default
        return self._comparators['default']
