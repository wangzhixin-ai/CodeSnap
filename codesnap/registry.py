"""Registry system for serializers and comparators.

This module provides an extensible registry pattern for handling different object types:
- Serializer: Abstract base for saving/loading objects to/from disk
- Comparator: Abstract base for comparing objects
- SerializerRegistry: Central registry mapping object types to handlers

The registry enables CodeSnap to support new data types by allowing users to
register custom serializers and comparators.
"""

from typing import Any
from abc import ABC, abstractmethod


class Serializer(ABC):
    """Abstract base class for object serializers.

    Serializers handle saving and loading objects to/from disk in a specific format.
    Each serializer is responsible for one format (e.g., PyTorch .pt files, NumPy .npy files).

    Subclasses must implement:
    - save(): Save object to file
    - load(): Load object from file
    - get_extension(): Return file extension for this format (e.g., ".pt")

    Examples:
        >>> class MySerializer(Serializer):
        ...     def save(self, obj, filepath):
        ...         # Custom save logic
        ...         pass
        ...     def load(self, filepath):
        ...         # Custom load logic
        ...         pass
        ...     def get_extension(self):
        ...         return ".custom"
    """

    @abstractmethod
    def save(self, obj: Any, filepath: str):
        """Save object to file.

        Args:
            obj: Object to save
            filepath: Destination file path
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> Any:
        """Load object from file.

        Args:
            filepath: Source file path

        Returns:
            Loaded object
        """
        pass

    @abstractmethod
    def get_extension(self) -> str:
        """Get file extension for this serializer.

        Returns:
            str: File extension including dot (e.g., ".pt", ".npy")
        """
        pass


class Comparator(ABC):
    """Abstract base class for object comparators.

    Comparators handle comparing two objects for equality, typically with
    numerical tolerance. Each comparator is responsible for one object type.

    Subclasses must implement:
    - compare(): Compare two objects and return True if equal

    Examples:
        >>> class MyComparator(Comparator):
        ...     def compare(self, a, b, **kwargs):
        ...         return a.custom_compare(b)
    """

    @abstractmethod
    def compare(self, a: Any, b: Any, **kwargs) -> bool:
        """Compare two objects.

        Args:
            a: First object
            b: Second object
            **kwargs: Additional parameters (e.g., atol, rtol for numerical comparison)

        Returns:
            bool: True if objects are equal, False otherwise
        """
        pass


class SerializerRegistry:
    """Central registry for managing serializers and comparators.

    Maps object types and file extensions to appropriate serializers/comparators.
    Provides default handlers for common types (PyTorch tensors, NumPy arrays, Python objects).

    Supports:
    - Type-based lookup: Find serializer by object type
    - Extension-based lookup: Find serializer by file extension
    - Custom registration: Register user-defined serializers/comparators

    Built-in serializers:
    - torch.Tensor -> TorchSerializer (.pt files)
    - numpy.ndarray -> NumpySerializer (.npy files)
    - default -> PickleSerializer (.pkl files)

    Built-in comparators:
    - torch.Tensor -> TorchComparator (with cross-type support)
    - numpy.ndarray -> NumpyComparator (with cross-type support)
    - default -> DefaultComparator (equality check)

    Attributes:
        _serializers: Dict mapping type names to Serializer instances
        _comparators: Dict mapping type names to Comparator instances
        _extension_serializers: Dict mapping file extensions to Serializer instances
    """

    def __init__(self):
        """Initialize registry and register default handlers."""
        self._serializers: dict[str, Serializer] = {}
        self._comparators: dict[str, Comparator] = {}
        self._extension_serializers: dict[str, Serializer] = {}  # Extension -> Serializer mapping
        self._register_defaults()

    def _register_defaults(self):
        """Register default serializers and comparators for built-in types.

        Registers handlers for:
        - PyTorch tensors (torch.Tensor)
        - NumPy arrays (numpy.ndarray)
        - Default fallback (pickle for any Python object)

        Also sets up file extension mappings (.pt, .npy, .pkl).
        """
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

        # Create serializer instances
        torch_serializer = TorchSerializer()
        numpy_serializer = NumpySerializer()
        pickle_serializer = PickleSerializer()

        # Register serializers by type
        self.register_serializer('torch.Tensor', torch_serializer)
        self.register_serializer('numpy.ndarray', numpy_serializer)
        self.register_serializer('default', pickle_serializer)

        # Register serializers by extension
        self.register_serializer_for_extension('.pt', torch_serializer)
        self.register_serializer_for_extension('.npy', numpy_serializer)
        self.register_serializer_for_extension('.pkl', pickle_serializer)

        # Register comparators
        self.register_comparator('torch.Tensor', TorchComparator())
        self.register_comparator('numpy.ndarray', NumpyComparator())
        self.register_comparator('default', DefaultComparator())

    def register_serializer(self, type_name: str, serializer: Serializer):
        """Register a serializer for a specific object type.

        Args:
            type_name: Fully qualified type name (e.g., "torch.Tensor", "numpy.ndarray")
                or simple name (e.g., "Tensor", "ndarray"), or "default" for fallback
            serializer: Serializer instance to handle this type

        Examples:
            >>> registry = SerializerRegistry()
            >>> registry.register_serializer("my_module.MyType", MySerializer())
        """
        self._serializers[type_name] = serializer

    def register_serializer_for_extension(self, extension: str, serializer: Serializer):
        """Register a serializer for a file extension.

        Args:
            extension: File extension (e.g., '.pt', '.npy')
            serializer: Serializer instance
        """
        if not extension.startswith('.'):
            extension = '.' + extension
        self._extension_serializers[extension.lower()] = serializer

    def register_comparator(self, type_name: str, comparator: Comparator):
        """Register a comparator for a specific object type.

        Args:
            type_name: Fully qualified type name (e.g., "torch.Tensor")
                or "default" for fallback
            comparator: Comparator instance to handle this type

        Examples:
            >>> registry = SerializerRegistry()
            >>> registry.register_comparator("my_module.MyType", MyComparator())
        """
        self._comparators[type_name] = comparator

    def get_serializer(self, obj: Any) -> Serializer:
        """Get serializer for an object based on its type.

        Tries to match the object type with registered serializers:
        1. Exact match with fully qualified type name
        2. Match with simple type name
        3. Fallback to default serializer

        Args:
            obj: Object to serialize

        Returns:
            Serializer: Appropriate serializer for this object type
        """
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

    def get_serializer_by_extension(self, extension: str) -> Serializer:
        """Get serializer for a file extension.

        Args:
            extension: File extension (e.g., '.pt', '.npy')

        Returns:
            Serializer instance

        Raises:
            ValueError: If extension is not registered
        """
        if not extension.startswith('.'):
            extension = '.' + extension

        ext_lower = extension.lower()
        if ext_lower in self._extension_serializers:
            return self._extension_serializers[ext_lower]

        raise ValueError(
            f"Unsupported file extension: {extension}. "
            f"Supported: {', '.join(self._extension_serializers.keys())}"
        )

    def get_comparator(self, a: Any, b: Any) -> Comparator:
        """Get comparator for two objects based on the first object's type.

        Tries to match the first object's type with registered comparators:
        1. Exact match with fully qualified type name
        2. Match with simple type name
        3. Fallback to default comparator

        Args:
            a: First object (type determines which comparator to use)
            b: Second object (may be converted by the comparator)

        Returns:
            Comparator: Appropriate comparator for these objects
        """
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
