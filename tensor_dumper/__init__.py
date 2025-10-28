"""
Tensor Dumper - A flexible debugging tool for ML/DL development.

This tool allows you to easily dump and compare tensors, numpy arrays,
and other Python objects during debugging.
"""

from .core import (
    init,
    dump,
    compare,
    enable,
    disable,
    update_metadata
)

__version__ = "0.1.0"
__all__ = [
    "init",
    "dump",
    "compare",
    "enable",
    "disable",
    "update_metadata"
]
