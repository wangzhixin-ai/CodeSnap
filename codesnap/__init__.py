"""CodeSnap - A comprehensive debugging tool for ML/DL development.

CodeSnap provides easy-to-use tensor/array dumping with complete reproducibility tracking.
It automatically captures git history, package versions, runtime environment, and uncommitted
changes to ensure your experiments are fully reproducible.

Main Features:
- Automatic serialization of PyTorch tensors, NumPy arrays, and Python objects
- Complete environment tracking (git, packages, runtime)
- Support for distributed training (multi-GPU/multi-node)
- Flexible dump modes (keep all, keep last, keep last N)
- Cross-type comparison (PyTorch vs NumPy)
- No required dependencies (PyTorch and NumPy are optional)

Quick Start:
    >>> import codesnap
    >>> codesnap.init("experiments")
    >>> # ... your code ...
    >>> codesnap.dump(tensor, name="loss", step=100)
    >>> # Compare outputs
    >>> codesnap.compare("saved.pt", my_tensor)

Public API:
    init(folder_name): Initialize CodeSnap with output directory
    dump(obj, name, step): Save an object to disk
    compare(a, b): Compare two objects or files
    enable(): Enable dumping
    disable(): Disable dumping
    update_metadata(): Manually update environment metadata

See documentation at: https://github.com/yourrepo/codesnap
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
