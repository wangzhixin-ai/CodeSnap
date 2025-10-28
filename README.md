# Tensor Dumper

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive debugging tool for machine learning and deep learning development. Tensor Dumper helps you save, compare, and track tensors, arrays, and other Python objects with complete reproducibility.

## Why Tensor Dumper?

When debugging ML/DL models, you need to save intermediate outputs, compare runs, and reproduce experiments months later. Tensor Dumper automatically captures your data along with everything needed for reproducibility: git commits and uncommitted changes, package versions, runtime environment, and command-line arguments.

## Features

**Simple API with complete reproducibility tracking** - Just `init()`, `dump()`, and `compare()` to save tensors/arrays with automatic metadata capture including git history, package versions, environment variables, and command-line arguments. Supports distributed training, cross-format comparison (`.pt` vs `.npy`), and is fully extensible via registry pattern.

**Key capabilities:**
- Multi-format support: PyTorch tensors (`.pt`), NumPy arrays (`.npy`), Python objects (`.pkl`)
- Distributed training: Single shared folder, rank-aware file naming, metadata saved by rank 0 only
- Smart comparison: Cross-format comparison with automatic type conversion and tolerance-based matching
- Complete reproducibility: Git commits, uncommitted changes, package versions, runtime environment
- Flexible organization: Step-based or auto-increment dump folders, timestamped run directories

## Installation


### From Source
```bash
git clone https://github.com/wangzhixin-ai/tensor-dumper.git
cd tensor-dumper
pip install -e .
```

### Optional Dependencies
```bash
pip install tensor-dumper[numpy]  # For NumPy support
pip install tensor-dumper[torch]  # For PyTorch support
pip install tensor-dumper[all]    # Install all optional dependencies
```

## Quick Start

### Single Process

```python
import tensor_dumper
import torch

# Initialize with base directory
# Creates: experiments/20251028_143041/ (timestamped)
tensor_dumper.init("experiments")

# Dump tensors
output = torch.randn(10, 10)
tensor_dumper.dump(output, "layer1_output")

# Dump multiple objects
tensor_dumper.dump(hidden_state, "hidden")
tensor_dumper.dump(attention_weights, "attention")

# Compare tensors with tolerance
is_close = tensor_dumper.compare(output1, output2, atol=1e-5, rtol=1e-5)
```

### Distributed Training (Multi-GPU)

In distributed training scenarios, `tensor_dumper` automatically:
- Creates **only one** timestamped folder (shared by all ranks)
- Only **rank 0** saves metadata (runtime_info.json, packages.json, git_info.json)
- All ranks can dump data with rank-specific filenames

```python
import torch
import torch.distributed as dist
import tensor_dumper

# Initialize distributed training (your existing code)
dist.init_process_group(backend='nccl')
rank = dist.get_rank()

# Initialize tensor_dumper (all ranks call this)
# Only rank 0 creates folder and saves metadata
tensor_dumper.init("experiments")

# All ranks can dump tensors
# Files will be named: layer1_output_rank0.pt, layer1_output_rank1.pt, etc.
output = model(input)
tensor_dumper.dump(output, "layer1_output")

# Only rank 0 will update metadata
loss = criterion(output, target)
tensor_dumper.dump(loss, "loss")
```


## Output Structure

Each run creates a timestamped directory with organized subfolders for each dump:

```
experiments/
└── 20251028_143041/              # Timestamp: YYYYMMDD_HHMMSS (Shanghai timezone)
    ├── runtime_info.json         # Command, environment variables, working directory
    ├── packages.json             # Python version, all installed packages
    ├── git_info.json             # Git commit, logs, runtime modifications
    ├── project.patch             # Git diff of current project (uncommitted changes)
    ├── step_000100/              # Step-based dump (when step parameter is provided)
    │   ├── loss_rank0.pt
    │   ├── loss_rank1.pt
    │   ├── gradient_rank0.pt
    │   └── gradient_rank1.pt
    ├── step_000200/              # Next step
    │   ├── loss_rank0.pt
    │   └── ...
    ├── dump_0000/                # Auto-increment dump (when step is not provided)
    │   ├── data_rank0.pt
    │   └── data_rank1.pt
    └── dump_0001/                # Next auto-increment dump
        └── ...
```

**Directory naming:**
- `step_{step:06d}/`: When you provide a `step` parameter to `dump()`
- `dump_{counter:04d}/`: Auto-increment when no `step` is provided
- Files include rank suffix in distributed mode: `{name}_rank{N}.{ext}`


## API Reference

### Initialization

```python
tensor_dumper.init(folder_name: str)
```
Initialize the dumper. Creates a timestamped subdirectory and saves environment metadata.

**Parameters:**
- `folder_name` (str): Base directory for dumps

**Creates:**
- `{folder_name}/{timestamp}/` directory
- `runtime_info.json`, `packages.json`, `git_info.json`
- `project.patch` (if there are uncommitted changes)

### Dumping Data

```python
tensor_dumper.dump(obj, name=None, update_metadata=True)
```
Dump an object to disk.

**Parameters:**
- `obj` (Any): Object to dump (tensor, array, or any Python object)
- `name` (str, optional): Name for the file. Auto-generated if None
- `update_metadata` (bool): Whether to check and update metadata (default: True)

**Supported Types:**
- PyTorch tensors → `.pt` files
- NumPy arrays → `.npy` files
- Python objects → `.pkl` files

### Comparing Data

```python
tensor_dumper.compare(a, b, atol=1e-8, rtol=1e-5) -> bool
```
Compare two objects or files with tolerance. **Supports cross-format comparison!**

**Parameters:**
- `a`, `b`: Objects to compare OR file paths (str/Path). Supports:
  - Direct objects (tensors, arrays, etc.)
  - File paths to `.pt`, `.npy`, or `.pkl` files
  - Mixed: object vs file path
  - **Cross-format: `.pt` vs `.npy` files**
- `atol` (float): Absolute tolerance (default: 1e-8)
- `rtol` (float): Relative tolerance (default: 1e-5)

**Comparison formula:** `|a - b| ≤ atol + rtol × |b|`

**Examples:**

```python
# Compare objects directly
tensor_dumper.compare(tensor1, tensor2)

# Compare files (no need to load manually!)
tensor_dumper.compare("experiments/20251028_143041/loss_rank0.pt",
                      "experiments/20251028_143041/loss_rank1.pt")

# Cross-format comparison (NEW!)
# Compare PyTorch tensor file with NumPy array file
tensor_dumper.compare("output.pt", "output.npy")

# Compare NumPy array with PyTorch tensor
import torch
import numpy as np
my_tensor = torch.randn(10, 10)
my_array = np.random.randn(10, 10)
tensor_dumper.compare(my_tensor, my_array)

# Compare file with in-memory object
tensor_dumper.compare("saved_tensor.pt", my_array)  # Works even if formats differ!
```

**How cross-format comparison works:**
- PyTorch tensors are automatically converted to NumPy when comparing with NumPy arrays
- NumPy arrays are automatically converted to PyTorch when comparing with PyTorch tensors
- This allows seamless comparison across different frameworks

### Control Functions

```python
tensor_dumper.enable()         # Enable dumping
tensor_dumper.disable()        # Disable dumping
tensor_dumper.update_metadata() # Manually update metadata
```

