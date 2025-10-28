# Tensor Dumper

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive debugging tool for machine learning and deep learning development. Tensor Dumper helps you save, compare, and track tensors, arrays, and other Python objects with complete reproducibility.

## Why Tensor Dumper?

When debugging ML/DL models, you often need to:
- 💾 Save intermediate outputs for inspection
- 🔍 Compare outputs across different runs
- 🔄 Reproduce experiments months later
- 📊 Track exact code versions and environment configurations

**Tensor Dumper solves all these problems** by automatically capturing:
- Your data (tensors, arrays, objects)
- Runtime information (commands, environment variables)
- Package versions (numpy, torch, etc.)
- Git history and uncommitted changes
- Runtime code modifications

## Features

### Core Features
- 🎯 **Simple API**: Just `init()`, `dump()`, and `compare()`
- 🔧 **Type-agnostic**: Supports PyTorch tensors, NumPy arrays, and any Python object
- 🔄 **Cross-format comparison**: Compare `.pt` vs `.npy` files seamlessly
- 📁 **Auto-organized**: Timestamped directories for each run
- ⚡ **Smart updates**: Only saves metadata when it changes
- 🌐 **Distributed training support**: Works seamlessly with multi-GPU/multi-node training
  - Single shared folder for all ranks
  - Only rank 0 saves metadata
  - Automatic rank detection and synchronization
- 🏗️ **Extensible**: Registry-based architecture for custom types

### Reproducibility Features
- 🚀 **Command tracking**: Saves the exact command used to start your program
- 🌍 **Environment capture**: Records all environment variables
- 📦 **Package tracking**: Logs all installed packages and versions
- 🔀 **Git integration**:
  - Saves commit ID and branch
  - Records last 50 commits
  - Captures all uncommitted changes (git diff)
- 🔍 **Runtime detection**: Detects monkey patching and dynamic code modifications

### Advanced Features
- 🎛️ **Enable/Disable**: Turn dumping on/off without code changes
- 🔄 **Smart comparison**: Tolerance-based comparison for numerical data with cross-format support
- 📊 **Metadata separation**: Different files for packages, git info, and runtime info

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

**Output structure in distributed mode:**
```
experiments/
└── 20251028_143041/              # Single folder (not 8 copies!)
    ├── runtime_info.json         # Only from rank 0
    ├── packages.json             # Only from rank 0
    ├── git_info.json             # Only from rank 0
    ├── project.patch             # Only from rank 0
    ├── layer1_output_rank0.pt    # From rank 0
    ├── layer1_output_rank1.pt    # From rank 1
    ├── layer1_output_rank2.pt    # From rank 2
    └── ...                       # From other ranks
```

## Output Structure

Each run creates a timestamped directory with the following structure:

```
experiments/
└── 20251028_143041/              # Timestamp: YYYYMMDD_HHMMSS (Shanghai timezone)
    ├── runtime_info.json         # Command, environment variables, working directory
    ├── packages.json             # Python version, all installed packages
    ├── git_info.json             # Git commit, logs, runtime modifications
    ├── project.patch             # Git diff of current project (uncommitted changes)
    ├── layer1_output.pt          # Your dumped data
    ├── layer2_output.pt
    └── ...
```

### File Contents

**runtime_info.json**
```json
{
  "command": {
    "executable": "/usr/bin/python3",
    "argv": ["train.py", "--epochs", "100"],
    "full_command": "train.py --epochs 100"
  },
  "environment_variables": {
    "CUDA_VISIBLE_DEVICES": "0,1",
    "PATH": "/usr/local/bin:...",
    ...
  },
  "working_directory": "/path/to/project",
  "timestamp": "2025-10-28T14:30:41+08:00"
}
```

**packages.json**
```json
{
  "python_version": "3.11.5",
  "platform": "linux",
  "all_packages": {
    "torch": "2.0.0",
    "numpy": "1.24.0",
    "transformers": "4.30.0",
    ...
  },
  "packages": {
    "torch": {
      "version": "2.0.0",
      "location": "/usr/local/lib/python3.11/site-packages/torch"
    }
  }
}
```

**git_info.json**
```json
{
  "project_git_info": {
    "commit_id": "abc123...",
    "branch": "main",
    "status": "dirty",
    "git_log": [
      {
        "commit": "abc123...",
        "author": "Your Name",
        "date": "2025-10-28",
        "message": "Add new feature"
      }
    ],
    "git_diff": "See project.patch",
    "log_entries_count": 50
  },
  "packages_git_info": {},
  "runtime_modifications": {}
}
```

**project.patch**
```diff
diff --git a/model.py b/model.py
index 1234567..abcdefg 100644
--- a/model.py
+++ b/model.py
@@ -10,7 +10,7 @@ class Model:
     def forward(self, x):
-        return x * 2
+        return x * 3  # Changed multiplier
```
This file contains the full git diff output showing all uncommitted changes in your project at the time of the run. It can be directly applied using `git apply project.patch`.

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

## Reproducing Experiments

### Scenario: Reproducing a run from 6 months ago

1. **Find the run directory**
   ```bash
   cd experiments/20251028_143041/
   ```

2. **Check the command used**
   ```bash
   cat runtime_info.json | jq '.command.full_command'
   # Output: "train.py --epochs 100 --lr 0.001"
   ```

3. **Restore Python environment**
   ```bash
   # Check Python version
   cat packages.json | jq '.python_version'

   # Install exact package versions
   cat packages.json | jq -r '.all_packages | to_entries[] | "\(.key)==\(.value)"' > requirements.txt
   pip install -r requirements.txt
   ```

4. **Restore code version**
   ```bash
   # Get commit ID
   cat git_info.json | jq -r '.project_git_info.commit_id'

   # Checkout that commit
   git checkout abc123...

   # Apply uncommitted changes from the saved patch file
   git apply project.patch
   ```

5. **Re-run the experiment**
   ```bash
   train.py --epochs 100 --lr 0.001
   ```

## Testing

Run all tests:
```bash
cd tests
python run_tests.py
```

Run specific test suites:
```bash
python -m pytest tests/test_basic.py   # Basic functionality
python -m pytest tests/test_numpy.py   # NumPy integration
python -m pytest tests/test_torch.py   # PyTorch integration
```

## Architecture

### Design Principles

1. **Registry Pattern**: Extensible serializers and comparators
2. **Separation of Concerns**: Different files for different metadata types
3. **Smart Caching**: Only update metadata when it actually changes
4. **Type Safety**: Static typing throughout the codebase

### Components

```
tensor_dumper/
├── core.py           # Main TensorDumper class
├── registry.py       # Serializer and comparator registry
├── serializers.py    # Type-specific serializers
├── comparators.py    # Type-specific comparators
└── metadata.py       # Metadata collection (git, packages, runtime)
```
