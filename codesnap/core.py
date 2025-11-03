"""
Core functionality for codesnap.
"""

import json
import sys
import os
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # For Python < 3.9, use pytz as fallback
    try:
        import pytz
        ZoneInfo = None
    except ImportError:
        ZoneInfo = None
        pytz = None

from .registry import SerializerRegistry
from .metadata import collect_metadata


def _get_rank_info():
    """
    Get distributed training rank information.

    Returns:
        tuple: (rank, world_size, is_distributed)
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size(), True
    except (ImportError, RuntimeError):
        pass

    # Try environment variables (common in distributed training)
    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', -1)))
    world_size = int(os.environ.get('WORLD_SIZE', -1))

    if rank >= 0 and world_size > 0:
        return rank, world_size, True

    # Not in distributed mode
    return 0, 1, False


class CodeSnap:
    """Main dumper class that handles initialization and dumping."""

    def __init__(self):
        self.folder: Optional[Path] = None
        self.latest_folder: Optional[Path] = None
        self.enabled = True
        self.registry = SerializerRegistry()
        self.counter = 0
        self.last_metadata = None  # Cache last metadata to detect changes
        self.rank = 0
        self.world_size = 1
        self.is_distributed = False

    def init(self, folder_name: str):
        """Initialize the dumper with output folder.

        In distributed training, only rank 0 will create the folder and save metadata.
        Other ranks will use the same folder path.

        Args:
            folder_name: Base directory for dumps
        """
        # Get distributed training info
        self.rank, self.world_size, self.is_distributed = _get_rank_info()

        # Create base folder (all ranks can do this, it's idempotent)
        base_folder = Path(folder_name)
        base_folder.mkdir(parents=True, exist_ok=True)

        # Generate timestamp (only rank 0 creates it, others will receive it)
        if self.rank == 0:
            if ZoneInfo is not None:
                # Python 3.9+
                shanghai_tz = ZoneInfo('Asia/Shanghai')
                timestamp = datetime.now(shanghai_tz).strftime('%Y%m%d_%H%M%S')
            elif pytz is not None:
                # Python < 3.9 with pytz
                shanghai_tz = pytz.timezone('Asia/Shanghai')
                timestamp = datetime.now(shanghai_tz).strftime('%Y%m%d_%H%M%S')
            else:
                # Fallback: use local time
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            timestamp = None

        # Synchronize timestamp across all ranks
        if self.is_distributed:
            timestamp = self._broadcast_timestamp(timestamp)

        self.folder = base_folder / timestamp

        # Only rank 0 creates the folder and saves metadata
        if self.rank == 0:
            self.folder.mkdir(parents=True, exist_ok=True)
            self.counter = 0

            self._save_runtime_info()

            self._update_metadata()

            print(f"[CodeSnap] Initialized. Output 1 : {self.folder}")
        else:
            self.counter = 0

        if self.is_distributed:
            self._barrier()

    def _broadcast_timestamp(self, timestamp: Optional[str]) -> str:
        """
        Broadcast timestamp from rank 0 to all other ranks.

        Args:
            timestamp: Timestamp string (only valid on rank 0)

        Returns:
            Broadcasted timestamp string
        """
        try:
            import torch
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                device = None
                backend = dist.get_backend()
                if backend == 'nccl':
                    if torch.cuda.is_available():
                        device = torch.device(f'cuda:{torch.cuda.current_device()}')
                else:
                    device = torch.device('cpu')

                if self.rank == 0:
                    ts_bytes = timestamp.encode('utf-8')
                    length = torch.tensor(len(ts_bytes), dtype=torch.long, device=device)
                else:
                    length = torch.tensor(0, dtype=torch.long, device=device)

                dist.broadcast(length, src=0)

                if self.rank == 0:
                    ts_tensor = torch.tensor(list(ts_bytes), dtype=torch.uint8, device=device)
                else:
                    ts_tensor = torch.zeros(length.item(), dtype=torch.uint8, device=device)

                dist.broadcast(ts_tensor, src=0)

                timestamp = bytes(ts_tensor.cpu().tolist()).decode('utf-8')
                return timestamp
        except (ImportError, RuntimeError) as e:
            import warnings
            warnings.warn(f"[Rank {self.rank}] Timestamp broadcast failed: {e}. Using fallback.")


        if timestamp is None:
            import time
            time.sleep(0.1 * self.rank)  # Small delay to avoid race conditions
            if ZoneInfo is not None:
                shanghai_tz = ZoneInfo('Asia/Shanghai')
                timestamp = datetime.now(shanghai_tz).strftime('%Y%m%d_%H%M%S')
            elif pytz is not None:
                shanghai_tz = pytz.timezone('Asia/Shanghai')
                timestamp = datetime.now(shanghai_tz).strftime('%Y%m%d_%H%M%S')
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        return timestamp

    def _barrier(self):
        """Synchronization barrier for distributed training."""
        try:
            import torch
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                # For NCCL backend, specify device_ids to avoid warning
                backend = dist.get_backend()
                if backend == 'nccl' and torch.cuda.is_available():
                    device_ids = [torch.cuda.current_device()]
                    dist.barrier(device_ids=device_ids)
                else:
                    dist.barrier()
        except (ImportError, RuntimeError):
            # If barrier fails, just continue (may cause race conditions)
            import time
            time.sleep(0.5)  # Small delay to reduce race conditions

    def _save_runtime_info(self):
        """Save runtime information (command, environment variables)."""
        if self.folder is None:
            return

        # Get timezone-aware timestamp
        if ZoneInfo is not None:
            shanghai_tz = ZoneInfo('Asia/Shanghai')
            timestamp = datetime.now(shanghai_tz).isoformat()
        elif pytz is not None:
            shanghai_tz = pytz.timezone('Asia/Shanghai')
            timestamp = datetime.now(shanghai_tz).isoformat()
        else:
            timestamp = datetime.now().isoformat()

        runtime_info = {
            'command': {
                'executable': sys.executable,
                'argv': sys.argv,
                'full_command': ' '.join(sys.argv)
            },
            'environment_variables': dict(os.environ),
            'working_directory': str(Path.cwd()),
            'timestamp': timestamp
        }

        runtime_path = self.folder / "runtime_info.json"
        with open(runtime_path, 'w') as f:
            json.dump(runtime_info, f, indent=2)

    def _update_metadata(self, force: bool = False):
        """Update metadata if changed.

        In distributed training, only rank 0 updates metadata.

        Args:
            force: If True, update metadata even if unchanged
        """
        if self.folder is None:
            return

        # Only rank 0 updates metadata
        if self.rank != 0:
            return

        packages_path = self.folder / "packages.json"
        git_info_path = self.folder / "git_info.json"

        # Collect current metadata
        current_metadata = collect_metadata()

        # Split metadata into packages and git_info
        packages_data = {
            'python_version': current_metadata.get('python_version'),
            'platform': current_metadata.get('platform'),
            'all_packages': current_metadata.get('all_packages', {}),
            'local_packages': {}
        }

        # Extract package info (without git info) for local packages
        for pkg_name, pkg_info in current_metadata.get('local_packages', {}).items():
            packages_data['local_packages'][pkg_name] = {
                'name': pkg_info.get('name'),
                'installed': pkg_info.get('installed'),
                'version': pkg_info.get('version'),
                'location': pkg_info.get('location')
            }

        # Prepare git_data and extract diffs to separate files
        git_data = {
            'local_packages_git_info': {}
        }

        # Extract git info from local packages
        for pkg_name, pkg_info in current_metadata.get('local_packages', {}).items():
            git_info = pkg_info.get('git_info')
            if git_info is not None:
                # Extract git_diff to separate file
                git_diff = git_info.get('git_diff')
                if git_diff:
                    diff_path = self.folder / f"{pkg_name}_local.patch"
                    with open(diff_path, 'w') as f:
                        f.write(git_diff)

                    # Store git info without the diff content
                    git_info_copy = git_info.copy()
                    git_info_copy['git_diff'] = f"See {pkg_name}_local.patch"
                    git_data['local_packages_git_info'][pkg_name] = git_info_copy
                else:
                    git_data['local_packages_git_info'][pkg_name] = git_info

        # First time initialization or force update
        if force or self.last_metadata is None:
            packages_changed = True
            git_changed = True
        else:
            # Check if packages data has changed
            last_packages = {
                'python_version': self.last_metadata.get('python_version'),
                'platform': self.last_metadata.get('platform'),
                'all_packages': self.last_metadata.get('all_packages', {}),
                'local_packages': {}
            }
            for pkg_name, pkg_info in self.last_metadata.get('local_packages', {}).items():
                last_packages['local_packages'][pkg_name] = {
                    'name': pkg_info.get('name'),
                    'installed': pkg_info.get('installed'),
                    'version': pkg_info.get('version'),
                    'location': pkg_info.get('location')
                }
            packages_changed = (packages_data != last_packages)

            # Check if git data has changed
            last_git = {
                'local_packages_git_info': {}
            }
            for pkg_name, pkg_info in self.last_metadata.get('local_packages', {}).items():
                git_info = pkg_info.get('git_info')
                if git_info is not None:
                    last_git['local_packages_git_info'][pkg_name] = git_info
            git_changed = (git_data != last_git)

        # Save packages data if changed
        if packages_changed:
            # Sort all_packages alphabetically before saving
            if 'all_packages' in packages_data and isinstance(packages_data['all_packages'], dict):
                packages_data['all_packages'] = dict(sorted(packages_data['all_packages'].items()))

            with open(packages_path, 'w') as f:
                json.dump(packages_data, f, indent=2)

        # Save git data if changed
        if git_changed:
            with open(git_info_path, 'w') as f:
                json.dump(git_data, f, indent=2)

        # Cache current metadata
        self.last_metadata = current_metadata

    def _metadata_equal(self, meta1: dict, meta2: dict) -> bool:
        """Check if two metadata dictionaries are equal (ignoring minor differences)."""
        # Compare critical fields
        if meta1.get('python_version') != meta2.get('python_version'):
            return False
        if meta1.get('platform') != meta2.get('platform'):
            return False
        if meta1.get('all_packages') != meta2.get('all_packages'):
            return False

        # Compare project git info (important!)
        project_git1 = meta1.get('project_git_info')
        project_git2 = meta2.get('project_git_info')

        if project_git1 is None and project_git2 is None:
            pass  # Both None, equal
        elif project_git1 is None or project_git2 is None:
            return False  # One is None, one is not
        else:
            # Compare project git info
            if project_git1.get('commit_id') != project_git2.get('commit_id'):
                return False
            if project_git1.get('git_diff') != project_git2.get('git_diff'):
                return False
            if project_git1.get('branch') != project_git2.get('branch'):
                return False

        # Compare packages git info (important for detecting code changes)
        packages1 = meta1.get('packages', {})
        packages2 = meta2.get('packages', {})

        if set(packages1.keys()) != set(packages2.keys()):
            return False

        for pkg_name in packages1:
            pkg1 = packages1[pkg_name]
            pkg2 = packages2[pkg_name]

            # Check version
            if pkg1.get('version') != pkg2.get('version'):
                return False

            # Check git info (commit, diff, etc.)
            git1 = pkg1.get('git_info')
            git2 = pkg2.get('git_info')

            if git1 is None and git2 is None:
                continue
            if git1 is None or git2 is None:
                return False

            # Check commit ID and diff
            if git1.get('commit_id') != git2.get('commit_id'):
                return False
            if git1.get('git_diff') != git2.get('git_diff'):
                return False

        # Compare runtime modifications
        runtime1 = meta1.get('runtime_modifications', {})
        runtime2 = meta2.get('runtime_modifications', {})
        if runtime1 != runtime2:
            return False

        return True

    def update_metadata(self):
        """Manually update metadata (check for changes and save if different).

        In distributed training, only rank 0 updates metadata.
        """
        if self.rank == 0:
            self._update_metadata()
            print(f"[CodeSnap] Metadata updated.")
        # Other ranks stay silent

    def enable(self):
        """Enable dumping."""
        self.enabled = True

    def disable(self):
        """Disable dumping."""
        self.enabled = False

    def _get_variable_name(self) -> Optional[str]:
        """
        Try to detect the variable name from the calling code.

        This inspects the call stack to find the variable name used
        in the dump() call.

        Returns:
            Variable name if detected, None otherwise
        """
        import inspect
        import re

        try:
            # Get the caller's frame (skip _get_variable_name and dump)
            frame = inspect.currentframe()
            caller_frame = frame.f_back.f_back  # Skip dump() -> _get_variable_name()

            if caller_frame is None:
                return None

            # Get the source code line that called dump()
            filename = caller_frame.f_code.co_filename
            lineno = caller_frame.f_lineno

            # Read the source line
            with open(filename, 'r') as f:
                lines = f.readlines()
                if lineno > 0 and lineno <= len(lines):
                    source_line = lines[lineno - 1].strip()

                    # Try to extract variable name from patterns like:
                    # codesnap.dump(var_name)
                    # dumper.dump(var_name)
                    # self.dump(var_name)
                    # dump(var_name)

                    # Match: .dump(var_name) or dump(var_name)
                    match = re.search(r'\.dump\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[,)]', source_line)
                    if match:
                        return match.group(1)

                    match = re.search(r'^dump\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[,)]', source_line)
                    if match:
                        return match.group(1)
        except:
            # If anything fails, just return None
            pass

        return None

    def dump(self, obj: Any = None, name: Optional[str] = None, step: Optional[int] = None, update_metadata: bool = True):
        """
        Dump an object to disk.

        Each dump is saved in a separate subfolder to organize data by step/iteration.

        In distributed training:
        - All ranks can dump data
        - File names will include rank information (e.g., "loss_rank0.pt")
        - Only rank 0 updates metadata

        Args:
            obj: Object to dump (torch.Tensor, numpy.ndarray, or any Python object)
            name: Optional name for the dump file and subfolder. If not provided, will try to
                  auto-detect the variable name from the calling code.
            step: Optional step/iteration number. If provided, data will be saved in "step_{step:06d}/" subfolder.
                  If not provided, uses the variable name or internal counter.
            update_metadata: Whether to check and update metadata if changed (default: True)

        Examples:
            # Auto-detect variable name
            ts1 = torch.tensor([1, 2, 3])
            dumper.dump(ts1)
            # -> saves to: folder/ts1/ts1_rank0.pt

            # Save loss at step 100
            dumper.dump(loss, name="loss", step=100)
            # -> saves to: folder/step_000100/loss_rank0.pt

            # Save multiple tensors at the same step
            dumper.dump(loss, name="loss", step=100)
            dumper.dump(grad, name="gradient", step=100)
            # -> saves to: folder/step_000100/loss_rank0.pt
            #              folder/step_000100/gradient_rank0.pt
        """
        if not self.enabled:
            return

        if self.folder is None:
            raise RuntimeError("CodeSnap not initialized. Call init() first.")

        if obj is None:
            return

        # Auto-detect variable name if name is not provided
        if name is None:
            name = self._get_variable_name()
            if name is None:
                name = "data"

        # Determine subfolder name
        if step is not None:
            # Use user-provided step number
            subfolder_name = f"step_{step:06d}"
        else:
            # Use variable name as subfolder if detected, otherwise use counter
            if name != "data":
                subfolder_name = name
            else:
                subfolder_name = f"dump_{self.counter:04d}"
                self.counter += 1

        # Create subfolder for this dump
        dump_folder = self.folder / subfolder_name
        dump_folder.mkdir(parents=True, exist_ok=True)

        # Add rank suffix in distributed mode
        if self.is_distributed and self.world_size > 1:
            filename = f"{name}_rank{self.rank}"
        else:
            filename = name

        # Update metadata if there are changes (only rank 0)
        if update_metadata and self.rank == 0:
            self._update_metadata()

        # Get serializer for this object type
        serializer = self.registry.get_serializer(obj)

        # Determine file extension
        ext = serializer.get_extension()
        filepath = dump_folder / f"{filename}{ext}"

        # Serialize and save (silently, without logging)
        serializer.save(obj, filepath)

        # Update symlink in latest_tensor folder
        self._update_symlink(filepath, f"{filename}{ext}")

    def _update_symlink(self, source_path: Path, link_name: str):
        """
        Create or update a symbolic link in the latest_tensor folder.

        Args:
            source_path: Path to the actual file
            link_name: Name of the symlink (including extension)
        """
        # Lazy creation of latest_folder on first dump
        if self.latest_folder is None:
            self.latest_folder = self.folder / "latest_tensor"
            self.latest_folder.mkdir(parents=True, exist_ok=True)

        link_path = self.latest_folder / link_name

        # Remove existing symlink if it exists
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()

        # Create new symlink (relative path for portability)
        try:
            # Use relative path from latest_folder to source_path
            relative_path = os.path.relpath(source_path, self.latest_folder)
            link_path.symlink_to(relative_path)
        except OSError as e:
            # On some systems (like Windows without admin rights), symlinks may fail
            import warnings
            warnings.warn(f"Failed to create symlink: {e}")

    def _load_from_file(self, filepath):
        """
        Load an object from a file based on its extension.

        Uses the registry system to find the appropriate serializer.

        Supports:
        - .pt: PyTorch tensors
        - .npy: NumPy arrays
        - .pkl: Pickle files

        Args:
            filepath: Path to the file (string or Path object)

        Returns:
            Loaded object

        Raises:
            ValueError: If file extension is not supported
            FileNotFoundError: If file doesn't exist
        """
        from pathlib import Path

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Get serializer by file extension using registry
        ext = filepath.suffix
        serializer = self.registry.get_serializer_by_extension(ext)

        # Load using the serializer
        return serializer.load(filepath)

    def compare(self, a: Any, b: Any, atol: float = 1e-8, rtol: float = 1e-5, **kwargs) -> bool:
        """
        Compare two objects or files.

        Supports:
        - Direct object comparison (tensors, arrays, etc.)
        - File path comparison (.pt, .npy, .pkl files)
        - Mixed: object vs file path

        Args:
            a: First object or file path (str/Path)
            b: Second object or file path (str/Path)
            atol: Absolute tolerance
            rtol: Relative tolerance
            **kwargs: Additional comparison parameters

        Returns:
            True if objects are equal within tolerance

        Examples:
            >>> # Compare objects directly
            >>> codesnap.compare(tensor1, tensor2)

            >>> # Compare files
            >>> codesnap.compare("output_rank0.pt", "output_rank1.pt")

            >>> # Compare file with object
            >>> codesnap.compare("saved.pt", my_tensor)
        """
        # Load from file if input is a string or Path
        if isinstance(a, (str, Path)):
            a = self._load_from_file(a)

        if isinstance(b, (str, Path)):
            b = self._load_from_file(b)

        # Get comparator for these object types
        comparator = self.registry.get_comparator(a, b)
        return comparator.compare(a, b, atol=atol, rtol=rtol, **kwargs)


# Global instance
_dumper = CodeSnap()

# Public API
init = _dumper.init
dump = _dumper.dump
compare = _dumper.compare
enable = _dumper.enable
disable = _dumper.disable
update_metadata = _dumper.update_metadata
