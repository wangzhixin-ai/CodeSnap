"""
Core functionality for codesnap.
"""

import json
import sys
import os
from pathlib import Path
from typing import Any
from datetime import datetime
from zoneinfo import ZoneInfo

from .registry import SerializerRegistry
from .metadata import collect_metadata


def _get_local_timezone():
    """
    Get the local system timezone from the system configuration.

    Returns:
        ZoneInfo: ZoneInfo timezone object for local time, or None as fallback
    """
    try:
        # Method 1: Try TZ environment variable
        tz_name = os.environ.get('TZ')
        if tz_name:
            try:
                return ZoneInfo(tz_name)
            except Exception:
                pass

        # Method 2: Try reading /etc/timezone (Debian/Ubuntu)
        if os.path.exists('/etc/timezone'):
            with open('/etc/timezone', 'r') as f:
                tz_name = f.read().strip()
                if tz_name:
                    try:
                        return ZoneInfo(tz_name)
                    except Exception:
                        pass

        # Method 3: Try reading /etc/localtime symlink (RHEL/CentOS/Fedora)
        if os.path.islink('/etc/localtime'):
            link_target = os.readlink('/etc/localtime')
            # Extract timezone name from path like /usr/share/zoneinfo/Asia/Shanghai
            if '/zoneinfo/' in link_target:
                tz_name = link_target.split('/zoneinfo/')[-1]
                try:
                    return ZoneInfo(tz_name)
                except Exception:
                    pass

    except Exception:
        pass

    # If all methods fail, return None (will use naive local time)
    return None


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
    """Main dumper class that handles initialization and dumping.

    This class provides a comprehensive debugging tool for ML/DL development with:
    - Automatic serialization of tensors, arrays, and Python objects
    - Complete reproducibility tracking (git info, packages, runtime environment)
    - Support for distributed training (multi-GPU/multi-node)
    - Flexible dump modes (keep all, keep last, keep last N)

    Attributes:
        folder: Output directory for current session (with timestamp)
        enabled: Whether dumping is currently enabled
        registry: SerializerRegistry for handling different object types
        counter: Global counter for auto-generated dump folder names
        last_metadata: Cached metadata to detect changes
        rank: Process rank in distributed training (0 for single-process)
        world_size: Total number of processes in distributed training
        is_distributed: Whether running in distributed training mode
        dump_history: Dict tracking dump folders for each variable
        var_counters: Dict tracking counters for each variable name
    """

    def __init__(self):
        self.folder: Path | None = None
        self.enabled = True
        self.registry = SerializerRegistry()
        self.counter = 0
        self.last_metadata = None  # Cache last metadata to detect changes
        self.rank = 0
        self.world_size = 1
        self.is_distributed = False
        self.dump_history = {}  # Track dump folders for each variable: {var_name: [folder1, folder2, ...]}
        self.var_counters = {}  # Track counter for each variable name

    def init(self, folder_name: str):
        """Initialize the dumper with output folder.

        Creates a timestamped subfolder under the specified base directory.
        In distributed training, only rank 0 creates the folder and saves metadata,
        while all ranks use the same folder path after timestamp synchronization.

        The timestamp uses local timezone (detected from TZ env var, /etc/timezone,
        or /etc/localtime), falling back to naive local time if detection fails.

        Args:
            folder_name: Base directory for dumps. A timestamped subfolder will be
                created inside (e.g., "experiments/20251110_143041/")

        Examples:
            >>> import codesnap
            >>> codesnap.init("experiments")
            [CodeSnap] Initialized. Output: experiments/20251110_143041/
        """
        # Get distributed training info
        self.rank, self.world_size, self.is_distributed = _get_rank_info()

        # Create base folder (all ranks can do this, it's idempotent)
        base_folder = Path(folder_name)
        base_folder.mkdir(parents=True, exist_ok=True)

        # Generate timestamp (only rank 0 creates it, others will receive it)
        if self.rank == 0:
            local_tz = _get_local_timezone()
            if local_tz is not None:
                timestamp = datetime.now(local_tz).strftime('%Y%m%d_%H%M%S')
            else:
                # Fallback: use naive local time
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

    def _broadcast_timestamp(self, timestamp: str | None) -> str:
        """Broadcast timestamp from rank 0 to all other ranks.

        Ensures all ranks use the same timestamp for consistent folder naming
        in distributed training. Uses torch.distributed for synchronization.

        If torch.distributed is not available or fails, falls back to generating
        a timestamp on each rank with a small delay to reduce race conditions.

        Args:
            timestamp: Timestamp string from rank 0, None for other ranks

        Returns:
            str: Synchronized timestamp string that all ranks will use
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
            local_tz = _get_local_timezone()
            if local_tz is not None:
                timestamp = datetime.now(local_tz).strftime('%Y%m%d_%H%M%S')
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        return timestamp

    def _barrier(self):
        """Synchronization barrier for distributed training.

        Ensures all processes reach this point before any proceed. Uses
        torch.distributed.barrier() with proper device handling for NCCL backend.

        Falls back to a small sleep delay if torch.distributed is not available.
        """
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
        """Save runtime information to metadata folder.

        Captures and saves:
        - Command-line arguments (sys.argv)
        - Python executable path
        - Environment variables
        - Working directory
        - Timestamp

        Saved to: {output_folder}/metadata/runtime_info.json
        """
        if self.folder is None:
            return

        # Create metadata folder
        metadata_folder = self.folder / "metadata"
        metadata_folder.mkdir(exist_ok=True)

        # Get timezone-aware timestamp
        local_tz = _get_local_timezone()
        if local_tz is not None:
            timestamp = datetime.now(local_tz).isoformat()
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

        runtime_path = metadata_folder / "runtime_info.json"
        with open(runtime_path, 'w') as f:
            json.dump(runtime_info, f, indent=2)

    def _update_metadata(self, force: bool = False):
        """Update metadata if changed.

        Collects current environment metadata (packages, git info) and saves
        to JSON files only if changes are detected compared to cached metadata.
        This minimizes expensive git operations.

        In distributed training, only rank 0 updates metadata.

        Saves:
        - metadata/packages.json: Python version, installed packages, local packages info
        - metadata/git_info.json: Git information for local packages
        - metadata/uncommitted_changes/{package_name}.patch: Git diffs for local packages

        Args:
            force: If True, update metadata even if unchanged (default: False)
        """
        if self.folder is None:
            return

        # Only rank 0 updates metadata
        if self.rank != 0:
            return

        # Create metadata folder structure
        metadata_folder = self.folder / "metadata"
        metadata_folder.mkdir(exist_ok=True)

        uncommitted_changes_folder = metadata_folder / "uncommitted_changes"
        uncommitted_changes_folder.mkdir(exist_ok=True)

        packages_path = metadata_folder / "packages.json"
        git_info_path = metadata_folder / "git_info.json"

        # Collect current metadata
        current_metadata = collect_metadata()

        # Split metadata into packages and git_info
        packages_data = {
            'python_version': current_metadata.get('python_version'),
            'platform': current_metadata.get('platform'),
            'installed_packages': current_metadata.get('all_packages', {}),
            'local_packages': {}
        }

        # Extract package info (without git info) for local packages
        for pkg_name, pkg_info in current_metadata.get('local_packages', {}).items():
            # Normalize package name to lowercase to avoid duplicate entries
            normalized_pkg_name = pkg_name.lower()
            packages_data['local_packages'][normalized_pkg_name] = {
                'name': pkg_info.get('name'),
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
                # Normalize package name to lowercase to avoid duplicate entries
                normalized_pkg_name = pkg_name.lower()

                # Extract git_diff to separate file
                git_diff = git_info.get('git_diff')
                untracked_diff = git_info.get('untracked_diff')

                git_info_copy = git_info.copy()

                if git_diff:
                    # Save tracked changes to .patch file
                    patch_filename = f"{normalized_pkg_name}.patch"
                    diff_path = uncommitted_changes_folder / patch_filename
                    with open(diff_path, 'w') as f:
                        f.write(git_diff)
                    git_info_copy['git_diff'] = f"See metadata/uncommitted_changes/{patch_filename}"

                if untracked_diff:
                    # Save untracked files to separate .untracked file
                    untracked_filename = f"{normalized_pkg_name}.untracked"
                    untracked_path = uncommitted_changes_folder / untracked_filename
                    with open(untracked_path, 'w') as f:
                        f.write(untracked_diff)
                    git_info_copy['untracked_diff'] = f"See metadata/uncommitted_changes/{untracked_filename}"

                git_data['local_packages_git_info'][normalized_pkg_name] = git_info_copy

        # First time initialization or force update
        if force or self.last_metadata is None:
            packages_changed = True
            git_changed = True
        else:
            # Check if packages data has changed
            last_packages = {
                'python_version': self.last_metadata.get('python_version'),
                'platform': self.last_metadata.get('platform'),
                'installed_packages': self.last_metadata.get('all_packages', {}),
                'local_packages': {}
            }
            for pkg_name, pkg_info in self.last_metadata.get('local_packages', {}).items():
                # Normalize package name to lowercase for consistency
                normalized_pkg_name = pkg_name.lower()
                last_packages['local_packages'][normalized_pkg_name] = {
                    'name': pkg_info.get('name'),
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
                    # Normalize package name to lowercase for consistency
                    normalized_pkg_name = pkg_name.lower()
                    last_git['local_packages_git_info'][normalized_pkg_name] = git_info
            git_changed = (git_data != last_git)

        # Save packages data if changed
        if packages_changed:
            # Sort installed_packages alphabetically before saving
            if 'installed_packages' in packages_data and isinstance(packages_data['installed_packages'], dict):
                packages_data['installed_packages'] = dict(sorted(packages_data['installed_packages'].items()))

            with open(packages_path, 'w') as f:
                json.dump(packages_data, f, indent=2)

        # Save git data if changed
        if git_changed:
            with open(git_info_path, 'w') as f:
                json.dump(git_data, f, indent=2)

        # Cache current metadata
        self.last_metadata = current_metadata

    def _metadata_equal(self, meta1: dict, meta2: dict) -> bool:
        """Check if two metadata dictionaries are equal.

        Compares critical fields to detect meaningful changes:
        - Python version and platform
        - All installed packages
        - Project and package git information (commits, diffs, branches)
        - Runtime modifications

        Minor differences that don't affect reproducibility are ignored.

        Args:
            meta1: First metadata dictionary
            meta2: Second metadata dictionary

        Returns:
            bool: True if metadata are effectively equal, False otherwise
        """
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

        Forces a check of current environment state (packages, git info) and
        saves to JSON files if changes are detected. Useful when you know the
        environment has changed (e.g., after installing packages or committing code).

        In distributed training, only rank 0 updates metadata.

        Examples:
            >>> import codesnap
            >>> # ... install some packages or commit code changes ...
            >>> codesnap.update_metadata()
            [CodeSnap] Metadata updated.
        """
        if self.rank == 0:
            self._update_metadata()
            print(f"[CodeSnap] Metadata updated.")
        # Other ranks stay silent

    def enable(self):
        """Enable dumping.

        After calling this, dump() will save objects to disk.

        Examples:
            >>> import codesnap
            >>> codesnap.disable()  # Temporarily stop dumping
            >>> # ... some code ...
            >>> codesnap.enable()   # Resume dumping
        """
        self.enabled = True

    def disable(self):
        """Disable dumping.

        After calling this, dump() will do nothing (skip saving).
        Useful for temporarily disabling dumps without removing dump() calls.

        Examples:
            >>> import codesnap
            >>> codesnap.disable()
            >>> codesnap.dump(tensor)  # This will not save anything
        """
        self.enabled = False

    def _get_variable_name(self) -> str | None:
        """Try to detect the variable name from the calling code.

        Inspects the call stack to find the variable name used in the dump() call.
        Supports patterns like:
        - codesnap.dump(var_name)
        - dumper.dump(var_name)
        - self.dump(var_name)
        - dump(var_name)

        Returns:
            str | None: Variable name if detected, None if detection fails

        Note:
            This is a best-effort feature. Complex expressions (e.g., dump(obj.attr))
            may not be detected correctly.
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

    def _cleanup_dumps(self, var_name: str, keep_count: int):
        """Remove old dumps for a specific variable, keeping only the last keep_count dumps.

        Used internally by dump() to implement 'last' and 'last_n' modes.
        Removes entire dump folders from the filesystem.

        Args:
            var_name: Variable name or subfolder identifier
            keep_count: Number of dumps to keep for this variable (oldest are removed first)
        """
        import shutil

        if var_name not in self.dump_history:
            return

        history = self.dump_history[var_name]
        while len(history) > keep_count:
            old_folder = history.pop(0)
            if old_folder.exists():
                try:
                    shutil.rmtree(old_folder)
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to remove old dump folder {old_folder}: {e}")

    def dump(self, obj: Any = None, name: str | None = None, step: int | None = None, update_metadata: bool = True, mode: str = 'last', max_keep: int = 1):
        """Dump an object to disk with automatic serialization and organization.

        Each dump is saved in a separate subfolder to organize data by step/iteration.
        Automatically detects object type and uses appropriate serializer (.pt, .npy, .pkl).

        In distributed training:
        - All ranks can dump data independently
        - File names include rank information (e.g., "loss_rank0.pt")
        - Only rank 0 updates metadata to avoid conflicts

        Args:
            obj: Object to dump (torch.Tensor, numpy.ndarray, or any Python object).
                If None, nothing is saved.
            name: Optional name for the dump file and subfolder. If not provided,
                attempts to auto-detect the variable name from the calling code.
            step: Optional step/iteration number. If provided, data is saved in
                "step_{step:06d}/" subfolder. Allows multiple variables per step.
                If not provided, uses variable name or internal counter.
            update_metadata: Whether to check and update metadata if changed (default: True).
                Set to False to skip metadata checks for performance in tight loops.
            mode: Dump mode controlling how many dumps to keep:
                - 'all': Keep all dumps (no cleanup)
                - 'last': Keep only the latest dump (removes all previous)
                - 'last_n': Keep the latest n dumps (controlled by max_keep)
                Default: 'last'
            max_keep: For 'last_n' mode, the number of dumps to keep (default: 1).
                Ignored for 'all' and 'last' modes.

        Raises:
            RuntimeError: If called before init()
            ValueError: If mode is not 'all', 'last', or 'last_n'

        Examples:
            # Auto-detect variable name
            ts1 = torch.tensor([1, 2, 3])
            dumper.dump(ts1)
            # -> saves to: folder/ts1/ts1_rank0.pt

            # Save loss at step 100, keep only the latest
            dumper.dump(loss, name="loss", step=100, mode='last')
            # -> saves to: folder/step_000100/loss_rank0.pt

            # Save gradient, keep the latest 5
            dumper.dump(grad, name="gradient", step=100, mode='last_n', max_keep=5)
            # -> saves to: folder/step_000100/gradient_rank0.pt

            # Save accuracy, keep all dumps
            dumper.dump(accuracy, name="accuracy", step=100, mode='all')
            # -> saves to: folder/step_000100/accuracy_rank0.pt

            # Skip metadata updates for performance in tight loops
            for i in range(1000):
                dumper.dump(intermediate, name="inter", update_metadata=False)
        """
        # Validate mode
        if mode not in ['all', 'last', 'last_n']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'all', 'last', or 'last_n'")

        # Ensure max_keep is at least 1
        max_keep = max(1, max_keep)

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
            # When mode='all' or 'last_n' and no step is provided, use counter to create unique folders
            if mode in ['all', 'last_n']:
                # Initialize counter for this variable if not exists
                if name not in self.var_counters:
                    self.var_counters[name] = 0

                # Use variable name with counter
                if name != "data":
                    subfolder_name = f"{name}_{self.var_counters[name]:04d}"
                else:
                    subfolder_name = f"dump_{self.var_counters[name]:04d}"

                # Increment counter for next dump
                self.var_counters[name] += 1
            else:
                # For 'last' mode, use fixed variable name as subfolder (will be overwritten)
                if name != "data":
                    subfolder_name = name
                else:
                    subfolder_name = f"dump_{self.counter:04d}"
                    self.counter += 1

        # Create subfolder for this dump
        dump_folder = self.folder / subfolder_name

        # Initialize history tracking for this variable if not exists
        if name not in self.dump_history:
            self.dump_history[name] = []

        # Check if this is a new dump point for this variable (different subfolder from last)
        history = self.dump_history[name]
        is_new_dump = (not history or history[-1] != dump_folder)

        if is_new_dump:
            # This is a new dump point for this variable, handle mode logic
            if mode == 'last':
                # Remove all previous dumps for this variable
                self._cleanup_dumps(name, keep_count=0)
            elif mode == 'last_n':
                # Keep last n-1 dumps (will add the new one to make n total)
                self._cleanup_dumps(name, keep_count=max_keep - 1)
            # else: mode == 'all', no cleanup

            # Track this new dump folder for this variable
            self.dump_history[name].append(dump_folder)

        # Create the dump folder (may already exist for same step/name)
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

    def _load_from_file(self, filepath):
        """Load an object from a file based on its extension.

        Uses the registry system to find the appropriate serializer for the file type.
        Automatically detects format from file extension and applies correct deserializer.

        Supported formats:
        - .pt: PyTorch tensors (requires torch)
        - .npy: NumPy arrays (requires numpy)
        - .pkl: Pickle files (any Python object)

        Args:
            filepath: Path to the file (string or Path object)

        Returns:
            Loaded object (type depends on file format)

        Raises:
            ValueError: If file extension is not supported
            FileNotFoundError: If file doesn't exist
            ImportError: If required package (torch/numpy) is not installed
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
        """Compare two objects or files for equality.

        Supports flexible comparison with automatic type conversion:
        - Direct object comparison (tensors, arrays, etc.)
        - File path comparison (.pt, .npy, .pkl files)
        - Mixed comparison: object vs file path
        - Cross-type comparison: PyTorch tensor vs NumPy array

        Uses appropriate comparison methods based on object types:
        - torch.allclose() for tensors
        - numpy.allclose() for arrays
        - Equality (==) for other types

        Args:
            a: First object or file path (str/Path)
            b: Second object or file path (str/Path)
            atol: Absolute tolerance for numerical comparison (default: 1e-8)
            rtol: Relative tolerance for numerical comparison (default: 1e-5)
            **kwargs: Additional comparison parameters passed to the comparator

        Returns:
            bool: True if objects are equal within tolerance, False otherwise

        Examples:
            >>> # Compare objects directly
            >>> codesnap.compare(tensor1, tensor2)
            True

            >>> # Compare files
            >>> codesnap.compare("output_rank0.pt", "output_rank1.pt")
            False

            >>> # Compare file with object
            >>> codesnap.compare("saved.pt", my_tensor)
            True

            >>> # Cross-type comparison
            >>> codesnap.compare(torch_tensor, numpy_array)
            True
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

# Public API - expose methods from the global instance
init = _dumper.init
dump = _dumper.dump
compare = _dumper.compare
enable = _dumper.enable
disable = _dumper.disable
update_metadata = _dumper.update_metadata
