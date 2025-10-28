"""
Core functionality for tensor_dumper.
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


class TensorDumper:
    """Main dumper class that handles initialization and dumping."""

    def __init__(self):
        self.folder: Optional[Path] = None
        self.enabled = True
        self.registry = SerializerRegistry()
        self.counter = 0
        self.last_metadata = None  # Cache last metadata to detect changes

    def init(self, folder_name: str):
        """Initialize the dumper with output folder."""
        # Create base folder
        base_folder = Path(folder_name)
        base_folder.mkdir(parents=True, exist_ok=True)

        # Create timestamped subfolder (Shanghai timezone)
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

        self.folder = base_folder / timestamp
        self.folder.mkdir(parents=True, exist_ok=True)

        self.counter = 0

        # Save runtime information
        self._save_runtime_info()

        # Collect and save initial metadata (split into separate files)
        self._update_metadata()

        print(f"TensorDumper initialized. Output folder: {self.folder}")

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

        Args:
            force: If True, update metadata even if unchanged
        """
        if self.folder is None:
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
            'packages': {}
        }

        # Extract package info (without git info)
        for pkg_name, pkg_info in current_metadata.get('packages', {}).items():
            packages_data['packages'][pkg_name] = {
                'name': pkg_info.get('name'),
                'installed': pkg_info.get('installed'),
                'version': pkg_info.get('version'),
                'location': pkg_info.get('location')
            }

        # Prepare git_data and extract diffs to separate files
        git_data = {
            'project_git_info': None,
            'packages_git_info': {},
            'runtime_modifications': current_metadata.get('runtime_modifications', {})
        }

        # Process project git info
        project_git = current_metadata.get('project_git_info')
        if project_git:
            # Extract git_diff to separate file
            git_diff = project_git.get('git_diff')
            if git_diff:
                diff_path = self.folder / "project.patch"
                with open(diff_path, 'w') as f:
                    f.write(git_diff)

                # Store git info without the diff content
                project_git_copy = project_git.copy()
                project_git_copy['git_diff'] = f"See project.patch"
                git_data['project_git_info'] = project_git_copy
            else:
                git_data['project_git_info'] = project_git

        # Extract git info from packages
        for pkg_name, pkg_info in current_metadata.get('packages', {}).items():
            git_info = pkg_info.get('git_info')
            if git_info is not None:
                # Extract git_diff to separate file
                git_diff = git_info.get('git_diff')
                if git_diff:
                    diff_path = self.folder / f"{pkg_name}.patch"
                    with open(diff_path, 'w') as f:
                        f.write(git_diff)

                    # Store git info without the diff content
                    git_info_copy = git_info.copy()
                    git_info_copy['git_diff'] = f"See {pkg_name}.patch"
                    git_data['packages_git_info'][pkg_name] = git_info_copy
                else:
                    git_data['packages_git_info'][pkg_name] = git_info

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
                'packages': {}
            }
            for pkg_name, pkg_info in self.last_metadata.get('packages', {}).items():
                last_packages['packages'][pkg_name] = {
                    'name': pkg_info.get('name'),
                    'installed': pkg_info.get('installed'),
                    'version': pkg_info.get('version'),
                    'location': pkg_info.get('location')
                }
            packages_changed = (packages_data != last_packages)

            # Check if git data has changed
            last_git = {
                'project_git_info': self.last_metadata.get('project_git_info'),
                'packages_git_info': {},
                'runtime_modifications': self.last_metadata.get('runtime_modifications', {})
            }
            for pkg_name, pkg_info in self.last_metadata.get('packages', {}).items():
                git_info = pkg_info.get('git_info')
                if git_info is not None:
                    last_git['packages_git_info'][pkg_name] = git_info
            git_changed = (git_data != last_git)

        # Save packages data if changed
        if packages_changed:
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
        """Manually update metadata (check for changes and save if different)."""
        self._update_metadata()
        print(f"Metadata updated:")
        print(f"  - {self.folder / 'packages.json'}")
        print(f"  - {self.folder / 'git_info.json'}")

    def enable(self):
        """Enable dumping."""
        self.enabled = True

    def disable(self):
        """Disable dumping."""
        self.enabled = False

    def dump(self, obj: Any = None, name: Optional[str] = None, update_metadata: bool = True):
        """
        Dump an object to disk.

        Args:
            obj: Object to dump (torch.Tensor, numpy.ndarray, or any Python object)
            name: Optional name for the dump file
            update_metadata: Whether to check and update metadata if changed (default: True)
        """
        if not self.enabled:
            return

        if self.folder is None:
            raise RuntimeError("TensorDumper not initialized. Call init() first.")

        if obj is None:
            return

        # Generate filename
        if name is None:
            name = f"dump_{self.counter:04d}"
        self.counter += 1

        # Update metadata if there are changes
        if update_metadata:
            self._update_metadata()

        # Get serializer for this object type
        serializer = self.registry.get_serializer(obj)

        # Determine file extension
        ext = serializer.get_extension()
        filepath = self.folder / f"{name}{ext}"

        # Serialize and save
        serializer.save(obj, filepath)
        print(f"Dumped {type(obj).__name__} to {filepath}")

    def compare(self, a: Any, b: Any, atol: float = 1e-8, rtol: float = 1e-5, **kwargs) -> bool:
        """
        Compare two objects.

        Args:
            a: First object
            b: Second object
            atol: Absolute tolerance
            rtol: Relative tolerance
            **kwargs: Additional comparison parameters

        Returns:
            True if objects are equal within tolerance
        """
        # Get comparator for these object types
        comparator = self.registry.get_comparator(a, b)
        return comparator.compare(a, b, atol=atol, rtol=rtol, **kwargs)


# Global instance
_dumper = TensorDumper()

# Public API
init = _dumper.init
dump = _dumper.dump
compare = _dumper.compare
enable = _dumper.enable
disable = _dumper.disable
update_metadata = _dumper.update_metadata
