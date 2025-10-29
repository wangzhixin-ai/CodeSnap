"""
Metadata collection for packages.
"""

import sys
import subprocess
import inspect
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
import importlib.metadata


def get_package_version(package_name: str) -> Optional[str]:
    """Get version of an installed package."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_git_info(package_path: Path, max_log_entries: int = 50) -> Optional[Dict[str, Any]]:
    """Get git information for a local package.

    Args:
        package_path: Path to the package directory
        max_log_entries: Maximum number of git log entries to save (default: 50)
    """
    try:
        # Check if it's a git repo
        git_dir = package_path / '.git'
        if not git_dir.exists():
            # Try parent directories
            current = package_path
            for _ in range(5):  # Check up to 5 levels up
                current = current.parent
                if (current / '.git').exists():
                    package_path = current
                    break
            else:
                return None

        # Get current commit id
        commit_result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=package_path,
            capture_output=True,
            text=True,
            timeout=5
        )

        if commit_result.returncode != 0:
            return None

        commit_id = commit_result.stdout.strip()

        # Get git status
        status_result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=package_path,
            capture_output=True,
            text=True,
            timeout=5
        )

        is_dirty = bool(status_result.stdout.strip())

        # Get git log (last N commits)
        log_result = subprocess.run(
            ['git', 'log', f'-{max_log_entries}', '--pretty=format:%H|%an|%ae|%ad|%s', '--date=iso'],
            cwd=package_path,
            capture_output=True,
            text=True,
            timeout=10
        )

        git_log = []
        if log_result.returncode == 0 and log_result.stdout.strip():
            for line in log_result.stdout.strip().split('\n'):
                parts = line.split('|', 4)
                if len(parts) == 5:
                    git_log.append({
                        'commit': parts[0],
                        'author': parts[1],
                        'email': parts[2],
                        'date': parts[3],
                        'message': parts[4]
                    })

        # Get git diff (current changes)
        diff_result = subprocess.run(
            ['git', 'diff', 'HEAD'],
            cwd=package_path,
            capture_output=True,
            text=True,
            timeout=10
        )

        git_diff = diff_result.stdout if diff_result.returncode == 0 else ''

        # Get current branch
        branch_result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=package_path,
            capture_output=True,
            text=True,
            timeout=5
        )

        branch = branch_result.stdout.strip() if branch_result.returncode == 0 else 'unknown'

        return {
            'commit_id': commit_id,
            'branch': branch,
            'is_dirty': is_dirty,
            'status': 'dirty' if is_dirty else 'clean',
            'git_log': git_log,
            'git_diff': git_diff,
            'log_entries_count': len(git_log)
        }

    except Exception as e:
        return {'error': str(e)}


def detect_runtime_modifications(package_name: str) -> Optional[Dict[str, Any]]:
    """Detect if a package has been modified at runtime (monkey patching, etc.).

    Args:
        package_name: Name of the package to check

    Returns:
        Dictionary with runtime modification info, or None if package not loaded
    """
    try:
        module = __import__(package_name)
        modifications = {
            'package': package_name,
            'modified_files': [],
            'total_files_checked': 0
        }

        # Get all modules under this package
        package_modules = []
        for name, mod in sys.modules.items():
            if name.startswith(package_name):
                package_modules.append((name, mod))

        for mod_name, mod in package_modules:
            try:
                # Skip modules without __file__ (built-in modules)
                if not hasattr(mod, '__file__') or mod.__file__ is None:
                    continue

                file_path = Path(mod.__file__)
                if not file_path.exists() or file_path.suffix not in ['.py']:
                    continue

                modifications['total_files_checked'] += 1

                # Read file from disk
                with open(file_path, 'r', encoding='utf-8') as f:
                    disk_source = f.read()

                disk_hash = hashlib.md5(disk_source.encode()).hexdigest()

                # Try to get source from memory
                try:
                    memory_source = inspect.getsource(mod)
                    memory_hash = hashlib.md5(memory_source.encode()).hexdigest()

                    # Compare hashes
                    if disk_hash != memory_hash:
                        modifications['modified_files'].append({
                            'module': mod_name,
                            'file': str(file_path),
                            'disk_hash': disk_hash,
                            'memory_hash': memory_hash,
                            'status': 'modified_at_runtime'
                        })
                except (OSError, TypeError):
                    # Can't get source from memory, skip
                    pass

            except Exception:
                # Skip files that can't be processed
                continue

        modifications['has_runtime_modifications'] = len(modifications['modified_files']) > 0
        return modifications

    except ImportError:
        return None


def get_package_info(package_name: str) -> Dict[str, Any]:
    """Get complete information about a package."""
    info = {
        'name': package_name,
        'installed': False,
        'version': None,
        'location': None,
        'git_info': None
    }

    try:
        # Try to import the package
        module = __import__(package_name)
        info['installed'] = True

        # Get version
        version = get_package_version(package_name)
        info['version'] = version

        # Get location
        if hasattr(module, '__file__') and module.__file__:
            location = Path(module.__file__).parent
            info['location'] = str(location)

            # If no version (local package), try to get git info
            if version is None:
                git_info = get_git_info(location)
                info['git_info'] = git_info

    except ImportError:
        pass

    return info


def collect_metadata() -> Dict[str, Any]:
    """Collect metadata about the environment and all installed packages."""
    metadata = {
        'python_version': sys.version,
        'platform': sys.platform,
        'all_packages': {},
        'packages': {},
        'runtime_modifications': {},
        'project_git_info': None  # Current working directory git info
    }

    # Get current working directory git info
    try:
        cwd = Path.cwd()
        project_git = get_git_info(cwd)
        if project_git and 'error' not in project_git:
            metadata['project_git_info'] = project_git
    except Exception:
        pass

    # Get all installed packages with their versions
    try:
        for dist in importlib.metadata.distributions():
            metadata['all_packages'][dist.name] = dist.version
    except Exception as e:
        # Fallback: record the error but continue
        metadata['all_packages'] = {'error': str(e)}

    # Get detailed info for important ML/DL packages
    packages_to_check = [
        'torch',
        'numpy',
        'tensorflow',
        'jax',
        'pandas',
        'scipy'
    ]

    for package_name in packages_to_check:
        package_info = get_package_info(package_name)
        if package_info['installed']:
            metadata['packages'][package_name] = package_info

            # Check for runtime modifications
            runtime_mods = detect_runtime_modifications(package_name)
            if runtime_mods and runtime_mods.get('has_runtime_modifications'):
                metadata['runtime_modifications'][package_name] = runtime_mods

    return metadata
