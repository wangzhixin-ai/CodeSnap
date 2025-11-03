"""
Metadata collection for packages.
"""

import sys
import subprocess
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


def is_local_package(location: str) -> bool:
    """Check if a package is installed locally (not from PyPI).

    Args:
        location: Package installation location

    Returns:
        True if package is installed locally (editable install or local path)
    """
    if not location:
        return False

    location_path = Path(location)
    location_str = str(location_path)

    # Check if it's in site-packages or dist-packages (standard installation)
    if 'site-packages' in location_str or 'dist-packages' in location_str:
        return False

    return True


def get_editable_install_location(dist) -> Optional[str]:
    """Get the actual source location for an editable install.

    For editable installs, the distribution files are in site-packages,
    but we want the actual source location. This function checks:
    1. direct_url.json for the source path
    2. .pth file content (for older style editable installs)

    Args:
        dist: Distribution object

    Returns:
        Actual source location if found, None otherwise
    """
    try:
        # Method 1: Check direct_url.json (modern editable installs)
        if hasattr(dist, 'read_text'):
            try:
                import json
                direct_url_content = dist.read_text('direct_url.json')
                if direct_url_content:
                    direct_url_data = json.loads(direct_url_content)
                    if direct_url_data.get('dir_info', {}).get('editable'):
                        url = direct_url_data.get('url', '')
                        # Remove file:// prefix
                        # file:///path means file:// + /path (absolute path on Unix)
                        # file://host/path means file:// + host/path (network path)
                        if url.startswith('file://'):
                            path = url[7:]  # Remove 'file://', keep the rest (including leading /)
                        else:
                            path = url

                        if path and Path(path).exists():
                            return path
            except Exception:
                pass

        # Method 2: Check .pth file (older style editable installs)
        if dist.files:
            for file in dist.files:
                if file.name.endswith('.pth') and '__editable__' in file.name:
                    pth_file = dist.locate_file(file)
                    if pth_file and pth_file.exists():
                        # Read the .pth file to get the actual location
                        with open(pth_file, 'r') as f:
                            content = f.read().strip()
                            # .pth file may contain a path directly (not import statements)
                            if content and not content.startswith('import') and Path(content).exists():
                                return content
        return None
    except Exception:
        return None


def get_package_info(package_name: str, location: Optional[str] = None, version: Optional[str] = None) -> Dict[str, Any]:
    """Get complete information about a package.

    Args:
        package_name: Name of the package
        location: Optional pre-fetched location
        version: Optional pre-fetched version
    """
    info = {
        'name': package_name,
        'installed': False,
        'version': version,
        'location': location,
        'git_info': None
    }

    try:
        # Try to import the package
        module = __import__(package_name)
        info['installed'] = True

        # Get version if not provided
        if version is None:
            version = get_package_version(package_name)
            info['version'] = version

        # Get location if not provided
        if location is None and hasattr(module, '__file__') and module.__file__:
            location = str(Path(module.__file__).parent)
            info['location'] = location

        # Check if it's a local package and get git info
        if location and is_local_package(location):
            git_info = get_git_info(Path(location))
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
        'local_packages': {}
    }

    # Get all installed packages with their versions (from pip/conda installs)
    try:
        for dist in importlib.metadata.distributions():
            package_name = dist.name
            version = dist.version
            metadata['all_packages'][package_name] = version

            # Try to get location from distribution
            location = None
            if dist.files:
                # First, check if this is an editable install and get the actual source location
                editable_location = get_editable_install_location(dist)
                if editable_location:
                    location = editable_location
                else:
                    # Get the first file's parent directory
                    first_file = list(dist.files)[0]
                    if dist.locate_file(first_file):
                        location = str(dist.locate_file(first_file).parent)

            # Check if this is a local package
            if location and is_local_package(location) and package_name not in metadata['local_packages']:
                # Collect detailed info including git info for local packages
                package_info = get_package_info(package_name, location=location, version=version)
                if package_info['installed']:
                    metadata['local_packages'][package_name] = package_info

    except Exception as e:
        # Fallback: record the error but continue
        metadata['all_packages'] = {'error': str(e)}

    # Scan sys.modules to find packages loaded from PYTHONPATH or sys.path
    # This catches packages not installed via pip (e.g., direct imports)
    try:
        processed_git_repos = {}  # Map git repo root -> package name to avoid duplicates

        for module_name, module in sys.modules.items():
            # Skip built-in modules and modules without __file__
            if not hasattr(module, '__file__') or module.__file__ is None:
                continue

            # Skip __main__ module (scripts being run directly)
            if module_name == '__main__':
                continue

            # Get module file path
            module_file = Path(module.__file__).resolve()

            # Get the top-level package name (e.g., 'codesnap' from 'codesnap.core')
            top_level_name = module_name.split('.')[0]

            # Skip if already processed
            if top_level_name in metadata['local_packages']:
                continue

            # Get the package root directory
            # For a module like codesnap/core.py, we want the parent directory containing codesnap/
            if module_file.name == '__init__.py':
                # This is a package directory, use its parent
                package_root = module_file.parent
            else:
                # This is a module file, use its parent (the package directory)
                package_root = module_file.parent

            # For nested modules, go up to find the actual package root
            # Keep going up while the parent contains __init__.py
            while (package_root.parent / '__init__.py').exists():
                package_root = package_root.parent

            # Check if this is a local package (not in site-packages)
            package_root_str = str(package_root)
            if not is_local_package(package_root_str):
                continue

            # Get git repo root to avoid duplicate packages from same repo
            git_info = get_git_info(package_root)
            if not git_info or 'error' in git_info:
                continue

            # Find the git repo root
            git_repo_root = package_root
            current = package_root
            for _ in range(10):  # Check up to 10 levels up
                if (current / '.git').exists():
                    git_repo_root = current
                    break
                current = current.parent

            git_repo_root_str = str(git_repo_root)

            # Check if we've already processed this git repo
            if git_repo_root_str in processed_git_repos:
                # Skip this package, we already have one from the same repo
                continue

            # Mark this git repo as processed
            processed_git_repos[git_repo_root_str] = top_level_name

            # Try to get version from importlib.metadata first
            version = get_package_version(top_level_name)

            # Add to local packages
            metadata['local_packages'][top_level_name] = {
                'name': top_level_name,
                'installed': True,
                'version': version or 'unknown',
                'location': package_root_str,
                'git_info': git_info
            }

    except Exception as e:
        # Don't fail the entire metadata collection if sys.modules scan fails
        pass

    return metadata
