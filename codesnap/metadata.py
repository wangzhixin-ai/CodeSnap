"""Metadata collection for packages and environment tracking.

This module collects comprehensive environment information for reproducibility:
- Python version and platform
- All installed packages with versions
- Local/editable packages with git information
- Git commits, branches, and uncommitted changes
- Package installation locations

Key functions:
- collect_metadata(): Main entry point for collecting all metadata
- get_git_info(): Get git information for a package directory
- get_package_info(): Get comprehensive information about a package
- is_local_package(): Check if a package is installed locally (not from PyPI)
"""

import sys
import subprocess
from pathlib import Path
from typing import Any
import importlib.metadata


def get_package_version(package_name: str) -> str | None:
    """Get version of an installed package using importlib.metadata.

    Args:
        package_name: Name of the package (e.g., "numpy", "torch")

    Returns:
        str | None: Version string if found, None if package not found
    """
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_git_info(package_path: Path, max_log_entries: int = 20) -> dict[str, Any] | None:
    """Get git information for a local package directory.

    Collects comprehensive git state including:
    - Current commit ID and branch
    - Repository status (clean/dirty)
    - Recent commit history with author and date
    - Uncommitted changes (tracked files + untracked files in package directory)

    Searches up to 5 parent directories to find .git folder if not in package_path.

    Args:
        package_path: Path to the package directory to inspect
        max_log_entries: Maximum number of git log entries to collect (default: 20)

    Returns:
        dict | None: Git information dictionary with keys:
            - commit_id: Current commit hash
            - branch: Current branch name
            - is_dirty: Whether there are uncommitted changes (bool)
            - git_log: List of recent commits with author, date, message
            - git_diff: Combined diff of tracked changes and untracked files
            - log_entries_count: Number of log entries collected
        Returns None if not a git repository or on error.
    """
    try:
        # Store the original package path to filter untracked files later
        original_package_path = package_path

        # Check if it's a git repo
        git_dir = package_path / '.git'
        git_root = package_path
        if not git_dir.exists():
            # Try parent directories
            current = package_path
            for _ in range(5):  # Check up to 5 levels up
                current = current.parent
                if (current / '.git').exists():
                    git_root = current
                    break
            else:
                return None
        else:
            git_root = package_path

        commit_result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=git_root,
            capture_output=True,
            text=True,
            timeout=5
        )

        if commit_result.returncode != 0:
            return None

        commit_id = commit_result.stdout.strip()

        # Get git status for the entire repository
        status_result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=git_root,
            capture_output=True,
            text=True,
            timeout=5
        )

        # Check if there are changes within the package directory
        all_status_lines = status_result.stdout.strip().split('\n') if status_result.stdout.strip() else []

        # Filter to only files within the original package path
        package_has_changes = False
        for line in all_status_lines:
            if not line:
                continue
            # Extract filename from status line (format: "XY filename")
            file_path = line[3:].strip()  # Skip status indicators
            full_path = (git_root / file_path).resolve()

            # Check if this file is within the package directory
            try:
                full_path.relative_to(original_package_path)
                package_has_changes = True
                break
            except ValueError:
                # File is outside package directory
                continue

        is_dirty = package_has_changes

        # Get git log (last N commits)
        log_result = subprocess.run(
            ['git', 'log', f'-{max_log_entries}', '--pretty=format:%H|%an|%ae|%ad|%s', '--date=iso'],
            cwd=git_root,
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
        # Capture changes within the package directory:
        # 1. Tracked files (unstaged + staged changes)
        # 2. Untracked files
        # All combined into a single diff

        git_diff_parts = []

        # Helper function to check if a file is within the package directory
        def is_in_package(file_path_str: str) -> bool:
            """Check if a file path is within the original package directory."""
            try:
                full_path = (git_root / file_path_str).resolve()
                full_path.relative_to(original_package_path)
                return True
            except ValueError:
                return False

        # 1. Get unstaged changes (modified tracked files not yet staged)
        unstaged_result = subprocess.run(
            ['git', 'diff'],
            cwd=git_root,
            capture_output=True,
            text=True,
            timeout=10
        )
        if unstaged_result.returncode == 0 and unstaged_result.stdout.strip():
            # Parse diff and only keep changes for files in the package directory
            diff_lines = unstaged_result.stdout.split('\n')
            current_file = None
            file_diff_lines = []

            for line in diff_lines:
                if line.startswith('diff --git '):
                    if current_file and is_in_package(current_file) and file_diff_lines:
                        git_diff_parts.extend(file_diff_lines)

                    parts = line.split(' ')
                    if len(parts) >= 3:
                        current_file = parts[2][2:]  # Remove "a/" prefix
                    file_diff_lines = [line + '\n']
                else:
                    file_diff_lines.append(line + '\n')

            if current_file and is_in_package(current_file) and file_diff_lines:
                git_diff_parts.extend(file_diff_lines)

        # 2. Get staged changes (files in staging area not yet committed)
        staged_result = subprocess.run(
            ['git', 'diff', '--cached'],
            cwd=git_root,
            capture_output=True,
            text=True,
            timeout=10
        )
        if staged_result.returncode == 0 and staged_result.stdout.strip():
            # Parse diff and only keep changes for files in the package directory
            diff_lines = staged_result.stdout.split('\n')
            current_file = None
            file_diff_lines = []

            for line in diff_lines:
                if line.startswith('diff --git '):
                    # Save previous file's diff if it was in package
                    if current_file and is_in_package(current_file) and file_diff_lines:
                        git_diff_parts.extend(file_diff_lines)

                    # Start new file
                    parts = line.split(' ')
                    if len(parts) >= 3:
                        current_file = parts[2][2:]  # Remove "a/" prefix
                    file_diff_lines = [line + '\n']
                else:
                    file_diff_lines.append(line + '\n')

            # Don't forget the last file
            if current_file and is_in_package(current_file) and file_diff_lines:
                git_diff_parts.extend(file_diff_lines)

        # 3. Get untracked files within the package directory and create diff-like output
        untracked_result = subprocess.run(
            ['git', 'ls-files', '--others', '--exclude-standard'],
            cwd=git_root,
            capture_output=True,
            text=True,
            timeout=10
        )

        if untracked_result.returncode == 0 and untracked_result.stdout.strip():
            untracked_files = untracked_result.stdout.strip().split('\n')
            for untracked_file in untracked_files:
                if not untracked_file:
                    continue

                # Only process files within the package directory
                if not is_in_package(untracked_file):
                    continue

                file_path = git_root / untracked_file

                # Skip if file doesn't exist or is a directory
                if not file_path.exists() or file_path.is_dir():
                    continue

                # Skip binary files and very large files (>1MB)
                try:
                    if file_path.stat().st_size > 1_000_000:
                        git_diff_parts.append(f"diff --git a/{untracked_file} b/{untracked_file}\n")
                        git_diff_parts.append(f"new file (skipped: file too large, {file_path.stat().st_size} bytes)\n")
                        continue

                    # Try to read as text
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Create a diff-like format for new files
                    git_diff_parts.append(f"diff --git a/{untracked_file} b/{untracked_file}\n")
                    git_diff_parts.append(f"new file mode 100644\n")
                    git_diff_parts.append(f"index 0000000..0000000\n")
                    git_diff_parts.append(f"--- /dev/null\n")
                    git_diff_parts.append(f"+++ b/{untracked_file}\n")

                    # Add file content with '+' prefix (like git diff does for new files)
                    for line in content.splitlines(keepends=True):
                        git_diff_parts.append(f"+{line}" if line.endswith('\n') else f"+{line}\n")

                except (UnicodeDecodeError, PermissionError, OSError):
                    # Binary file or read error - just note it
                    git_diff_parts.append(f"diff --git a/{untracked_file} b/{untracked_file}\n")
                    git_diff_parts.append(f"new file (binary or unreadable)\n")

        git_diff = ''.join(git_diff_parts)

        # Get current branch
        branch_result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=git_root,
            capture_output=True,
            text=True,
            timeout=5
        )

        branch = branch_result.stdout.strip() if branch_result.returncode == 0 else 'unknown'

        return {
            'commit_id': commit_id,
            'branch': branch,
            'is_dirty': is_dirty,
            'git_log': git_log,
            'git_diff': git_diff,
            'log_entries_count': len(git_log)
        }

    except Exception as e:
        return {'error': str(e)}


def is_local_package(location: str) -> bool:
    """Check if a package is installed locally (not from PyPI).

    A package is considered "local" if it's NOT installed in standard
    site-packages or dist-packages directories, indicating it's either:
    - An editable install (pip install -e)
    - Installed from a local path
    - Added to PYTHONPATH

    Args:
        location: Package installation location path

    Returns:
        bool: True if package is local, False if from PyPI/conda
    """
    if not location:
        return False

    location_path = Path(location)
    location_str = str(location_path)

    # Check if it's in site-packages or dist-packages (standard installation)
    if 'site-packages' in location_str or 'dist-packages' in location_str:
        return False

    return True


def get_editable_install_location(dist) -> str | None:
    """Get the actual source location for an editable install.

    For editable installs (pip install -e), the distribution files are in
    site-packages, but we want the actual source location. This function checks:
    1. direct_url.json for the source path (modern PEP 660 style)
    2. .pth file content (older style editable installs)

    Args:
        dist: Distribution object from importlib.metadata

    Returns:
        str | None: Actual source location if found, None if not editable or not found
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


def get_package_info(package_name: str, location: str | None = None, version: str | None = None) -> dict[str, Any]:
    """Get complete information about a package including git state.

    Collects:
    - Package name, version, and installation location
    - Whether the package can be imported
    - Git information if it's a local package

    Args:
        package_name: Name of the package (e.g., "numpy", "codesnap")
        location: Optional pre-fetched location (to avoid redundant lookups)
        version: Optional pre-fetched version (to avoid redundant lookups)

    Returns:
        dict: Package information with keys:
            - name: Package name
            - installed: Whether package can be imported
            - version: Version string or None
            - location: Installation location or None
            - git_info: Git information dict or None (only for local packages)
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


def collect_metadata() -> dict[str, Any]:
    """Collect comprehensive metadata about the environment and all installed packages.

    This is the main entry point for metadata collection. It gathers:
    - Python version and platform
    - All installed packages from pip/conda (with versions)
    - Local/editable packages (with git information)
    - Packages loaded via PYTHONPATH or sys.path
    - Git state for all local packages (commits, branches, diffs)

    The function scans two sources:
    1. importlib.metadata.distributions() - packages installed via pip/conda
    2. sys.modules - packages loaded from PYTHONPATH or direct imports

    Duplicate packages (same git repo) are automatically deduplicated.

    Returns:
        dict: Metadata dictionary with keys:
            - python_version: Python version string
            - platform: Platform identifier (e.g., "linux", "darwin")
            - all_packages: Dict mapping package names to version strings
            - local_packages: Dict mapping package names to detailed info dicts
                Each local package info contains: name, version, location, git_info
    """
    # Extract Python version and compiler info
    # sys.version format: "3.10.0 (default, Mar  3 2022, 09:58:08) [GCC 7.5.0]"
    version_parts = sys.version.split()
    python_version = version_parts[0] if version_parts else sys.version

    # Extract compiler info (everything inside square brackets)
    compiler_info = None
    if '[' in sys.version and ']' in sys.version:
        start = sys.version.index('[')
        end = sys.version.index(']')
        compiler_info = sys.version[start+1:end]

    metadata = {
        'python_version': python_version,
        'compiler': compiler_info,
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

            # Get module file path
            module_file = Path(module.__file__).resolve()

            # Get the top-level package name (e.g., 'codesnap' from 'codesnap.core')
            top_level_name = module_name.split('.')[0]

            # For __main__ or __mp_main__ (multiprocessing), use a better name
            # These are user scripts, and we want to track their git info
            if top_level_name in ('__main__', '__mp_main__'):
                # Try to get a better name from the git repo root directory
                # First, find the git repo root
                git_repo_root = None
                current = module_file.parent
                for _ in range(10):  # Check up to 10 levels up
                    if (current / '.git').exists():
                        git_repo_root = current
                        break
                    current = current.parent

                # Use the git repo directory name as the package name
                if git_repo_root:
                    top_level_name = git_repo_root.name
                else:
                    # No git repo found, skip this module
                    continue

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


def get_imported_packages() -> dict[str, str | None]:
    """Get information about currently imported packages from sys.modules.

    Scans sys.modules to find all imported third-party packages (not built-in modules)
    and collects their version information. This is useful for tracking which packages
    are actually being used in the current session, not just installed.

    Filters out:
    - Python standard library modules
    - Built-in modules (without __file__)
    - __main__ and __mp_main__ modules
    - Internal/private modules (starting with _)

    Includes packages even if they don't have version info (e.g., not installed via pip).

    Returns:
        dict: Dictionary mapping package names to their versions (same format as all_packages).
            Version is None for packages without version info.

    Note:
        This function is provided for potential future use but is not currently
        called by collect_metadata().
    """
    import sys

    imported = {}
    processed_packages = set()

    # Get Python's standard library location
    stdlib_dir = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"

    for module_name, module in sys.modules.items():
        # Skip built-in modules and modules without __file__
        if not hasattr(module, '__file__') or module.__file__ is None:
            continue

        # Skip __main__ and __mp_main__ modules (scripts being run directly or via multiprocessing)
        if module_name == '__main__' or module_name == '__mp_main__':
            continue

        # Get top-level package name (e.g., 'numpy' from 'numpy.core.numeric')
        top_level_name = module_name.split('.')[0]

        # Skip if already processed
        if top_level_name in processed_packages:
            continue

        # Skip internal/private modules (starting with _)
        if top_level_name.startswith('_'):
            continue

        processed_packages.add(top_level_name)

        # Get package location
        location = None
        try:
            module_file = Path(module.__file__).resolve()
            # For a module file, get its parent (package directory)
            if module_file.name == '__init__.py':
                location = str(module_file.parent)
            else:
                location = str(module_file.parent)

            # For nested modules, find the top-level package directory
            current = Path(location)
            while (current.parent / '__init__.py').exists():
                current = current.parent
            location = str(current)

        except Exception:
            continue

        # Filter out standard library modules
        # Standard library modules are in Python's lib directory but NOT in site-packages
        if location:
            location_path = Path(location)

            # Check if it's in standard library location
            try:
                is_in_stdlib = stdlib_dir in location_path.parents or location_path == stdlib_dir
            except:
                is_in_stdlib = str(stdlib_dir) in location

            # Check if it's in site-packages or dist-packages
            is_in_site_packages = 'site-packages' in location or 'dist-packages' in location

            # Skip if it's in standard library but not in site-packages
            if is_in_stdlib and not is_in_site_packages:
                continue

        # Try to get package version (may be None for non-pip-installed packages)
        version = get_package_version(top_level_name)

        # Store package version (same format as all_packages)
        imported[top_level_name] = version

    return imported
