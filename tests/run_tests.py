"""
Main test runner - runs all available tests.
"""

import sys
import subprocess
from pathlib import Path


def run_test(test_file, description):
    """Run a single test file."""
    print(f"\n{'=' * 70}")
    print(f"Running: {description}")
    print(f"{'=' * 70}")

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    test_path = script_dir / test_file

    result = subprocess.run(
        [sys.executable, str(test_path)],
        capture_output=False,
        text=True
    )

    return result.returncode == 0


def main():
    """Run all test scripts."""
    print("\n" + "=" * 70)
    print("CODESNAP - COMPLETE TEST SUITE")
    print("=" * 70)

    tests = [
        ("test_basic.py", "Basic functionality tests (no external dependencies)"),
    ]

    # Check for NumPy
    try:
        import numpy
        tests.append(("test_numpy.py", "NumPy integration tests"))
    except ImportError:
        print("\n⚠ NumPy not installed - skipping NumPy tests")
        print("  Install with: pip install numpy")

    # Check for PyTorch
    try:
        import torch
        tests.append(("test_torch.py", "PyTorch integration tests"))
    except ImportError:
        print("\n⚠ PyTorch not installed - skipping PyTorch tests")
        print("  Install with: pip install torch")

    # Run tests
    results = []
    for test_file, description in tests:
        success = run_test(test_file, description)
        results.append((test_file, description, success))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, _, success in results if success)
    total = len(results)

    for test_file, description, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {description}")

    print(f"\n{passed}/{total} test suites passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
