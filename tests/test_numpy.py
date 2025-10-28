"""
Test script for tensor_dumper with NumPy arrays.
"""

import sys
from pathlib import Path

# Add parent directory to path to import tensor_dumper
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
except ImportError:
    print("NumPy is not installed. Please install it with: pip install numpy")
    sys.exit(1)

import tensor_dumper


def test_numpy_dump():
    """Test dumping NumPy arrays."""
    print("=" * 60)
    print("Testing NumPy array dumping")
    print("=" * 60)

    tensor_dumper.init("test_output/numpy")

    # Test with various NumPy arrays
    print("\n1. Testing with 1D array:")
    arr1d = np.array([1, 2, 3, 4, 5])
    print(f"   Shape: {arr1d.shape}, dtype: {arr1d.dtype}")
    tensor_dumper.dump(arr1d, "numpy_1d")

    print("\n2. Testing with 2D array:")
    arr2d = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"   Shape: {arr2d.shape}, dtype: {arr2d.dtype}")
    tensor_dumper.dump(arr2d, "numpy_2d")

    print("\n3. Testing with 3D array:")
    arr3d = np.random.randn(2, 3, 4)
    print(f"   Shape: {arr3d.shape}, dtype: {arr3d.dtype}")
    tensor_dumper.dump(arr3d, "numpy_3d")

    print("\n4. Testing with float32 array:")
    arr_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    print(f"   Shape: {arr_float32.shape}, dtype: {arr_float32.dtype}")
    tensor_dumper.dump(arr_float32, "numpy_float32")

    print("\n5. Testing with large array:")
    arr_large = np.random.randn(100, 100)
    print(f"   Shape: {arr_large.shape}, dtype: {arr_large.dtype}")
    tensor_dumper.dump(arr_large, "numpy_large")

    print("\n✓ NumPy dump tests completed successfully!")


def test_numpy_compare():
    """Test comparing NumPy arrays."""
    print("\n" + "=" * 60)
    print("Testing NumPy array comparison")
    print("=" * 60)

    print("\n1. Comparing identical arrays:")
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([1, 2, 3, 4, 5])
    result = tensor_dumper.compare(arr1, arr2)
    print(f"   Result: {result}")
    assert result == True, "Identical arrays should be equal"

    print("\n2. Comparing different arrays:")
    arr3 = np.array([1, 2, 3, 4, 6])
    result = tensor_dumper.compare(arr1, arr3)
    print(f"   Result: {result}")
    assert result == False, "Different arrays should not be equal"

    print("\n3. Comparing arrays with small difference (within tolerance):")
    arr4 = np.array([1.0, 2.0, 3.0])
    arr5 = np.array([1.0000001, 2.0000001, 3.0000001])
    result = tensor_dumper.compare(arr4, arr5, atol=1e-5, rtol=1e-5)
    print(f"   Result: {result}")
    assert result == True, "Arrays within tolerance should be equal"

    print("\n4. Comparing arrays with large difference (outside tolerance):")
    arr6 = np.array([1.0, 2.0, 3.0])
    arr7 = np.array([1.1, 2.1, 3.1])
    result = tensor_dumper.compare(arr6, arr7, atol=1e-5, rtol=1e-5)
    print(f"   Result: {result}")
    assert result == False, "Arrays outside tolerance should not be equal"

    print("\n✓ NumPy comparison tests completed successfully!")


def main():
    """Run all NumPy tests."""
    print("\n" + "=" * 60)
    print("TENSOR DUMPER - NUMPY TESTS")
    print("=" * 60)

    try:
        test_numpy_dump()
        test_numpy_compare()

        print("\n" + "=" * 60)
        print("ALL NUMPY TESTS PASSED! ✓")
        print("=" * 60)

        print("\nCheck the 'test_output/numpy/' directory for dumped .npy files.")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
