"""
Basic test script for tensor_dumper (no external dependencies required).
"""

import sys
from pathlib import Path

# Add parent directory to path to import tensor_dumper
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensor_dumper


def test_basic():
    """Test basic functionality with Python objects."""
    print("=" * 60)
    print("Testing basic functionality with Python objects")
    print("=" * 60)

    # Initialize
    tensor_dumper.init("test_output/basic")

    # Test with various Python objects
    print("\n1. Testing with integer:")
    tensor_dumper.dump(42, "test_int")

    print("\n2. Testing with string:")
    tensor_dumper.dump("Hello, World!", "test_string")

    print("\n3. Testing with list:")
    tensor_dumper.dump([1, 2, 3, 4, 5], "test_list")

    print("\n4. Testing with dict:")
    tensor_dumper.dump({"a": 1, "b": 2, "c": 3}, "test_dict")

    print("\n5. Testing with nested structure:")
    complex_obj = {
        "numbers": [1, 2, 3],
        "strings": ["a", "b", "c"],
        "nested": {
            "key1": "value1",
            "key2": [10, 20, 30]
        }
    }
    tensor_dumper.dump(complex_obj, "test_complex")

    print("\n6. Testing auto-naming (no name parameter):")
    tensor_dumper.dump([100, 200, 300])
    tensor_dumper.dump([400, 500, 600])

    print("\n✓ Basic tests completed successfully!")


def test_enable_disable():
    """Test enable/disable functionality."""
    print("\n" + "=" * 60)
    print("Testing enable/disable functionality")
    print("=" * 60)

    tensor_dumper.init("test_output/enable_disable")

    print("\n1. Dumping with enabled (default):")
    tensor_dumper.dump([1, 2, 3], "enabled_dump")

    print("\n2. Disabling dumper:")
    tensor_dumper.disable()
    print("Attempting to dump (should not output):")
    tensor_dumper.dump([4, 5, 6], "disabled_dump")

    print("\n3. Re-enabling dumper:")
    tensor_dumper.enable()
    tensor_dumper.dump([7, 8, 9], "re_enabled_dump")

    print("\n✓ Enable/disable tests completed successfully!")


def test_compare():
    """Test comparison functionality."""
    print("\n" + "=" * 60)
    print("Testing comparison functionality")
    print("=" * 60)

    # Compare basic Python objects
    print("\n1. Comparing identical integers:")
    result = tensor_dumper.compare(42, 42)
    print(f"   42 == 42: {result}")
    assert result == True, "Integers should be equal"

    print("\n2. Comparing different integers:")
    result = tensor_dumper.compare(42, 43)
    print(f"   42 == 43: {result}")
    assert result == False, "Different integers should not be equal"

    print("\n3. Comparing identical strings:")
    result = tensor_dumper.compare("hello", "hello")
    print(f"   'hello' == 'hello': {result}")
    assert result == True, "Strings should be equal"

    print("\n4. Comparing identical lists:")
    result = tensor_dumper.compare([1, 2, 3], [1, 2, 3])
    print(f"   [1,2,3] == [1,2,3]: {result}")
    assert result == True, "Lists should be equal"

    print("\n✓ Comparison tests completed successfully!")


def test_metadata():
    """Test metadata collection."""
    print("\n" + "=" * 60)
    print("Testing metadata collection")
    print("=" * 60)

    import json

    tensor_dumper.init("test_output/metadata")

    # Read and display metadata
    metadata_file = Path("test_output/metadata/metadata.json")
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        print("\nCollected metadata:")
        print(f"  Python version: {metadata.get('python_version', 'N/A')[:50]}...")
        print(f"  Platform: {metadata.get('platform', 'N/A')}")
        print(f"  Packages detected: {len(metadata.get('packages', {}))}")

        for pkg_name, pkg_info in metadata.get('packages', {}).items():
            print(f"\n  Package: {pkg_name}")
            print(f"    Version: {pkg_info.get('version', 'N/A')}")
            if pkg_info.get('git_info'):
                print(f"    Git commit: {pkg_info['git_info'].get('commit_id', 'N/A')[:8]}")
                print(f"    Git status: {pkg_info['git_info'].get('status', 'N/A')}")

    print("\n✓ Metadata tests completed successfully!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TENSOR DUMPER - BASIC TESTS")
    print("=" * 60)

    try:
        test_basic()
        test_enable_disable()
        test_compare()
        test_metadata()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)

        print("\nCheck the 'test_output/' directory for dumped files.")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
