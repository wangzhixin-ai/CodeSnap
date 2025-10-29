"""
Test script for codesnap with PyTorch tensors.
"""

import sys
from pathlib import Path

# Add parent directory to path to import codesnap
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
except ImportError:
    print("PyTorch is not installed. Please install it with: pip install torch")
    sys.exit(1)

import codesnap


def test_torch_dump():
    """Test dumping PyTorch tensors."""
    print("=" * 60)
    print("Testing PyTorch tensor dumping")
    print("=" * 60)

    codesnap.init("test_output/torch")

    # Test with various PyTorch tensors
    print("\n1. Testing with 1D tensor:")
    tensor1d = torch.tensor([1, 2, 3, 4, 5])
    print(f"   Shape: {tensor1d.shape}, dtype: {tensor1d.dtype}")
    codesnap.dump(tensor1d, "torch_1d")

    print("\n2. Testing with 2D tensor:")
    tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"   Shape: {tensor2d.shape}, dtype: {tensor2d.dtype}")
    codesnap.dump(tensor2d, "torch_2d")

    print("\n3. Testing with 3D tensor:")
    tensor3d = torch.randn(2, 3, 4)
    print(f"   Shape: {tensor3d.shape}, dtype: {tensor3d.dtype}")
    codesnap.dump(tensor3d, "torch_3d")

    print("\n4. Testing with float32 tensor:")
    tensor_float32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    print(f"   Shape: {tensor_float32.shape}, dtype: {tensor_float32.dtype}")
    codesnap.dump(tensor_float32, "torch_float32")

    print("\n5. Testing with large tensor:")
    tensor_large = torch.randn(100, 100)
    print(f"   Shape: {tensor_large.shape}, dtype: {tensor_large.dtype}")
    codesnap.dump(tensor_large, "torch_large")

    # Test with GPU tensor if available
    if torch.cuda.is_available():
        print("\n6. Testing with GPU tensor:")
        tensor_gpu = torch.randn(10, 10).cuda()
        print(f"   Shape: {tensor_gpu.shape}, dtype: {tensor_gpu.dtype}, device: {tensor_gpu.device}")
        codesnap.dump(tensor_gpu, "torch_gpu")
    else:
        print("\n6. GPU not available, skipping GPU tensor test")

    print("\n✓ PyTorch dump tests completed successfully!")


def test_torch_compare():
    """Test comparing PyTorch tensors."""
    print("\n" + "=" * 60)
    print("Testing PyTorch tensor comparison")
    print("=" * 60)

    print("\n1. Comparing identical tensors:")
    tensor1 = torch.tensor([1, 2, 3, 4, 5])
    tensor2 = torch.tensor([1, 2, 3, 4, 5])
    result = codesnap.compare(tensor1, tensor2)
    print(f"   Result: {result}")
    assert result == True, "Identical tensors should be equal"

    print("\n2. Comparing different tensors:")
    tensor3 = torch.tensor([1, 2, 3, 4, 6])
    result = codesnap.compare(tensor1, tensor3)
    print(f"   Result: {result}")
    assert result == False, "Different tensors should not be equal"

    print("\n3. Comparing tensors with small difference (within tolerance):")
    tensor4 = torch.tensor([1.0, 2.0, 3.0])
    tensor5 = torch.tensor([1.0000001, 2.0000001, 3.0000001])
    result = codesnap.compare(tensor4, tensor5, atol=1e-5, rtol=1e-5)
    print(f"   Result: {result}")
    assert result == True, "Tensors within tolerance should be equal"

    print("\n4. Comparing tensors with large difference (outside tolerance):")
    tensor6 = torch.tensor([1.0, 2.0, 3.0])
    tensor7 = torch.tensor([1.1, 2.1, 3.1])
    result = codesnap.compare(tensor6, tensor7, atol=1e-5, rtol=1e-5)
    print(f"   Result: {result}")
    assert result == False, "Tensors outside tolerance should not be equal"

    print("\n✓ PyTorch comparison tests completed successfully!")


def test_torch_nn_module():
    """Test usage within a neural network module (like in the requirements)."""
    print("\n" + "=" * 60)
    print("Testing PyTorch nn.Module integration")
    print("=" * 60)

    import torch.nn as nn

    class DecodeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            codesnap.dump(x, "DecodeLayer_input")
            y = self.linear(x)
            codesnap.dump(y, "DecodeLayer_output")
            return y

    codesnap.init("test_output/torch_nn")

    # Create and test the module
    layer = DecodeLayer()
    input_tensor = torch.randn(5, 10)

    print("\n1. Running forward pass with tensor dumping:")
    output = layer(input_tensor)
    print(f"   Input shape: {input_tensor.shape}")
    print(f"   Output shape: {output.shape}")

    print("\n✓ PyTorch nn.Module tests completed successfully!")


def main():
    """Run all PyTorch tests."""
    print("\n" + "=" * 60)
    print("CODESNAP - PYTORCH TESTS")
    print("=" * 60)

    try:
        test_torch_dump()
        test_torch_compare()
        test_torch_nn_module()

        print("\n" + "=" * 60)
        print("ALL PYTORCH TESTS PASSED! ✓")
        print("=" * 60)

        print("\nCheck the 'test_output/torch/' and 'test_output/torch_nn/' directories for dumped .pt files.")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
