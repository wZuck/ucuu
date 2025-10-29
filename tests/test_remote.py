"""
Tests for remote execution functionality in the decorator.
"""

import pytest
import sys

# Check if PyTorch is available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestRemoteExecution:
    """
    Test cases for remote execution feature of @ucuu decorator.
    """

    def test_remote_decorator_with_pytorch(self):
        """Test that remote decorator can be applied with PyTorch available."""
        from ucuu.decorator import ucuu

        @ucuu("package_utils.print_ucuu_hello", remote=True, peer_rank=1)
        def test_func(x):
            return x * 2

        # Since we don't have actual distributed setup, this should fall back to local execution
        result = test_func(5)
        assert result == 10

    def test_remote_with_tensors(self):
        """Test remote execution with tensor arguments."""
        from ucuu.decorator import ucuu

        @ucuu("package_utils.print_ucuu_hello", remote=True, peer_rank=1)
        def tensor_func(x):
            return x * 2

        tensor_input = torch.tensor([1.0, 2.0, 3.0])
        result = tensor_func(tensor_input)

        # Should return tensor with doubled values
        assert torch.allclose(result, torch.tensor([2.0, 4.0, 6.0]))

    def test_remote_with_custom_preprocess(self):
        """Test remote execution with custom preprocessing."""
        from ucuu.decorator import ucuu

        def custom_preprocess(input_dict):
            """Custom preprocessing that modifies input."""
            if "x" in input_dict:
                input_dict["x"] = input_dict["x"] + 10
            return input_dict

        @ucuu(
            "package_utils.print_ucuu_hello",
            remote=True,
            peer_rank=1,
            custom_preprocess=custom_preprocess,
        )
        def test_func(x):
            return x * 2

        result = test_func(5)
        # Preprocessor adds 10, then multiplies by 2: (5 + 10) * 2 = 30
        assert result == 30

    def test_remote_with_custom_postprocess(self):
        """Test remote execution with custom postprocessing."""
        from ucuu.decorator import ucuu

        def custom_postprocess(output):
            """Custom postprocessing that modifies output."""
            return output + 100

        @ucuu(
            "package_utils.print_ucuu_hello",
            remote=True,
            peer_rank=1,
            custom_postprocess=custom_postprocess,
        )
        def test_func(x):
            return x * 2

        result = test_func(5)
        # Multiplies by 2, then adds 100: (5 * 2) + 100 = 110
        assert result == 110

    def test_remote_with_both_preprocess_and_postprocess(self):
        """Test remote execution with both pre and post processing."""
        from ucuu.decorator import ucuu

        def custom_preprocess(input_dict):
            if "x" in input_dict:
                input_dict["x"] = input_dict["x"] + 1
            return input_dict

        def custom_postprocess(output):
            return output * 10

        @ucuu(
            "package_utils.print_ucuu_hello",
            remote=True,
            peer_rank=1,
            custom_preprocess=custom_preprocess,
            custom_postprocess=custom_postprocess,
        )
        def test_func(x):
            return x * 2

        result = test_func(5)
        # Adds 1, multiplies by 2, then multiplies by 10: ((5 + 1) * 2) * 10 = 120
        assert result == 120

    def test_remote_without_peer_rank(self):
        """Test remote execution without specifying peer_rank (uses current rank)."""
        from ucuu.decorator import ucuu

        @ucuu("package_utils.print_ucuu_hello", remote=True)
        def test_func(x):
            return x * 2

        # Should work without peer_rank specified
        result = test_func(5)
        assert result == 10

    def test_remote_with_optional_peer_rank(self):
        """Test that peer_rank is truly optional."""
        from ucuu.decorator import ucuu

        # Test without peer_rank
        @ucuu("package_utils.print_ucuu_hello", remote=True)
        def test_func_no_rank(x):
            return x * 2

        # Test with peer_rank
        @ucuu("package_utils.print_ucuu_hello", remote=True, peer_rank=1)
        def test_func_with_rank(x):
            return x * 2

        # Both should work
        result1 = test_func_no_rank(5)
        result2 = test_func_with_rank(5)
        assert result1 == 10
        assert result2 == 10


@pytest.mark.skipif(TORCH_AVAILABLE, reason="Test for PyTorch not available scenario")
class TestRemoteExecutionWithoutPyTorch:
    """
    Test cases for remote execution when PyTorch is not available.
    """

    def test_remote_decorator_without_pytorch(self):
        """Test that remote decorator falls back gracefully without PyTorch."""
        from ucuu.decorator import ucuu

        @ucuu("package_utils.print_ucuu_hello", remote=True)
        def test_func(x):
            return x * 2

        # Should fall back to local execution without errors
        result = test_func(5)
        assert result == 10


class TestRemoteExecutionBasic:
    """
    Basic test cases for remote execution that don't require PyTorch.
    """

    def test_remote_decorator_basic(self):
        """Test basic remote decorator functionality."""
        from ucuu.decorator import ucuu

        @ucuu("package_utils.print_ucuu_hello", remote=True)
        def test_func(x):
            return x * 2

        result = test_func(5)
        assert result == 10

    def test_remote_false_normal_execution(self):
        """Test that remote=False uses normal execution path."""
        from ucuu.decorator import ucuu

        @ucuu("package_utils.print_ucuu_hello", remote=False)
        def test_func(x):
            return x * 2

        result = test_func(5)
        assert result == 10

    def test_default_remote_false(self):
        """Test that remote defaults to False."""
        from ucuu.decorator import ucuu

        @ucuu("package_utils.print_ucuu_hello")
        def test_func(x):
            return x * 2

        result = test_func(5)
        assert result == 10
