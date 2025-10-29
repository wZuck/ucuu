"""
Tests for distributed communication functionality.
"""

import pytest
import os

# Check if PyTorch is available
try:
    import torch
    import torch.distributed as dist

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestCPUCommunicationGroup:
    """
    Test cases for CPU communication group functionality.
    """

    def test_import_distributed(self):
        """Test that distributed module can be imported."""
        from ucuu import distributed

        assert distributed is not None

    def test_create_communication_group(self):
        """Test creating a CPU communication group."""
        from ucuu.distributed import CPUCommunicationGroup

        # Create a group without initializing (to avoid requiring actual distributed setup)
        comm_group = CPUCommunicationGroup(
            backend="gloo",
            world_size=2,
            rank=0,
        )

        assert comm_group.backend == "gloo"
        assert comm_group.world_size == 2
        assert comm_group.rank == 0
        assert not comm_group.is_initialized()

    def test_get_rank_and_world_size(self):
        """Test getting rank and world size."""
        from ucuu.distributed import CPUCommunicationGroup

        comm_group = CPUCommunicationGroup(
            backend="gloo",
            world_size=4,
            rank=2,
        )

        assert comm_group.get_rank() == 2
        assert comm_group.get_world_size() == 4

    def test_global_comm_group(self):
        """Test global communication group management."""
        from ucuu.distributed import (
            CPUCommunicationGroup,
            get_global_comm_group,
            set_global_comm_group,
        )

        # Initially should be None
        assert get_global_comm_group() is None

        # Create and set a group
        comm_group = CPUCommunicationGroup(world_size=2, rank=0)
        set_global_comm_group(comm_group)

        # Should now return the group
        assert get_global_comm_group() is comm_group

        # Clean up
        set_global_comm_group(None)

    def test_initialization_with_environment_variables(self):
        """Test initialization using environment variables."""
        from ucuu.distributed import CPUCommunicationGroup

        # Set environment variables
        os.environ["WORLD_SIZE"] = "3"
        os.environ["RANK"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        comm_group = CPUCommunicationGroup()

        assert comm_group.world_size == 3
        assert comm_group.rank == 1
        assert "localhost" in comm_group.init_method
        assert "29500" in comm_group.init_method

        # Clean up
        del os.environ["WORLD_SIZE"]
        del os.environ["RANK"]


@pytest.mark.skipif(TORCH_AVAILABLE, reason="Test for PyTorch not available scenario")
class TestDistributedWithoutPyTorch:
    """
    Test cases for when PyTorch is not available.
    """

    def test_distributed_import_without_pytorch(self):
        """Test that distributed module handles missing PyTorch gracefully."""
        # This should not raise an error, just log a warning
        from ucuu import distributed

        assert distributed is not None

    def test_communication_group_raises_without_pytorch(self):
        """Test that creating a communication group raises ImportError without PyTorch."""
        from ucuu.distributed import CPUCommunicationGroup

        # Should raise ImportError when trying to create the group
        with pytest.raises(ImportError, match="PyTorch is required"):
            CPUCommunicationGroup()
