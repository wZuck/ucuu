"""
Distributed communication utilities for ucuu.

This module provides functionality for creating CPU communication groups
using PyTorch's distributed communication primitives.
"""

import os
from typing import Optional, List
from ucuu.log import setup_logger

logger = setup_logger()

try:
    import torch
    import torch.distributed as dist

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "[yellow]PyTorch not available. Install with 'pip install ucuu[distributed]' to use distributed features.[/yellow]"
    )


class CPUCommunicationGroup:
    """
    CPU communication group for distributed computing across nodes.

    This class manages CPU-based communication groups between different ranks
    on different nodes using PyTorch's distributed primitives.

    Attributes:
        backend: Communication backend (default: 'gloo' for CPU)
        world_size: Total number of processes in the group
        rank: Rank of the current process
        group: PyTorch distributed process group
    """

    def __init__(
        self,
        backend: str = "gloo",
        init_method: Optional[str] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        """
        Initialize a CPU communication group.

        Args:
            backend: Communication backend. Default is 'gloo' for CPU operations.
            init_method: URL specifying how to initialize the process group.
                        If None, reads from environment variable MASTER_ADDR/MASTER_PORT.
            world_size: Total number of processes. If None, reads from environment.
            rank: Rank of this process. If None, reads from environment.

        Raises:
            ImportError: If PyTorch is not installed.
            RuntimeError: If distributed initialization fails.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for distributed features. "
                "Install it with: pip install ucuu[distributed]"
            )

        self.backend = backend
        self.world_size = world_size or int(os.environ.get("WORLD_SIZE", 1))
        self.rank = rank or int(os.environ.get("RANK", 0))

        # Construct init_method from environment if not provided
        if init_method is None:
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = os.environ.get("MASTER_PORT", "29500")
            init_method = f"tcp://{master_addr}:{master_port}"

        self.init_method = init_method
        self.group = None
        self._initialized = False

    def initialize(self) -> None:
        """
        Initialize the process group.

        This method should be called once before any communication operations.
        """
        if self._initialized:
            logger.warning(
                "[yellow]Process group already initialized. Skipping initialization.[/yellow]"
            )
            return

        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                init_method=self.init_method,
                world_size=self.world_size,
                rank=self.rank,
            )

        self.group = dist.new_group(backend=self.backend)
        self._initialized = True

        logger.info(
            f"[bold green]CPU Communication Group Initialized[/bold green]\n"
            f"Backend: [cyan]{self.backend}[/cyan]\n"
            f"World Size: [cyan]{self.world_size}[/cyan]\n"
            f"Rank: [cyan]{self.rank}[/cyan]\n"
            f"Init Method: [cyan]{self.init_method}[/cyan]"
        )

    def create_peer_group(self, ranks: List[int]) -> "dist.ProcessGroup":
        """
        Create a communication group for specific ranks.

        Args:
            ranks: List of ranks to include in the peer group.

        Returns:
            PyTorch process group for the specified ranks.

        Raises:
            RuntimeError: If the process group is not initialized.
        """
        if not self._initialized:
            raise RuntimeError(
                "Process group not initialized. Call initialize() first."
            )

        peer_group = dist.new_group(ranks=ranks, backend=self.backend)

        logger.info(
            f"[bold green]Peer Group Created[/bold green]\n"
            f"Ranks: [cyan]{ranks}[/cyan]\n"
            f"Backend: [cyan]{self.backend}[/cyan]"
        )

        return peer_group

    def send_tensor(
        self,
        tensor: "torch.Tensor",
        dst: int,
        group: Optional["dist.ProcessGroup"] = None,
    ) -> None:
        """
        Send a tensor to a destination rank.

        Args:
            tensor: Tensor to send (will be moved to CPU if not already).
            dst: Destination rank.
            group: Process group to use. If None, uses the default group.
        """
        if not self._initialized:
            raise RuntimeError(
                "Process group not initialized. Call initialize() first."
            )

        # Ensure tensor is on CPU
        cpu_tensor = tensor.cpu() if tensor.device.type != "cpu" else tensor

        dist.send(tensor=cpu_tensor, dst=dst, group=group)

        logger.info(
            f"[bold green]Tensor Sent[/bold green]\n"
            f"Destination: [cyan]{dst}[/cyan]\n"
            f"Shape: [cyan]{cpu_tensor.shape}[/cyan]"
        )

    def recv_tensor(
        self,
        tensor: "torch.Tensor",
        src: int,
        group: Optional["dist.ProcessGroup"] = None,
    ) -> "torch.Tensor":
        """
        Receive a tensor from a source rank.

        Args:
            tensor: Tensor buffer to receive into.
            src: Source rank.
            group: Process group to use. If None, uses the default group.

        Returns:
            Received tensor.
        """
        if not self._initialized:
            raise RuntimeError(
                "Process group not initialized. Call initialize() first."
            )

        # Ensure tensor is on CPU
        cpu_tensor = tensor.cpu() if tensor.device.type != "cpu" else tensor

        dist.recv(tensor=cpu_tensor, src=src, group=group)

        logger.info(
            f"[bold green]Tensor Received[/bold green]\n"
            f"Source: [cyan]{src}[/cyan]\n"
            f"Shape: [cyan]{cpu_tensor.shape}[/cyan]"
        )

        return cpu_tensor

    def barrier(self, group: Optional["dist.ProcessGroup"] = None) -> None:
        """
        Synchronize all processes in the group.

        Args:
            group: Process group to synchronize. If None, uses the default group.
        """
        if not self._initialized:
            raise RuntimeError(
                "Process group not initialized. Call initialize() first."
            )

        dist.barrier(group=group)

    def cleanup(self) -> None:
        """
        Clean up the process group.

        Should be called when the communication group is no longer needed.
        """
        if self._initialized and dist.is_initialized():
            dist.destroy_process_group()
            self._initialized = False

            logger.info("[bold green]Process Group Cleaned Up[/bold green]")

    def is_initialized(self) -> bool:
        """
        Check if the process group is initialized.

        Returns:
            True if initialized, False otherwise.
        """
        return self._initialized

    def get_rank(self) -> int:
        """
        Get the rank of the current process.

        Returns:
            Current process rank.
        """
        return self.rank

    def get_world_size(self) -> int:
        """
        Get the total number of processes.

        Returns:
            World size.
        """
        return self.world_size


# Global communication group instance
_global_comm_group: Optional[CPUCommunicationGroup] = None


def get_global_comm_group() -> Optional[CPUCommunicationGroup]:
    """
    Get the global communication group instance.

    Returns:
        Global CPUCommunicationGroup instance or None if not initialized.
    """
    return _global_comm_group


def set_global_comm_group(comm_group: CPUCommunicationGroup) -> None:
    """
    Set the global communication group instance.

    Args:
        comm_group: CPUCommunicationGroup instance to set as global.
    """
    global _global_comm_group
    _global_comm_group = comm_group


def initialize_cpu_group(
    backend: str = "gloo",
    init_method: Optional[str] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
) -> CPUCommunicationGroup:
    """
    Initialize and return a CPU communication group.

    This is a convenience function that creates and initializes a
    CPUCommunicationGroup and sets it as the global instance.

    Args:
        backend: Communication backend (default: 'gloo').
        init_method: URL specifying how to initialize the process group.
        world_size: Total number of processes.
        rank: Rank of this process.

    Returns:
        Initialized CPUCommunicationGroup instance.
    """
    comm_group = CPUCommunicationGroup(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    comm_group.initialize()
    set_global_comm_group(comm_group)

    return comm_group
