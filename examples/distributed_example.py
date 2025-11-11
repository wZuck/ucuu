"""
Example demonstrating CPU communication group and remote execution features.

This example shows how to use UCUU's distributed features for remote execution.
Note: For actual distributed execution, you need to run this script on multiple
nodes with proper distributed setup (MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK).
"""

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install ucuu[distributed]")

if TORCH_AVAILABLE:
    from ucuu.decorator import ucuu

    # Example 1: Basic remote execution
    print("=" * 60)
    print("Example 1: Basic Remote Execution")
    print("=" * 60)

    # For demonstration, we'll use local execution
    # In production, initialize with proper distributed settings
    # comm_group = initialize_cpu_group(
    #     backend="gloo",
    #     init_method="tcp://master_node:29500",
    #     world_size=2,
    #     rank=0  # or 1 for the peer node
    # )

    @ucuu("package_utils.print_ucuu_hello", remote=True)
    def compute_remotely(x):
        """This function will execute on the peer node"""
        return x * 2

    result = compute_remotely(5)
    print(f"Result: {result}")
    print()

    # Example 2: Remote execution with tensors
    print("=" * 60)
    print("Example 2: Remote Execution with Tensors")
    print("=" * 60)

    @ucuu("package_utils.print_ucuu_hello", remote=True)
    def process_tensor(x):
        """Process tensor on remote peer"""
        return x * 2 + 1

    tensor_input = torch.tensor([1.0, 2.0, 3.0])
    print(f"Input tensor: {tensor_input}")
    result_tensor = process_tensor(tensor_input)
    print(f"Result tensor: {result_tensor}")
    print()

    # Example 3: Remote execution with custom preprocessing
    print("=" * 60)
    print("Example 3: Remote Execution with Custom Preprocessing")
    print("=" * 60)

    def normalize_input(input_dict):
        """Normalize input values"""
        if "x" in input_dict:
            print(f"Preprocessing: Normalizing input {input_dict['x']}")
            input_dict["x"] = input_dict["x"] / 10.0
        return input_dict

    def scale_output(output):
        """Scale output values"""
        print(f"Postprocessing: Scaling output {output}")
        return output * 10.0

    @ucuu(
        "package_utils.print_ucuu_hello",
        remote=True,
        custom_preprocess=normalize_input,
        custom_postprocess=scale_output,
    )
    def process_with_transforms(x):
        """Process with preprocessing and postprocessing"""
        return x * 2

    result = process_with_transforms(50)
    print(f"Final result: {result}")
    print()

    # Example 4: Normal (non-remote) execution
    print("=" * 60)
    print("Example 4: Normal Execution (remote=False)")
    print("=" * 60)

    @ucuu("package_utils.print_ucuu_hello", remote=False, ending_words="local mode")
    def local_function(x):
        """This executes locally"""
        print(f"Executing locally with input: {x}")
        return x * 3

    result = local_function(7)
    print(f"Result: {result}")
    print()

else:
    print("\nTo use distributed features, install PyTorch:")
    print("pip install ucuu[distributed]")
