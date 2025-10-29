# You Can You Up (UCUU) 🚀
--------

[![Tests](https://github.com/wZuck/ucuu/actions/workflows/python-app.yml/badge.svg)](https://github.com/wZuck/ucuu/actions/workflows/python-app.yml) [![GitHub Pages](https://github.com/wZuck/ucuu/actions/workflows/gh-pages.yml/badge.svg?branch=master)](https://github.com/wZuck/ucuu/actions/workflows/gh-pages.yml) [![PyPI Package](https://github.com/wZuck/ucuu/actions/workflows/publish.yml/badge.svg)](https://github.com/wZuck/ucuu/actions/workflows/publish.yml)
> **⚠️ Important:**  
> The majority of the code in this repository is generated using AI coding tools such as GitHub Copilot (GPT-4o) and TRAE (Doubao 1.5 Pro).  
> 
> **Distributed features contributor:** GitHub Copilot (Claude 3.7 Sonnet)  

## 1. Brief Introduction

**UCUU** is a Python utility library for function wrapping, proxying, and distributed computing. It supports adding proxy logic to functions via decorators or patching, and provides distributed communication capabilities for remote execution across multiple nodes using PyTorch.

---

## 2. Install and Usage ⚙️

### 2.1 Installation

You can install UCUU from PyPI:

```bash
pip install ucuu
```

For distributed computing features with PyTorch support:

```bash
pip install ucuu[distributed]
```

Or, clone this repository and install locally:

```bash
git clone https://github.com/wZuck/ucuu.git
cd ucuu
pip install .
# or with distributed features
pip install .[distributed]
```

### 2.2 Usage

#### 2.2.1 Decorator mode (wrap for functions)

> Example: Use a decorator to add a proxy function.  
> **Effect:** When `my_func` is called, it prints "My function logic" and then the proxy prints "Hello, hello ucuu".

```python
from ucuu.decorator import ucuu

@ucuu("package_utils.print_ucuu_hello", ending_words="hello ucuu")
def my_func():
    print("My function logic")
```

#### 2.2.2 Patch mode (wrap for external functions)

> Example: Patch an existing function with a proxy.  
> **Effect:** When `some_func` is called, it prints "Original logic" and then the proxy prints "Hello, patch mode".

```python
from ucuu.decorator import ucuu

def some_func():
    print("Original logic")

some_func = ucuu("package_utils.print_ucuu_hello", ending_words="patch mode")(some_func)
```

#### 2.2.3 Register proxy functions

> Example: Implement a proxy function to be called by the decorator or patch.  
> **Effect:** Prints different messages depending on the `ending_words` argument.

```python
# tests/package_utils/test_print.py
def print_ucuu_hello(ending_words=None, *args, **kwargs):
    if ending_words is None:
        print("No Ending Words.")
    elif ending_words == "please raise errors":
        raise NotImplementedError('Raise Error due to requests')
    else:
        print(f"Hello, {ending_words}")
```

#### 2.2.4 Remote execution with CPU communication groups

> Example: Set up distributed communication and execute functions remotely across nodes.  
> **Effect:** Functions can be executed on remote peers with automatic tensor device management.

```python
import torch
from ucuu.decorator import ucuu
from ucuu.distributed import initialize_cpu_group

# Initialize CPU communication group on each node
# Node A (rank 0):
comm_group = initialize_cpu_group(
    backend="gloo",
    init_method="tcp://master_node:29500",
    world_size=2,
    rank=0
)

# Node B (rank 1):
comm_group = initialize_cpu_group(
    backend="gloo",
    init_method="tcp://master_node:29500",
    world_size=2,
    rank=1
)

# Use remote decorator to execute on peer
# peer_rank is optional - if not specified, uses current rank
# In typical scenarios, both peers have matching ranks
@ucuu("package_utils.print_ucuu_hello", remote=True)
def compute_on_peer(x):
    """This function will execute on the peer node with the same rank"""
    return x * 2

# Tensors are automatically moved to CPU for communication
# and restored to original device after execution
gpu_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
result = compute_on_peer(gpu_tensor)  # Executed on peer, result back on GPU
```

#### 2.2.5 Custom preprocessing and postprocessing for remote execution

> Example: Apply custom transformations to inputs and outputs during remote execution.  
> **Effect:** Allows flexible data handling for distributed computing scenarios.

```python
from ucuu.decorator import ucuu

def custom_preprocess(input_dict):
    """Preprocess inputs before sending to remote peer"""
    if 'x' in input_dict:
        # Normalize input
        input_dict['x'] = input_dict['x'] / 255.0
    return input_dict

def custom_postprocess(output):
    """Postprocess output received from remote peer"""
    # Scale output back
    return output * 255.0

@ucuu(
    "package_utils.print_ucuu_hello",
    remote=True,
    custom_preprocess=custom_preprocess,
    custom_postprocess=custom_postprocess
)
def process_data(x):
    return x * 2
```

---

## 3. Distributed Communication Features 🌐

### 3.1 CPU Communication Groups

UCUU provides CPU-based communication groups for distributed computing across multiple nodes using PyTorch's distributed primitives.

#### Key Features:
- **Cross-node communication**: Establish communication channels between processes on different nodes
- **CPU-optimized**: Uses `gloo` backend for efficient CPU-based operations
- **Automatic device management**: Tensors are automatically moved to CPU for communication
- **Flexible initialization**: Supports environment variables or explicit configuration

#### Basic Usage:

```python
from ucuu.distributed import initialize_cpu_group, CPUCommunicationGroup

# Method 1: Using environment variables
# Set MASTER_ADDR, MASTER_PORT, WORLD_SIZE, and RANK
comm_group = initialize_cpu_group()

# Method 2: Explicit configuration
comm_group = initialize_cpu_group(
    backend="gloo",
    init_method="tcp://192.168.1.100:29500",
    world_size=4,
    rank=0
)

# Create peer-to-peer groups
peer_group = comm_group.create_peer_group(ranks=[0, 1])

# Send/receive tensors
if comm_group.get_rank() == 0:
    tensor = torch.tensor([1.0, 2.0, 3.0])
    comm_group.send_tensor(tensor, dst=1)
else:
    tensor = torch.zeros(3)
    comm_group.recv_tensor(tensor, src=0)

# Synchronization
comm_group.barrier()

# Cleanup when done
comm_group.cleanup()
```

### 3.2 Remote Decorator

The `@ucuu` decorator supports a `remote` attribute for executing functions on remote peers:

#### Parameters:
- `remote` (bool): Enable remote execution (default: False)
- `peer_rank` (int, optional): Rank of the peer to execute on. If not specified, uses the current rank (suitable for scenarios where both peers have matching ranks)
- `custom_preprocess` (Callable): Function to preprocess inputs before sending
- `custom_postprocess` (Callable): Function to postprocess outputs after receiving

#### Device Management:
- Input tensors are automatically converted to CPU before transmission
- Output tensors are automatically converted back to the original device
- Custom preprocessing/postprocessing can be applied at each stage

#### Note:
In typical distributed scenarios, peer nodes have matching ranks (e.g., rank 0 on node A communicates with rank 0 on node B). Therefore, `peer_rank` can usually be omitted and will default to the current rank.

---

## 4. Demos in Testcases 🧪

- See the `tests/` directory for test cases covering:
  - Decorator usage and patching
  - Exception handling
  - Argument binding
  - Distributed communication groups
  - Remote execution with preprocessing/postprocessing


---

## 5. License 📄

[License](./LICENSE)

---

## 6. Contribute 🤝

Contributions via PR or issues are welcome!  
For suggestions or questions, please leave a message at [GitHub Issues](https://github.com/wZuck/ucuu/issues).

---

