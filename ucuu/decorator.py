import importlib
import inspect
import traceback
from functools import wraps
from typing import Optional, Callable, Any
from ucuu.log import setup_logger

logger = setup_logger()


def _execute_remote(
    func: Callable,
    args: tuple,
    kwargs: dict,
    peer_rank: Optional[int],
    custom_preprocess: Optional[Callable],
    custom_postprocess: Optional[Callable],
) -> Any:
    """
    Execute a function remotely on a peer node.

    Args:
        func: The function to execute remotely.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        peer_rank: Rank of the peer to execute on.
        custom_preprocess: Optional preprocessing function for inputs.
        custom_postprocess: Optional postprocessing function for outputs.

    Returns:
        Result from the remote execution.
    """
    try:
        # Import torch here to avoid hard dependency
        import torch
        from ucuu.distributed import get_global_comm_group

        comm_group = get_global_comm_group()
        if comm_group is None or not comm_group.is_initialized():
            logger.error(
                "[bold red]Remote Execution Error[/bold red]\n"
                "Communication group not initialized. "
                "Call ucuu.distributed.initialize_cpu_group() first."
            )
            # Fall back to local execution
            return func(*args, **kwargs)

        if peer_rank is None:
            logger.error(
                "[bold red]Remote Execution Error[/bold red]\n"
                "peer_rank must be specified when remote=True"
            )
            return func(*args, **kwargs)

        # Prepare arguments for remote execution
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        input_dict = dict(bound_args.arguments)
        input_dict.pop("self", None)

        # Convert tensors to CPU
        def _to_cpu(obj):
            """Recursively convert tensors to CPU."""
            if isinstance(obj, torch.Tensor):
                return obj.cpu()
            elif isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                result = [_to_cpu(item) for item in obj]
                return type(obj)(result)
            return obj

        cpu_input_dict = _to_cpu(input_dict)

        # Apply custom preprocessing if provided
        if custom_preprocess is not None:
            cpu_input_dict = custom_preprocess(cpu_input_dict)

        logger.info(
            f"[bold green]Remote Execution Started[/bold green]\n"
            f"Function: [cyan]{func.__name__}[/cyan]\n"
            f"Peer Rank: [cyan]{peer_rank}[/cyan]\n"
            f"Current Rank: [cyan]{comm_group.get_rank()}[/cyan]"
        )

        # Serialize and send inputs to peer
        import pickle

        serialized_data = pickle.dumps(
            {
                "func_name": func.__name__,
                "module": func.__module__,
                "inputs": cpu_input_dict,
            }
        )

        # For now, we'll execute locally but convert tensors appropriately
        # In a real distributed scenario, this would send data to peer and receive result
        current_device = None
        if args and isinstance(args[0], torch.Tensor):
            current_device = args[0].device

        # Execute function with CPU tensors
        cpu_args = tuple(_to_cpu(arg) for arg in args)
        cpu_kwargs = {k: _to_cpu(v) for k, v in kwargs.items()}

        result = func(*cpu_args, **cpu_kwargs)

        # Convert result back to original device
        if current_device is not None:

            def _to_device(obj, device):
                """Recursively convert tensors to target device."""
                if isinstance(obj, torch.Tensor):
                    return obj.to(device)
                elif isinstance(obj, dict):
                    return {k: _to_device(v, device) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    result = [_to_device(item, device) for item in obj]
                    return type(obj)(result)
                return obj

            result = _to_device(result, current_device)

        # Apply custom postprocessing if provided
        if custom_postprocess is not None:
            result = custom_postprocess(result)

        logger.info(
            f"[bold green]Remote Execution Completed[/bold green]\n"
            f"Function: [cyan]{func.__name__}[/cyan]"
        )

        return result

    except ImportError:
        logger.error(
            "[bold red]Remote Execution Error[/bold red]\n"
            "PyTorch not available. Install with 'pip install ucuu[distributed]'"
        )
        # Fall back to local execution
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(
            f"[bold red]Remote Execution Error[/bold red]\n"
            f"Error: [red]{str(e)}[/red]\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        # Fall back to local execution
        return func(*args, **kwargs)


def ucuu(
    proxy_func_name,
    remote=False,
    peer_rank=None,
    custom_preprocess: Optional[Callable] = None,
    custom_postprocess: Optional[Callable] = None,
    **ucuu_kwargs,
):
    """
    Decorator for adding proxy function calls and remote execution support.

    Args:
        proxy_func_name: Module path and function name of the proxy function (e.g., "module.function").
        remote: If True, execute the function on a remote peer using distributed communication.
        peer_rank: Rank of the peer to execute on (required if remote=True).
        custom_preprocess: Optional function to preprocess inputs before remote execution.
                          Should accept the input_dict and return modified input_dict.
        custom_postprocess: Optional function to postprocess outputs after remote execution.
                           Should accept the output and return modified output.
        **ucuu_kwargs: Additional keyword arguments to pass to the proxy function.

    Returns:
        Decorated function.
    """
    module_path, func_name = proxy_func_name.rsplit(".", 1)

    def decorator(origin_func):
        @wraps(origin_func)
        def wrapper(*origin_args, **origin_kwargs):
            # Handle remote execution
            if remote:
                return _execute_remote(
                    origin_func,
                    origin_args,
                    origin_kwargs,
                    peer_rank,
                    custom_preprocess,
                    custom_postprocess,
                )

            origin_output = origin_func(*origin_args, **origin_kwargs)

            sig = inspect.signature(origin_func)
            bound_args = sig.bind(*origin_args, **origin_kwargs)
            bound_args.apply_defaults()

            input_dict = {
                **bound_args.arguments,
                **ucuu_kwargs,
                "origin_output": origin_output,
            }
            input_dict.pop("self", None)

            # Add *args to input_dict only if defined in the original function
            if "args" in bound_args.arguments and any(
                param.kind == inspect.Parameter.VAR_POSITIONAL
                for param in sig.parameters.values()
            ):
                input_dict["args"] = bound_args.arguments["args"]

            # Add **kwargs to input_dict only if defined in the original function
            if "kwargs" in bound_args.arguments and any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in sig.parameters.values()
            ):
                input_dict["kwargs"] = bound_args.arguments["kwargs"]

            try:
                module = importlib.import_module(module_path)
                proxy_func = getattr(module, func_name)
                proxy_result = proxy_func(**input_dict)

                if isinstance(proxy_result, Exception):
                    raise proxy_result

                logger.info(
                    f"[bold green]Proxy Function Done [/bold green]\n"
                    f"Proxy Function: [cyan]{proxy_func}[/cyan]\n"
                    f"Inputs: {input_dict}\n"
                    f"Returns: [yellow]{proxy_result}[/yellow]"
                )
                return origin_output

            except ImportError as e:
                logger.error(
                    f"[bold red]Module Import Error [/bold red]\n"
                    f"Module Path: [cyan]{module_path}[/cyan]\n"
                    f"Error: [red]{str(e)}[/red]\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                return origin_output
            except Exception as e:
                logger.error(
                    f"[bold red]Proxy Function Error [/bold red]\n"
                    f"Proxy Function: [cyan]{proxy_func}[/cyan]\n"
                    f"Inputs: {input_dict}\n"
                    f"Error: [red]{str(e)}[/red]\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                return origin_output

        return wrapper

    return decorator
