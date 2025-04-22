import importlib
import inspect
import traceback
from functools import wraps
from ucuu.log import setup_logger

logger = setup_logger()


def ucuu(proxy_func_name, **ucuu_kwargs):
    module_path, func_name = proxy_func_name.rsplit(".", 1)

    def decorator(origin_func):
        @wraps(origin_func)
        def wrapper(self, *origin_args, **origin_kwargs):
            origin_output = origin_func(self, *origin_args, **origin_kwargs)

            sig = inspect.signature(origin_func)
            bound_args = sig.bind(self, *origin_args, **origin_kwargs)
            bound_args.apply_defaults()

            input_dict = {**bound_args.arguments, **ucuu_kwargs, "origin_output": origin_output}
            input_dict.pop("self", None)

            try:
                module = importlib.import_module(module_path)
                proxy_func = getattr(module, func_name)
                proxy_result = proxy_func(**input_dict)

                logger.info(
                    f"[bold green] Proxy Function Done [/bold green]\n"
                    f"Proxy Function: [cyan]{proxy_func}[/cyan]\n"
                    f"Inputs: {input_dict}\n"
                    f"Returns: [yellow]{proxy_result}[/yellow]"
                )
                return origin_output

            except Exception as e:
                logger.error(
                    f"[bold red] Proxy Function Error [/bold red]\n"
                    f"Proxy Function: [cyan]{proxy_func}[/cyan]\n"
                    f"Inputs: {input_dict}\n"
                    f"Error: [red]{str(e)}[/red]\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                return origin_output

        return wrapper

    return decorator
