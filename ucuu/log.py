import logging
import os
from rich.logging import RichHandler


def setup_logger(logger_name=None):
    """
    Configure an independent rich logger for the calling module.
    If no logger_name is specified, use the caller's __file__ as the logger name.
    """
    # Get the caller's filename if no name is explicitly provided.
    if logger_name is None:
        import inspect

        frame = inspect.currentframe()
        # Trace back two levels (setup_logger -> caller)
        caller_frame = frame.f_back.f_back
        logger_name = os.path.basename(caller_frame.f_globals["__file__"])

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handler addition
    if not logger.handlers:
        handler = RichHandler(show_path=False, markup=True, rich_tracebacks=True)
        logger.addHandler(handler)

    return logger
