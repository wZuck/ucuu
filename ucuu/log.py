import logging
import os
from rich.logging import RichHandler


def setup_logger(logger_name=None):
    """
    Set up an independent Rich logger for the calling module.
    If logger_name is not provided, use the caller's __file__ as the logger name.
    """
    # Use the caller's filename as logger name if not explicitly specified.
    if logger_name is None:
        import inspect

        frame = inspect.currentframe()
        # Go back two frames in the stack (setup_logger -> caller)
        caller_frame = frame.f_back.f_back
        logger_name = os.path.basename(caller_frame.f_globals["__file__"])

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Prevent adding duplicate handlers
    if not logger.handlers:
        handler = RichHandler(show_path=False, markup=True, rich_tracebacks=True)
        logger.addHandler(handler)

    return logger
