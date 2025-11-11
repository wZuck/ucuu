"""
UCUU - You Can You Up

A Python utility library for function wrapping, proxying, and distributed computing.
"""

from ucuu.decorator import ucuu

__version__ = "0.1.1"
__all__ = ["ucuu"]

# Try to import distributed module, but don't fail if PyTorch is not available
try:
    from ucuu import distributed

    __all__.append("distributed")
except ImportError:
    # Optional dependency: ignore if PyTorch is not available
    pass
