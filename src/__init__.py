"""
__init__ module
"""

# src/__init__.py
from . import preprocessing

__version__ = "1.0.0"
__author__ = "Diego Suarez Etulain"

print(f"Loading {__name__} version {__version__}")

__all__ = ["preprocessing", "__version__"]
