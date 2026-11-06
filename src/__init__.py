# src/__init__.py

__version__ = "1.0.0"
__author__ = "Diego Suarez Etulain"

print(f"Loading {__name__} version {__version__}")

from . import preprocessing

__all__ = ['preprocessing', '__version__']
