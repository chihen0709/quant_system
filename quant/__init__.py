"""Reusable research pipeline primitives for the hybrid quant system.

Keep this package initializer lightweight. In particular, do not import
``quant.dl`` here, because ``python -m quant.dl`` first imports ``quant`` and
would otherwise load the module before runpy executes it.
"""

from .features import build_feature_frame

__all__ = ["build_feature_frame"]
