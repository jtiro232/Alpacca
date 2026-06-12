# Alpacca - NumPy-facing quantized matrix type.
# MIT License. See LICENSE.
"""Compatibility surface for quantized GGUF matrix weights.

The implementation lives in :mod:`alpacca.tensor` so both dense and quantized
matrices share the same dispatch helpers. This module gives the quantized
matrix backend a focused import path for tests and future extensions.
"""

from __future__ import annotations

from .tensor import QuantizedMatrix as QuantMatrix

__all__ = ["QuantMatrix"]
