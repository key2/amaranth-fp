"""First-fitting compression strategy."""
from __future__ import annotations
from .compression_strategy import CompressionStrategy

__all__ = ["FirstFittingCompressionStrategy"]


class FirstFittingCompressionStrategy(CompressionStrategy):
    """Greedy first-fitting compressor selection."""
    pass
