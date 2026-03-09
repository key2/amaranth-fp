"""Max-efficiency compression strategy."""
from __future__ import annotations
from .compression_strategy import CompressionStrategy

__all__ = ["MaxEfficiencyCompressionStrategy"]


class MaxEfficiencyCompressionStrategy(CompressionStrategy):
    """Selects compressors maximizing bit reduction efficiency."""
    pass
