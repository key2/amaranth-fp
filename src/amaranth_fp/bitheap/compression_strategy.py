"""Base compression strategy for bit heaps."""
from __future__ import annotations

from amaranth import *
from ..pipelined import PipelinedComponent

__all__ = ["CompressionStrategy"]


class CompressionStrategy(PipelinedComponent):
    """Base class for bit heap compression strategies.

    Subclasses implement different heuristics (first-fitting,
    max-efficiency, Parandeh-Afshar, etc.).

    Parameters
    ----------
    heap_width : int
        Maximum bit heap column count.
    """

    def __init__(self, heap_width: int = 64):
        super().__init__()
        self.heap_width = heap_width
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        # Base strategy: pass-through (subclasses override)
        return m
