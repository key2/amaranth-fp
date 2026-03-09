"""One-hot decoder."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["OneHotDecoder"]


class OneHotDecoder(PipelinedComponent):
    """One-hot decoder.

    Parameters
    ----------
    width : int
        Number of input bits (output is 2**width).
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.x = Signal(width, name="x")
        self.o = Signal(1 << width, name="o")
        self.latency = 0

    def elaborate(self, platform) -> Module:
        m = Module()
        m.d.comb += self.o.eq(1 << self.x)
        return m
