"""Constant division by 3 for sine polynomial."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["ConstDiv3ForSinPoly"]


class ConstDiv3ForSinPoly(PipelinedComponent):
    """Constant division by 3 for sine polynomial.

    Parameters
    ----------
    width : int
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.x = Signal(width, name="x")
        self.o = Signal(width, name="o")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        # x/3 ≈ x * 0xAAAB >> 17 for 16-bit
        out = Signal(self.width, name="out")
        m.d.sync += out.eq(self.x // 3)
        m.d.comb += self.o.eq(out)
        return m
