"""Base squarer using LUT."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["BaseSquarerLUT"]


class BaseSquarerLUT(PipelinedComponent):
    """Base squarer using LUT.

    Parameters
    ----------
    width : int
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.x = Signal(width, name="x")
        self.o = Signal(2 * width, name="o")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        sq = Signal(2 * self.width, name="sq")
        m.d.comb += sq.eq(self.x * self.x)
        out = Signal.like(sq, name="out")
        m.d.sync += out.eq(sq)
        m.d.comb += self.o.eq(out)
        return m
