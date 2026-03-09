"""DSP block multiplier."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["DSPBlock"]


class DSPBlock(PipelinedComponent):
    """DSP block multiplier.

    Parameters
    ----------
    x_width, y_width : int
    """

    def __init__(self, x_width: int = 18, y_width: int = 25) -> None:
        super().__init__()
        self.x_width = x_width
        self.y_width = y_width
        self.x = Signal(x_width, name="x")
        self.y = Signal(y_width, name="y")
        self.o = Signal(x_width + y_width, name="o")
        self.latency = 3

    def elaborate(self, platform) -> Module:
        m = Module()
        prod = Signal(self.x_width + self.y_width, name="prod")
        m.d.comb += prod.eq(self.x * self.y)
        p1 = Signal.like(prod, name="p1")
        m.d.sync += p1.eq(prod)
        p2 = Signal.like(prod, name="p2")
        m.d.sync += p2.eq(p1)
        out = Signal.like(prod, name="out")
        m.d.sync += out.eq(p2)
        m.d.comb += self.o.eq(out)
        return m
