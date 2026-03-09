"""Base multiplier abstraction."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["BaseMultiplier"]


class BaseMultiplier(PipelinedComponent):
    """Base multiplier abstraction.

    Parameters
    ----------
    x_width, y_width : int
    """

    def __init__(self, x_width: int, y_width: int) -> None:
        super().__init__()
        self.x_width = x_width
        self.y_width = y_width
        self.x = Signal(x_width, name="x")
        self.y = Signal(y_width, name="y")
        self.o = Signal(x_width + y_width, name="o")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        prod = Signal(self.x_width + self.y_width, name="prod")
        m.d.comb += prod.eq(self.x * self.y)
        p1 = Signal.like(prod, name="p1")
        m.d.sync += p1.eq(prod)
        out = Signal.like(prod, name="out")
        m.d.sync += out.eq(p1)
        m.d.comb += self.o.eq(out)
        return m
