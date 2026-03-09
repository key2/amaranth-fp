"""Generic multiplier primitive (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["GenericMult"]


class GenericMult(PipelinedComponent):
    """Parameterised unsigned multiplier.

    Parameters
    ----------
    a_width : int
    b_width : int
    """

    def __init__(self, a_width: int, b_width: int) -> None:
        super().__init__()
        self.a_width = a_width
        self.b_width = b_width
        self.a = Signal(a_width, name="a")
        self.b = Signal(b_width, name="b")
        self.o = Signal(a_width + b_width, name="o")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        prod = Signal(self.a_width + self.b_width, name="prod")
        m.d.comb += prod.eq(self.a * self.b)
        o_r = Signal.like(self.o, name="o_r")
        m.d.sync += o_r.eq(prod)
        m.d.comb += self.o.eq(o_r)
        return m
