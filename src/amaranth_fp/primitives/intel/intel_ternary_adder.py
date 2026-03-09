"""Intel ternary adder primitive."""
from __future__ import annotations
from amaranth import *
from ...pipelined import PipelinedComponent

__all__ = ["IntelTernaryAdder"]


class IntelTernaryAdder(PipelinedComponent):
    """Three-input adder using Intel ALM ternary addition.

    Parameters
    ----------
    width : int
    """

    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.a = Signal(width, name="a")
        self.b = Signal(width, name="b")
        self.c = Signal(width, name="c")
        self.o = Signal(width + 2, name="o")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        out = Signal(self.width + 2, name="out")
        m.d.sync += out.eq(self.a + self.b + self.c)
        m.d.comb += self.o.eq(out)
        return m
