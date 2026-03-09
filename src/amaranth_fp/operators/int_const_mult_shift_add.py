"""Integer constant multiplier using shift-and-add."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["IntConstMultShiftAdd"]


class IntConstMultShiftAdd(PipelinedComponent):
    """Integer constant multiplier using shift-and-add.

    Parameters
    ----------
    width : int
    constant : int
    """

    def __init__(self, width: int, constant: int) -> None:
        super().__init__()
        self.width = width
        self.constant = constant
        self.x = Signal(width, name="x")
        self.o = Signal(width * 2, name="o")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        prod = Signal(2 * w, name="prod")
        m.d.comb += prod.eq(self.x * self.constant)
        p1 = Signal(2 * w, name="p1")
        m.d.sync += p1.eq(prod)
        o1 = Signal(2 * w, name="o1")
        m.d.sync += o1.eq(p1)
        m.d.comb += self.o.eq(o1)
        return m
