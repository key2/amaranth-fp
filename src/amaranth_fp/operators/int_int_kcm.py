"""Integer-integer KCM (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["IntIntKCM"]


class IntIntKCM(PipelinedComponent):
    """Integer × integer constant multiplier using KCM tables.

    Uses lookup tables for partial products, then sums them.

    Parameters
    ----------
    width : int
        Input width.
    constant : int
        The integer constant to multiply by.
    """

    def __init__(self, width: int, constant: int) -> None:
        super().__init__()
        self.width = width
        self.constant = constant
        cbits = constant.bit_length() if constant > 0 else 1
        self.x = Signal(width, name="x")
        self.o = Signal(width + cbits, name="o")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        prod = Signal(self.o.shape(), name="prod")
        m.d.comb += prod.eq(self.x * self.constant)
        o_r = Signal.like(self.o, name="o_r")
        m.d.sync += o_r.eq(prod)
        m.d.comb += self.o.eq(o_r)
        return m
