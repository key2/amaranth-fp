"""Carry generation circuit for fast addition."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["CarryGenerationCircuit"]


class CarryGenerationCircuit(PipelinedComponent):
    """Carry generation circuit for fast addition.

    Parameters
    ----------
    width : int
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.a = Signal(width, name="a")
        self.b = Signal(width, name="b")
        self.carry = Signal(width, name="carry")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        c = Signal(w, name="c")
        m.d.comb += c.eq(self.a & self.b)
        cr = Signal(w, name="cr")
        m.d.sync += cr.eq(c)
        m.d.comb += self.carry.eq(cr)
        return m
