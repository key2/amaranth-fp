"""Fixed-point multiply-accumulate."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixMultAdd"]


class FixMultAdd(PipelinedComponent):
    """Fixed-point multiply-accumulate.

    Parameters
    ----------
    width : int
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.a = Signal(width, name="a")
        self.b = Signal(width, name="b")
        self.c = Signal(width, name="c")
        self.o = Signal(2 * width, name="o")
        self.latency = 3

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        prod = Signal(2 * w, name="prod")
        m.d.comb += prod.eq(self.a * self.b)
        p1 = Signal(2 * w, name="p1")
        m.d.sync += p1.eq(prod)
        acc = Signal(2 * w, name="acc")
        c_ext = Signal(2 * w, name="c_ext")
        m.d.comb += c_ext.eq(self.c)
        m.d.sync += acc.eq(p1 + c_ext)
        out = Signal(2 * w, name="out")
        m.d.sync += out.eq(acc)
        m.d.comb += self.o.eq(out)
        return m
