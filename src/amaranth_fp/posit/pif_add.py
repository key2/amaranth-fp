"""PIF (Posit Internal Format) addition."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["PIFAdd"]


class PIFAdd(PipelinedComponent):
    """PIF (Posit Internal Format) addition.

    Parameters
    ----------
    width : int
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.a = Signal(width, name="a")
        self.b = Signal(width, name="b")
        self.o = Signal(width, name="o")
        self.latency = 3

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        s = Signal(signed(w+1), name="s")
        m.d.comb += s.eq(self.a.as_signed() + self.b.as_signed())
        s1 = Signal(signed(w+1), name="s1")
        m.d.sync += s1.eq(s)
        s2 = Signal(w, name="s2")
        m.d.sync += s2.eq(s1[:w])
        out = Signal(w, name="out")
        m.d.sync += out.eq(s2)
        m.d.comb += self.o.eq(out)
        return m
