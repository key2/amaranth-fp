"""Fixed-point 3D norm (non-CORDIC)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["Fix3DNorm"]


class Fix3DNorm(PipelinedComponent):
    """Fixed-point 3D norm (non-CORDIC).

    Parameters
    ----------
    msb_in, lsb_in : int
    """

    def __init__(self, msb_in: int, lsb_in: int) -> None:
        super().__init__()
        self.msb_in = msb_in
        self.lsb_in = lsb_in
        w = msb_in - lsb_in + 1
        self.x = Signal(w, name="x")
        self.y = Signal(w, name="y")
        self.z = Signal(w, name="z")
        self.o = Signal(w, name="o")
        self.latency = 3

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.msb_in - self.lsb_in + 1
        sq = Signal(2*w+2, name="sq")
        m.d.comb += sq.eq(self.x * self.x + self.y * self.y + self.z * self.z)
        s1 = Signal(2*w+2, name="s1")
        m.d.sync += s1.eq(sq)
        s2 = Signal(w, name="s2")
        m.d.sync += s2.eq(s1[w:2*w])
        out = Signal(w, name="out")
        m.d.sync += out.eq(s2)
        m.d.comb += self.o.eq(out)
        return m
