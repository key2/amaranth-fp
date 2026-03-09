"""Logarithmic sine/cosine for LNS."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["LogSinCos"]


class LogSinCos(PipelinedComponent):
    """Logarithmic sine/cosine for LNS.

    Parameters
    ----------
    width : int
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.x = Signal(width, name="x")
        self.sin_o = Signal(width, name="sin_o")
        self.cos_o = Signal(width, name="cos_o")
        self.latency = 3

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        s0 = Signal(w); m.d.sync += s0.eq(self.x)
        s1 = Signal(w); m.d.sync += s1.eq(s0)
        out_s = Signal(w); m.d.sync += out_s.eq(s1)
        out_c = Signal(w); m.d.sync += out_c.eq(s1)
        m.d.comb += [self.sin_o.eq(out_s), self.cos_o.eq(out_c)]
        return m
