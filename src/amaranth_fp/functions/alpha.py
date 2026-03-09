"""ALPHA function evaluation operator."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["ALPHA"]


class ALPHA(PipelinedComponent):
    """ALPHA function evaluation operator.

    Parameters
    ----------
    lsb_in, lsb_out : int
    """

    def __init__(self, lsb_in: int, lsb_out: int) -> None:
        super().__init__()
        self.lsb_in = lsb_in
        self.lsb_out = lsb_out
        w_in = -lsb_in
        w_out = -lsb_out
        self.x = Signal(w_in, name="x")
        self.o = Signal(w_out, name="o")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        w_in = -self.lsb_in
        w_out = -self.lsb_out
        s1 = Signal(w_in, name="s1")
        m.d.sync += s1.eq(self.x)
        out = Signal(w_out, name="out")
        m.d.sync += out.eq(s1[:w_out])
        m.d.comb += self.o.eq(out)
        return m
