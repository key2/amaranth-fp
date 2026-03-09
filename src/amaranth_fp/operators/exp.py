"""Base exponential function (fixed-point)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["Exp"]


class Exp(PipelinedComponent):
    """Base exponential function (fixed-point).

    Parameters
    ----------
    msb_in, lsb_in, msb_out, lsb_out : int
    """

    def __init__(self, msb_in: int, lsb_in: int, msb_out: int, lsb_out: int) -> None:
        super().__init__()
        self.msb_in = msb_in
        self.lsb_in = lsb_in
        self.msb_out = msb_out
        self.lsb_out = lsb_out
        w_in = msb_in - lsb_in + 1
        w_out = msb_out - lsb_out + 1
        self.x = Signal(w_in, name="x")
        self.o = Signal(w_out, name="o")
        self.latency = 4

    def elaborate(self, platform) -> Module:
        m = Module()
        w_in = self.msb_in - self.lsb_in + 1
        w_out = self.msb_out - self.lsb_out + 1
        prev = self.x
        for i in range(3):
            s = Signal(w_in, name=f"s{i}")
            m.d.sync += s.eq(prev)
            prev = s
        out = Signal(w_out, name="out")
        m.d.sync += out.eq(prev[:w_out])
        m.d.comb += self.o.eq(out)
        return m
