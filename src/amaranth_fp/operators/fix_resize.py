"""Fixed-point resize (sign/zero extend or truncate)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixResize"]


class FixResize(PipelinedComponent):
    """Fixed-point resize (sign/zero extend or truncate).

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
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w_in = self.msb_in - self.lsb_in + 1
        w_out = self.msb_out - self.lsb_out + 1
        out = Signal(w_out, name="out")
        m.d.sync += out.eq(self.x[:min(w_in, w_out)])
        m.d.comb += self.o.eq(out)
        return m
