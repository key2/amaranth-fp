"""Fixed-point real constant multiplier."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixRealConstMult"]


class FixRealConstMult(PipelinedComponent):
    """Fixed-point real constant multiplier.

    Parameters
    ----------
    msb_in, lsb_in, lsb_out : int
    constant : float
    """

    def __init__(self, msb_in: int, lsb_in: int, lsb_out: int, constant: float) -> None:
        super().__init__()
        self.msb_in = msb_in
        self.lsb_in = lsb_in
        self.lsb_out = lsb_out
        self.constant = constant
        w_in = msb_in - lsb_in + 1
        w_out = msb_in - lsb_out + 1
        self.x = Signal(w_in, name="x")
        self.o = Signal(w_out, name="o")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        w_in = self.msb_in - self.lsb_in + 1
        w_out = self.msb_in - self.lsb_out + 1
        c_int = int(self.constant * (1 << (-self.lsb_in)))
        prod = Signal(2 * w_in, name="prod")
        m.d.comb += prod.eq(self.x * c_int)
        p1 = Signal(2 * w_in, name="p1")
        m.d.sync += p1.eq(prod)
        o1 = Signal(w_out, name="o1")
        m.d.sync += o1.eq(p1[:w_out])
        m.d.comb += self.o.eq(o1)
        return m
