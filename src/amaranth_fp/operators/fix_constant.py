"""Fixed-point constant generator."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixConstant"]


class FixConstant(PipelinedComponent):
    """Fixed-point constant generator.

    Parameters
    ----------
    msb, lsb : int
    value : float
    """

    def __init__(self, msb: int, lsb: int, value: float) -> None:
        super().__init__()
        self.msb = msb
        self.lsb = lsb
        self.value = value
        w = msb - lsb + 1
        self.o = Signal(w, name="o")
        self.latency = 0

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.msb - self.lsb + 1
        int_val = int(self.value * (1 << (-self.lsb))) & ((1 << w) - 1)
        m.d.comb += self.o.eq(int_val)
        return m
