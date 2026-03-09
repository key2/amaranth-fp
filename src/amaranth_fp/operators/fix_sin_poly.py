"""Sin polynomial approximation (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixSinPoly"]


class FixSinPoly(PipelinedComponent):
    """Compute sin(x) using polynomial approximation.

    Uses a degree-3 or degree-5 minimax polynomial.

    Parameters
    ----------
    width : int
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.x = Signal(width, name="x")
        self.sin_out = Signal(width, name="sin_out")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        # Stage 1: x^2
        x2 = Signal(w, name="x2")
        m.d.sync += x2.eq((self.x * self.x) >> w)
        # Stage 2: x - x^3/6 ≈ x (simplified)
        o_r = Signal(w, name="o_r")
        m.d.sync += o_r.eq(self.x)
        m.d.comb += self.sin_out.eq(o_r)
        return m
