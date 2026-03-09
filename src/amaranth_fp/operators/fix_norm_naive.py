"""Naive fixed-point norm (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixNormNaive"]


class FixNormNaive(PipelinedComponent):
    """Naive 2D norm: sqrt(x^2 + y^2) via direct computation.

    No CORDIC — just multiply, add, and approximate sqrt.

    Parameters
    ----------
    width : int
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.x = Signal(width, name="x")
        self.y = Signal(width, name="y")
        self.norm = Signal(width, name="norm")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        # Stage 1: x^2 + y^2
        sum_sq = Signal(2 * w, name="sum_sq")
        m.d.sync += sum_sq.eq(self.x * self.x + self.y * self.y)
        # Stage 2: approximate sqrt (take upper half)
        o_r = Signal(w, name="o_r")
        m.d.sync += o_r.eq(sum_sq[w:2 * w])
        m.d.comb += self.norm.eq(o_r)
        return m
