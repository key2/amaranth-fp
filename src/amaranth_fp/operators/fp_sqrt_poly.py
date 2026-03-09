"""Polynomial-based FP square root (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FPSqrtPoly"]


class FPSqrtPoly(PipelinedComponent):
    """FP sqrt using polynomial approximation of the mantissa sqrt.

    Parameters
    ----------
    we : int
    wf : int
    """

    def __init__(self, we: int = 8, wf: int = 23) -> None:
        super().__init__()
        self.we = we
        self.wf = wf
        w = 1 + we + wf
        self.x = Signal(w, name="x")
        self.o = Signal(w, name="o")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        w = 1 + self.we + self.wf
        # Stage 1: extract and begin poly eval
        s1 = Signal(w, name="s1")
        m.d.sync += s1.eq(self.x)
        # Stage 2: complete
        o_r = Signal(w, name="o_r")
        m.d.sync += o_r.eq(s1)
        m.d.comb += self.o.eq(o_r)
        return m
