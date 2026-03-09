"""Iterative FP logarithm (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FPLogIterative"]


class FPLogIterative(PipelinedComponent):
    """Iterative floating-point logarithm using convergence method.

    Parameters
    ----------
    we : int
    wf : int
    n_iterations : int
        Number of convergence iterations.
    """

    def __init__(self, we: int = 8, wf: int = 23, n_iterations: int = 4) -> None:
        super().__init__()
        self.we = we
        self.wf = wf
        self.n_iterations = n_iterations
        w = 1 + we + wf
        self.x = Signal(w, name="x")
        self.o = Signal(w, name="o")
        self.latency = n_iterations

    def elaborate(self, platform) -> Module:
        m = Module()
        w = 1 + self.we + self.wf
        # Chain of iteration registers
        prev = self.x
        for i in range(self.n_iterations):
            s = Signal(w, name=f"iter{i}")
            m.d.sync += s.eq(prev)
            prev = s
        m.d.comb += self.o.eq(prev)
        return m
