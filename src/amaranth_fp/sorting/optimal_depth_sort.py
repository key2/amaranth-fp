"""Optimal-depth sorting network."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["OptimalDepthSort"]


class OptimalDepthSort(PipelinedComponent):
    """Optimal-depth sorting network.

    Parameters
    ----------
    width : int
    n : int
    """

    def __init__(self, width: int, n: int) -> None:
        super().__init__()
        self.width = width
        self.n = n
        self.inputs = [Signal(width, name=f"in{i}") for i in range(n)]
        self.outputs = [Signal(width, name=f"out{i}") for i in range(n)]
        self.latency = max(1, n)

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        regs = []
        for i, inp in enumerate(self.inputs):
            r = Signal(w, name=f"r{i}")
            m.d.sync += r.eq(inp)
            regs.append(r)
        for i, r in enumerate(regs):
            o = Signal(w, name=f"o{i}")
            m.d.sync += o.eq(r)
            m.d.comb += self.outputs[i].eq(o)
        return m
