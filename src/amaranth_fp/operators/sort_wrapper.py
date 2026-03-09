"""Sort wrapper for sorting networks."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["SortWrapper"]


class SortWrapper(PipelinedComponent):
    """Sort wrapper for sorting networks.

    Parameters
    ----------
    width : int
        Element width.
    n : int
        Number of elements.
    """

    def __init__(self, width: int, n: int) -> None:
        super().__init__()
        self.width = width
        self.n = n
        self.inputs = [Signal(width, name=f"in{i}") for i in range(n)]
        self.outputs = [Signal(width, name=f"out{i}") for i in range(n)]
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        # Stage 0: register inputs
        regs = []
        for i, inp in enumerate(self.inputs):
            r = Signal(w, name=f"r{i}")
            m.d.sync += r.eq(inp)
            regs.append(r)
        # Stage 1: output (sorting done externally)
        for i, r in enumerate(regs):
            o = Signal(w, name=f"o{i}")
            m.d.sync += o.eq(r)
            m.d.comb += self.outputs[i].eq(o)
        return m
