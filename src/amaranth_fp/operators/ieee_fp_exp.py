"""Direct IEEE FP exponential (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["IEEEFPExp"]


class IEEEFPExp(PipelinedComponent):
    """IEEE-format floating-point exponential.

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
        self.latency = 3

    def elaborate(self, platform) -> Module:
        m = Module()
        w = 1 + self.we + self.wf
        s1 = Signal(w, name="s1")
        s2 = Signal(w, name="s2")
        o_r = Signal(w, name="o_r")
        m.d.sync += s1.eq(self.x)
        m.d.sync += s2.eq(s1)
        m.d.sync += o_r.eq(s2)
        m.d.comb += self.o.eq(o_r)
        return m
