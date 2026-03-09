"""Xilinx FDCE (D flip-flop with clock enable and clear)."""
from __future__ import annotations
from amaranth import *
from ...pipelined import PipelinedComponent

__all__ = ["XilinxFDCE"]


class XilinxFDCE(PipelinedComponent):
    """Xilinx FDCE (D flip-flop with clock enable and clear)."""

    def __init__(self):
        super().__init__()
        self.d = Signal(1, name="d")
        self.ce = Signal(1, name="ce")
        self.clr = Signal(1, name="clr")
        self.q = Signal(1, name="q")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        with m.If(self.clr):
            m.d.sync += self.q.eq(0)
        with m.Elif(self.ce):
            m.d.sync += self.q.eq(self.d)
        return m
