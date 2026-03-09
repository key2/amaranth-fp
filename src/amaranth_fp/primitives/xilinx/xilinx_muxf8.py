"""Xilinx MUXF8 (2:1 mux)."""
from __future__ import annotations
from amaranth import *
from ...pipelined import PipelinedComponent

__all__ = ["XilinxMUXF8"]


class XilinxMUXF8(PipelinedComponent):
    """Xilinx MUXF8 (2:1 mux)."""

    def __init__(self):
        super().__init__()
        self.i0 = Signal(1, name="i0")
        self.i1 = Signal(1, name="i1")
        self.sel = Signal(1, name="sel")
        self.o = Signal(1, name="o")
        self.latency = 0

    def elaborate(self, platform) -> Module:
        m = Module()
        with m.If(self.sel):
            m.d.comb += self.o.eq(self.i1)
        with m.Else():
            m.d.comb += self.o.eq(self.i0)
        return m
