"""Xilinx n-to-m decoder."""
from __future__ import annotations
from amaranth import *
from ...pipelined import PipelinedComponent

__all__ = ["XilinxN2MDecoder"]


class XilinxN2MDecoder(PipelinedComponent):
    """Xilinx n-to-m decoder."""

    def __init__(self, n: int = 3):
        super().__init__()
        self.n = n
        self.sel = Signal(n, name="sel")
        self.o = Signal(1 << n, name="o")
        self.latency = 0

    def elaborate(self, platform) -> Module:
        m = Module()
        m.d.comb += self.o.eq(1 << self.sel)
        return m
