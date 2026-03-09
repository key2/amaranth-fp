"""Xilinx 4:2 compressor."""
from __future__ import annotations
from amaranth import *
from ...pipelined import PipelinedComponent

__all__ = ["XilinxFourToTwoCompressor"]


class XilinxFourToTwoCompressor(PipelinedComponent):
    """Xilinx 4:2 compressor."""

    def __init__(self, width: int = 1):
        super().__init__()
        self.width = width
        self.x0 = Signal(width, name="x0")
        self.x1 = Signal(width, name="x1")
        self.x2 = Signal(width, name="x2")
        self.x3 = Signal(width, name="x3")
        self.s = Signal(width, name="s")
        self.c = Signal(width, name="c")
        self.latency = 0

    def elaborate(self, platform) -> Module:
        m = Module()
        m.d.comb += [
            self.s.eq(self.x0 ^ self.x1 ^ self.x2 ^ self.x3),
            self.c.eq((self.x0 & self.x1) | (self.x2 & self.x3)),
        ]
        return m
