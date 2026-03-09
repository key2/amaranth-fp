"""Xilinx 5-input LUT."""
from __future__ import annotations
from amaranth import *
from ...pipelined import PipelinedComponent

__all__ = ["XilinxLUT5"]


class XilinxLUT5(PipelinedComponent):
    """Xilinx 5-input LUT."""

    def __init__(self, init: int = 0):
        super().__init__()
        self.init = init
        self.i = Signal(5, name="i")
        self.o = Signal(1, name="o")
        self.latency = 0

    def elaborate(self, platform) -> Module:
        m = Module()
        m.d.comb += self.o.eq((self.init >> self.i) & 1)
        return m
