"""Xilinx generic multiplexer."""
from __future__ import annotations
from amaranth import *
from ...pipelined import PipelinedComponent

__all__ = ["XilinxGenericMux"]


class XilinxGenericMux(PipelinedComponent):
    """Xilinx generic multiplexer."""

    def __init__(self, width: int = 8, sel_bits: int = 3):
        super().__init__()
        self.width = width
        self.sel_bits = sel_bits
        self.inputs = Signal(width * (1 << sel_bits), name="inputs")
        self.sel = Signal(sel_bits, name="sel")
        self.o = Signal(width, name="o")
        self.latency = 0

    def elaborate(self, platform) -> Module:
        m = Module()
        for i in range(1 << self.sel_bits):
            with m.If(self.sel == i):
                m.d.comb += self.o.eq(self.inputs[i*self.width:(i+1)*self.width])
        return m
