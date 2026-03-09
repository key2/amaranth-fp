"""Xilinx CARRY4 primitive for fast carry chains."""
from __future__ import annotations
from amaranth import *
from ...pipelined import PipelinedComponent

__all__ = ["XilinxCARRY4"]


class XilinxCARRY4(PipelinedComponent):
    """Xilinx CARRY4 primitive for fast carry chains."""

    def __init__(self):
        super().__init__()
        self.ci = Signal(1, name="ci")
        self.di = Signal(4, name="di")
        self.s = Signal(4, name="s")
        self.co = Signal(4, name="co")
        self.o = Signal(4, name="o")
        self.latency = 0

    def elaborate(self, platform) -> Module:
        m = Module()
        m.d.comb += [self.o.eq(self.s ^ self.di), self.co.eq(self.di)]
        return m
