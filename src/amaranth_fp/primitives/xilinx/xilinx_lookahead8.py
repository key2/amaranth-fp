"""Xilinx LOOKAHEAD8 carry-lookahead."""
from __future__ import annotations
from amaranth import *
from ...pipelined import PipelinedComponent

__all__ = ["XilinxLOOKAHEAD8"]


class XilinxLOOKAHEAD8(PipelinedComponent):
    """Xilinx LOOKAHEAD8 carry-lookahead."""

    def __init__(self):
        super().__init__()
        self.ci = Signal(1, name="ci")
        self.prop = Signal(8, name="prop")
        self.gen = Signal(8, name="gen")
        self.co = Signal(8, name="co")
        self.latency = 0

    def elaborate(self, platform) -> Module:
        m = Module()
        m.d.comb += self.co.eq(self.gen | (self.prop & self.ci))
        return m
