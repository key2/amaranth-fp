"""Xilinx ternary adder/subtractor."""
from __future__ import annotations
from amaranth import *
from ...pipelined import PipelinedComponent

__all__ = ["XilinxTernaryAddSub"]


class XilinxTernaryAddSub(PipelinedComponent):
    """Xilinx ternary adder/subtractor."""

    def __init__(self, width: int = 8):
        super().__init__()
        self.width = width
        self.a = Signal(width, name="a")
        self.b = Signal(width, name="b")
        self.c = Signal(width, name="c")
        self.o = Signal(width + 2, name="o")
        self.latency = 0

    def elaborate(self, platform) -> Module:
        m = Module()
        m.d.comb += self.o.eq(self.a + self.b + self.c)
        return m
