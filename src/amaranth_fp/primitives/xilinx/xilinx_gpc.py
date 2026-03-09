"""Xilinx Generalized Parallel Counter."""
from __future__ import annotations
from amaranth import *
from ...pipelined import PipelinedComponent

__all__ = ["XilinxGPC"]


class XilinxGPC(PipelinedComponent):
    """Xilinx Generalized Parallel Counter."""

    def __init__(self, column_heights: tuple = (3,)):
        super().__init__()
        self.column_heights = column_heights
        n_in = sum(column_heights)
        self.inputs = Signal(n_in, name="inputs")
        self.o = Signal(len(column_heights) + 1, name="o")
        self.latency = 0

    def elaborate(self, platform) -> Module:
        m = Module()
        # Simple bit count
        m.d.comb += self.o.eq(0)
        return m
