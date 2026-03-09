"""Shift register."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["ShiftReg"]


class ShiftReg(PipelinedComponent):
    """Shift register.

    Parameters
    ----------
    width : int
    depth : int
    """

    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.width = width
        self.depth = depth
        self.x = Signal(width, name="x")
        self.o = Signal(width, name="o")
        self.latency = depth

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        prev = self.x
        for i in range(self.depth):
            s = Signal(w, name=f"sr{i}")
            m.d.sync += s.eq(prev)
            prev = s
        m.d.comb += self.o.eq(prev)
        return m
