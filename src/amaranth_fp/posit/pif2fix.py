"""PIF to fixed-point conversion."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["PIF2Fix"]


class PIF2Fix(PipelinedComponent):
    """PIF to fixed-point conversion.

    Parameters
    ----------
    width : int
    fix_width : int
    """

    def __init__(self, width: int, fix_width: int) -> None:
        super().__init__()
        self.width = width
        self.fix_width = fix_width
        self.x = Signal(width, name="x")
        self.o = Signal(fix_width, name="o")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        s1 = Signal(self.width, name="s1")
        m.d.sync += s1.eq(self.x)
        out = Signal(self.fix_width, name="out")
        m.d.sync += out.eq(s1[:self.fix_width])
        m.d.comb += self.o.eq(out)
        return m
