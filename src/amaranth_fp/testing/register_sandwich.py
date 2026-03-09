"""Register sandwich for timing closure testing."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["RegisterSandwich"]


class RegisterSandwich(PipelinedComponent):
    """Register sandwich for timing closure testing.

    Parameters
    ----------
    width : int
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.x = Signal(width, name="x")
        self.o = Signal(width, name="o")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        s1 = Signal(self.width, name="s1")
        m.d.sync += s1.eq(self.x)
        out = Signal(self.width, name="out")
        m.d.sync += out.eq(s1)
        m.d.comb += self.o.eq(out)
        return m
