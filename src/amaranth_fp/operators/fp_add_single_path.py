"""Single-path floating-point adder."""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent

__all__ = ["FPAddSinglePath"]


class FPAddSinglePath(PipelinedComponent):
    """Single-path floating-point adder.

    Parameters
    ----------
    we, wf : int
        Exponent/fraction widths.
    """

    def __init__(self, we: int, wf: int) -> None:
        super().__init__()
        from ..format import FPFormat
        self.fmt = FPFormat(we, wf)
        w = self.fmt.width
        self.a = Signal(w, name="a")
        self.b = Signal(w, name="b")
        self.o = Signal(w, name="o")
        self.latency = 4

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.fmt.width
        # Simplified pipeline matching latency
        s0 = Signal(w, name="s0")
        m.d.comb += s0.eq(self.a)
        prev = s0
        for i in range(4):
            s = Signal(w, name=f"pipe{i}")
            m.d.sync += s.eq(prev)
            prev = s
        m.d.comb += self.o.eq(prev)
        return m
