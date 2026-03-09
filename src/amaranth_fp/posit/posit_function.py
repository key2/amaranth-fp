"""Generic posit function evaluation."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["PositFunction"]


class PositFunction(PipelinedComponent):
    """Generic posit function evaluation.

    Parameters
    ----------
    width : int
    es : int
    func : str
    """

    def __init__(self, width: int, es: int, func: str = "x") -> None:
        super().__init__()
        self.width = width
        self.es = es
        self.func = func
        self.x = Signal(width, name="x")
        self.o = Signal(width, name="o")
        self.latency = 3

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        s0 = Signal(w); m.d.sync += s0.eq(self.x)
        s1 = Signal(w); m.d.sync += s1.eq(s0)
        out = Signal(w); m.d.sync += out.eq(s1)
        m.d.comb += self.o.eq(out)
        return m
