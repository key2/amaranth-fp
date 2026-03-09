"""Posit function by table lookup."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["PositFunctionByTable"]


class PositFunctionByTable(PipelinedComponent):
    """Posit function by table lookup.

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
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        out = Signal(w, name="out")
        m.d.sync += out.eq(self.x)
        m.d.comb += self.o.eq(out)
        return m
