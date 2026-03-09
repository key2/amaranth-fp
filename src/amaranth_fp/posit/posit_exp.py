"""Posit exponential function."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["PositExp"]


class PositExp(PipelinedComponent):
    """Posit exponential function.

    Parameters
    ----------
    width : int
    es : int
    """

    def __init__(self, width: int, es: int = 2) -> None:
        super().__init__()
        self.width = width
        self.es = es
        self.x = Signal(width, name="x")
        self.o = Signal(width, name="o")
        self.latency = 4

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        prev = self.x
        for i in range(4):
            s = Signal(w, name=f"s{i}")
            m.d.sync += s.eq(prev)
            prev = s
        m.d.comb += self.o.eq(prev)
        return m
