"""Parameterized integer adder (pipelined, 1 stage)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["IntAdder"]


class IntAdder(PipelinedComponent):
    """Pipelined integer adder (1-cycle latency).

    Parameters
    ----------
    width : int
        Bit-width of operands.

    Attributes
    ----------
    a : Signal(width), in
    b : Signal(width), in
    cin : Signal(1), in
    s : Signal(width), out — sum
    cout : Signal(1), out — carry out
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.a = Signal(width, name="a")
        self.b = Signal(width, name="b")
        self.cin = Signal(1, name="cin")
        self.s = Signal(width, name="s")
        self.cout = Signal(1, name="cout")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width

        result = Signal(w + 1, name="result")
        m.d.comb += result.eq(self.a + self.b + self.cin)

        s_r = Signal(w, name="s_r")
        cout_r = Signal(1, name="cout_r")
        m.d.sync += [
            s_r.eq(result[:w]),
            cout_r.eq(result[w]),
        ]
        m.d.comb += [
            self.s.eq(s_r),
            self.cout.eq(cout_r),
        ]
        return m
