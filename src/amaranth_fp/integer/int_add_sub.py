"""Integer add/subtract (pipelined, 1 stage)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["IntAddSub"]


class IntAddSub(PipelinedComponent):
    """Pipelined integer add/subtract (1-cycle latency).

    Parameters
    ----------
    width : int

    Attributes
    ----------
    a, b : Signal(width), in
    op : Signal(1), in — 0=add, 1=sub
    s : Signal(width), out
    cout : Signal(1), out
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.a = Signal(width, name="a")
        self.b = Signal(width, name="b")
        self.op = Signal(1, name="op")
        self.s = Signal(width, name="s")
        self.cout = Signal(1, name="cout")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width

        b_eff = Signal(w, name="b_eff")
        with m.If(self.op):
            m.d.comb += b_eff.eq(~self.b)
        with m.Else():
            m.d.comb += b_eff.eq(self.b)

        result = Signal(w + 1, name="result")
        m.d.comb += result.eq(self.a + b_eff + self.op)

        s_r = Signal(w, name="s_r")
        cout_r = Signal(1, name="cout_r")
        m.d.sync += [s_r.eq(result[:w]), cout_r.eq(result[w])]
        m.d.comb += [self.s.eq(s_r), self.cout.eq(cout_r)]
        return m
