"""Compute both a+b and a-b simultaneously (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["IntDualAddSub"]


class IntDualAddSub(PipelinedComponent):
    """Compute a+b and a-b simultaneously.

    Parameters
    ----------
    width : int
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.a = Signal(width, name="a")
        self.b = Signal(width, name="b")
        self.sum_out = Signal(width + 1, name="sum_out")
        self.diff_out = Signal(signed(width + 1), name="diff_out")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width

        s = Signal(w + 1, name="s")
        d = Signal(signed(w + 1), name="d")
        m.d.comb += [s.eq(self.a + self.b), d.eq(self.a - self.b)]

        s_r = Signal(w + 1, name="s_r")
        d_r = Signal(signed(w + 1), name="d_r")
        m.d.sync += [s_r.eq(s), d_r.eq(d)]
        m.d.comb += [self.sum_out.eq(s_r), self.diff_out.eq(d_r)]

        return m
