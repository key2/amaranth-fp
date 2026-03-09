"""LNS division = subtract log values (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["LNSDiv"]


class LNSDiv(PipelinedComponent):
    """LNS division: subtract log values.

    Parameters
    ----------
    width : int
        Total LNS word width (sign + log_value).
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.a = Signal(width, name="a")
        self.b = Signal(width, name="b")
        self.o = Signal(width, name="o")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        lw = w - 1

        a_sign = self.a[w - 1]
        b_sign = self.b[w - 1]
        a_log = self.a[:lw]
        b_log = self.b[:lw]

        result_sign = Signal(name="div_rs")
        result_log = Signal(lw, name="div_rl")
        m.d.comb += [
            result_sign.eq(a_sign ^ b_sign),
            result_log.eq(a_log - b_log),
        ]

        o_r = Signal(w, name="lns_div_o")
        m.d.sync += o_r.eq(Cat(result_log, result_sign))
        m.d.comb += self.o.eq(o_r)

        return m
