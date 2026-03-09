"""LNS square root = halve log value (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["LNSSqrt"]


class LNSSqrt(PipelinedComponent):
    """LNS square root: halve the log value.

    Parameters
    ----------
    width : int
        Total LNS word width (sign + log_value).
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.a = Signal(width, name="a")
        self.o = Signal(width, name="o")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        lw = w - 1

        a_sign = self.a[w - 1]
        a_log = self.a[:lw]

        result_log = Signal(lw, name="sqrt_rl")
        m.d.comb += result_log.eq(a_log >> 1)

        o_r = Signal(w, name="lns_sqrt_o")
        m.d.sync += o_r.eq(Cat(result_log, a_sign))
        m.d.comb += self.o.eq(o_r)

        return m
