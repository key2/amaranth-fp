"""LNS atan and power operators (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["LNSAtanPow", "LNSLogSinCos"]


class LNSAtanPow(PipelinedComponent):
    """LNS-domain atan2(y,x) and power function.

    In LNS, power is a multiply: a^b = b * log(a).

    Parameters
    ----------
    width : int
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
        # Power in LNS: result_log = b_val * a_log (simplified as multiply)
        prod = Signal(2 * lw, name="prod")
        m.d.comb += prod.eq(self.a[:lw] * self.b[:lw])
        o_r = Signal(w, name="o_r")
        m.d.sync += o_r.eq(Cat(prod[:lw], self.a[w - 1]))
        m.d.comb += self.o.eq(o_r)
        return m


class LNSLogSinCos(PipelinedComponent):
    """LNS-domain log(sin) and log(cos) for trig in LNS.

    Parameters
    ----------
    width : int
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.angle = Signal(width, name="angle")
        self.log_sin = Signal(width, name="log_sin")
        self.log_cos = Signal(width, name="log_cos")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        # Placeholder: output registered angle as approximation
        s_r = Signal(w, name="s_r")
        c_r = Signal(w, name="c_r")
        m.d.sync += s_r.eq(self.angle)
        m.d.sync += c_r.eq(self.angle >> 1)
        m.d.comb += self.log_sin.eq(s_r)
        m.d.comb += self.log_cos.eq(c_r)
        return m
