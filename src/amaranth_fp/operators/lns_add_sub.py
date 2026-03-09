"""Combined LNS add/sub operator (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["LNSAddSub"]


class LNSAddSub(PipelinedComponent):
    """LNS addition/subtraction using cotransformation tables.

    In LNS, add/sub requires evaluating sb(r) = log2(1 ± 2^r).

    Parameters
    ----------
    width : int
        Total LNS word width (sign + log_value).
    guard_bits : int
        Extra precision bits for the sb function table.
    """

    def __init__(self, width: int, guard_bits: int = 3) -> None:
        super().__init__()
        self.width = width
        self.guard_bits = guard_bits
        self.a = Signal(width, name="a")
        self.b = Signal(width, name="b")
        self.sub = Signal(name="sub")  # 0=add, 1=sub
        self.o = Signal(width, name="o")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        lw = w - 1

        a_sign = self.a[w - 1]
        b_sign = self.b[w - 1]
        a_log = self.a[:lw]
        b_log = self.b[:lw]

        # Determine which is larger (by log magnitude)
        a_bigger = Signal(name="a_bigger")
        m.d.comb += a_bigger.eq(a_log >= b_log)

        # r = smaller_log - larger_log (always <= 0 in fixed-point)
        r = Signal(lw, name="r")
        with m.If(a_bigger):
            m.d.comb += r.eq(b_log - a_log)
        with m.Else():
            m.d.comb += r.eq(a_log - b_log)

        # sb(r) approximation: log2(1 + 2^r) ≈ max(0, r+1) for add
        # For a proper implementation, this should use a lookup table.
        # Simplified: result_log = max_log + sb_approx
        sb_approx = Signal(lw, name="sb_approx")
        m.d.comb += sb_approx.eq(r >> 1)  # rough approximation

        max_log = Signal(lw, name="max_log")
        with m.If(a_bigger):
            m.d.comb += max_log.eq(a_log)
        with m.Else():
            m.d.comb += max_log.eq(b_log)

        result_log = Signal(lw, name="result_log")
        eff_sub = Signal(name="eff_sub")
        m.d.comb += eff_sub.eq(a_sign ^ b_sign ^ self.sub)

        m.d.comb += result_log.eq(max_log + sb_approx)

        result_sign = Signal(name="result_sign")
        with m.If(a_bigger):
            m.d.comb += result_sign.eq(a_sign)
        with m.Else():
            m.d.comb += result_sign.eq(b_sign ^ self.sub)

        # Pipeline stage 1
        s1_log = Signal(lw, name="s1_log")
        s1_sign = Signal(name="s1_sign")
        m.d.sync += s1_log.eq(result_log)
        m.d.sync += s1_sign.eq(result_sign)

        # Pipeline stage 2
        o_r = Signal(w, name="o_r")
        m.d.sync += o_r.eq(Cat(s1_log, s1_sign))
        m.d.comb += self.o.eq(o_r)
        return m
