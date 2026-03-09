"""Integer constant division using multiply-by-reciprocal (pipelined, 2 stages)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["IntConstDiv"]


class IntConstDiv(PipelinedComponent):
    """Integer constant divider (2-cycle latency).

    Uses multiply-by-reciprocal approach.

    Parameters
    ----------
    width : int
    divisor : int
    """

    def __init__(self, width: int, divisor: int) -> None:
        super().__init__()
        if divisor == 0:
            raise ValueError("divisor must be non-zero")
        self.width = width
        self.divisor = divisor
        self.a = Signal(width, name="a")
        self.q = Signal(width, name="q")  # quotient
        self.r = Signal(width, name="r")  # remainder
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        d = self.divisor

        # Compute magic number for division
        # reciprocal * 2^(w+extra) / d, then shift right
        extra = w
        recip = ((1 << (w + extra)) + d - 1) // d

        prod = Signal(2 * w + extra, name="prod")
        m.d.comb += prod.eq(self.a * recip)

        q_approx = Signal(w, name="q_approx")
        m.d.comb += q_approx.eq(prod[w + extra:])

        # Pipeline stage 1
        q_r1 = Signal(w, name="q_r1")
        a_r1 = Signal(w, name="a_r1")
        m.d.sync += [q_r1.eq(q_approx), a_r1.eq(self.a)]

        # Stage 2: compute remainder and correct
        rem = Signal(w, name="rem")
        m.d.comb += rem.eq(a_r1 - q_r1 * d)

        q_r2 = Signal(w, name="q_r2")
        r_r2 = Signal(w, name="r_r2")
        m.d.sync += [q_r2.eq(q_r1), r_r2.eq(rem)]
        m.d.comb += [self.q.eq(q_r2), self.r.eq(r_r2)]

        return m
