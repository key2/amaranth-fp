"""Floating-point min and max operators (pipelined, 3 stages).

Uses FPComparator for comparison. NaN propagation: if either input is NaN,
result is NaN.
Operates on the internal FloPoCo format:
    [exception(2) | sign(1) | exponent(we) | mantissa(wf)]
"""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from .fp_cmp import FPComparator

__all__ = ["FPMin", "FPMax"]


class FPMin(PipelinedComponent):
    """Pipelined floating-point minimum: o = min(a, b) (3-cycle latency).

    If either operand is NaN, output is NaN.

    Parameters
    ----------
    fmt : FPFormat
        Floating-point format (defines we, wf).
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.b = Signal(fmt.width, name="b")
        self.o = Signal(fmt.width, name="o")
        self.latency = 3

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we = fmt.we
        wf = fmt.wf

        cmp = FPComparator(fmt)
        m.submodules.cmp = cmp
        m.d.comb += [
            cmp.a.eq(self.a),
            cmp.b.eq(self.b),
        ]

        # Delay a and b by cmp.latency (2) cycles to align with comparator output
        a_d1 = Signal(fmt.width, name="a_d1")
        b_d1 = Signal(fmt.width, name="b_d1")
        m.d.sync += [a_d1.eq(self.a), b_d1.eq(self.b)]

        a_d2 = Signal(fmt.width, name="a_d2")
        b_d2 = Signal(fmt.width, name="b_d2")
        m.d.sync += [a_d2.eq(a_d1), b_d2.eq(b_d1)]

        # Now cmp outputs and a_d2/b_d2 are aligned at cycle 2
        nan_val = (0b11 << (wf + we + 1))

        mux_out = Signal(fmt.width, name="mux_out")
        with m.If(cmp.unordered):
            m.d.comb += mux_out.eq(nan_val)
        with m.Elif(cmp.lt | cmp.eq):
            m.d.comb += mux_out.eq(a_d2)
        with m.Else():
            m.d.comb += mux_out.eq(b_d2)

        # ── Stage 2 → 3 pipeline register (output) ──
        o_r3 = Signal(fmt.width, name="o_r3")
        m.d.sync += o_r3.eq(mux_out)
        m.d.comb += self.o.eq(o_r3)

        return m


class FPMax(PipelinedComponent):
    """Pipelined floating-point maximum: o = max(a, b) (3-cycle latency).

    If either operand is NaN, output is NaN.

    Parameters
    ----------
    fmt : FPFormat
        Floating-point format (defines we, wf).
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.b = Signal(fmt.width, name="b")
        self.o = Signal(fmt.width, name="o")
        self.latency = 3

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we = fmt.we
        wf = fmt.wf

        cmp = FPComparator(fmt)
        m.submodules.cmp = cmp
        m.d.comb += [
            cmp.a.eq(self.a),
            cmp.b.eq(self.b),
        ]

        # Delay a and b by cmp.latency (2) cycles to align with comparator output
        a_d1 = Signal(fmt.width, name="a_d1")
        b_d1 = Signal(fmt.width, name="b_d1")
        m.d.sync += [a_d1.eq(self.a), b_d1.eq(self.b)]

        a_d2 = Signal(fmt.width, name="a_d2")
        b_d2 = Signal(fmt.width, name="b_d2")
        m.d.sync += [a_d2.eq(a_d1), b_d2.eq(b_d1)]

        nan_val = (0b11 << (wf + we + 1))

        mux_out = Signal(fmt.width, name="mux_out")
        with m.If(cmp.unordered):
            m.d.comb += mux_out.eq(nan_val)
        with m.Elif(cmp.gt | cmp.eq):
            m.d.comb += mux_out.eq(a_d2)
        with m.Else():
            m.d.comb += mux_out.eq(b_d2)

        # ── Stage 2 → 3 pipeline register (output) ──
        o_r3 = Signal(fmt.width, name="o_r3")
        m.d.sync += o_r3.eq(mux_out)
        m.d.comb += self.o.eq(o_r3)

        return m
