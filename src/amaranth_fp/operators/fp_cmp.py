"""Floating-point comparator (pipelined, 2 stages).

Operates on the internal FloPoCo format:
    [exception(2) | sign(1) | exponent(we) | mantissa(wf)]
Exception encoding: 00=zero, 01=normal, 10=inf, 11=NaN.
"""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent

__all__ = ["FPComparator"]


class FPComparator(PipelinedComponent):
    """Pipelined floating-point comparator (2-cycle latency).

    Parameters
    ----------
    fmt : FPFormat
        Floating-point format (defines we, wf).

    Attributes
    ----------
    a : Signal(fmt.width), in
    b : Signal(fmt.width), in
    lt : Signal(1), out
    eq : Signal(1), out
    gt : Signal(1), out
    unordered : Signal(1), out
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.b = Signal(fmt.width, name="b")
        self.lt = Signal(name="lt")
        self.eq = Signal(name="eq")
        self.gt = Signal(name="gt")
        self.unordered = Signal(name="unordered")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we = fmt.we
        wf = fmt.wf

        # ==================================================================
        # Stage 0: unpack, NaN detection, sign comparison
        # ==================================================================
        a_mant = Signal(wf, name="a_mant")
        a_exp = Signal(we, name="a_exp")
        a_sign = Signal(name="a_sign")
        a_exc = Signal(2, name="a_exc")

        b_mant = Signal(wf, name="b_mant")
        b_exp = Signal(we, name="b_exp")
        b_sign = Signal(name="b_sign")
        b_exc = Signal(2, name="b_exc")

        m.d.comb += [
            a_mant.eq(self.a[:wf]),
            a_exp.eq(self.a[wf:wf + we]),
            a_sign.eq(self.a[wf + we]),
            a_exc.eq(self.a[wf + we + 1:]),

            b_mant.eq(self.b[:wf]),
            b_exp.eq(self.b[wf:wf + we]),
            b_sign.eq(self.b[wf + we]),
            b_exc.eq(self.b[wf + we + 1:]),
        ]

        a_is_nan = Signal(name="a_is_nan")
        b_is_nan = Signal(name="b_is_nan")
        m.d.comb += [
            a_is_nan.eq(a_exc == 0b11),
            b_is_nan.eq(b_exc == 0b11),
        ]
        either_nan = Signal(name="either_nan")
        m.d.comb += either_nan.eq(a_is_nan | b_is_nan)

        a_is_zero = Signal(name="a_is_zero")
        b_is_zero = Signal(name="b_is_zero")
        m.d.comb += [
            a_is_zero.eq(a_exc == 0b00),
            b_is_zero.eq(b_exc == 0b00),
        ]
        both_zero = Signal(name="both_zero")
        m.d.comb += both_zero.eq(a_is_zero & b_is_zero)

        # ── Stage 0 → 1 pipeline register ──
        either_nan_r1 = Signal(name="either_nan_r1")
        both_zero_r1 = Signal(name="both_zero_r1")
        a_is_zero_r1 = Signal(name="a_is_zero_r1")
        b_is_zero_r1 = Signal(name="b_is_zero_r1")
        a_sign_r1 = Signal(name="a_sign_r1")
        b_sign_r1 = Signal(name="b_sign_r1")
        a_exp_r1 = Signal(we, name="a_exp_r1")
        b_exp_r1 = Signal(we, name="b_exp_r1")
        a_mant_r1 = Signal(wf, name="a_mant_r1")
        b_mant_r1 = Signal(wf, name="b_mant_r1")
        m.d.sync += [
            either_nan_r1.eq(either_nan),
            both_zero_r1.eq(both_zero),
            a_is_zero_r1.eq(a_is_zero),
            b_is_zero_r1.eq(b_is_zero),
            a_sign_r1.eq(a_sign),
            b_sign_r1.eq(b_sign),
            a_exp_r1.eq(a_exp),
            b_exp_r1.eq(b_exp),
            a_mant_r1.eq(a_mant),
            b_mant_r1.eq(b_mant),
        ]
        self.add_latency(either_nan_r1, 1)
        self.add_latency(both_zero_r1, 1)
        self.add_latency(a_is_zero_r1, 1)
        self.add_latency(b_is_zero_r1, 1)
        self.add_latency(a_sign_r1, 1)
        self.add_latency(b_sign_r1, 1)
        self.add_latency(a_exp_r1, 1)
        self.add_latency(b_exp_r1, 1)
        self.add_latency(a_mant_r1, 1)
        self.add_latency(b_mant_r1, 1)

        # ==================================================================
        # Stage 1: magnitude comparison, final output mux
        # ==================================================================
        exp_a_gt_b = Signal(name="exp_a_gt_b")
        exp_a_lt_b = Signal(name="exp_a_lt_b")
        exp_eq = Signal(name="exp_eq")
        m.d.comb += [
            exp_a_gt_b.eq(a_exp_r1 > b_exp_r1),
            exp_a_lt_b.eq(a_exp_r1 < b_exp_r1),
            exp_eq.eq(a_exp_r1 == b_exp_r1),
        ]

        mant_a_gt_b = Signal(name="mant_a_gt_b")
        mant_a_lt_b = Signal(name="mant_a_lt_b")
        mant_eq = Signal(name="mant_eq")
        m.d.comb += [
            mant_a_gt_b.eq(a_mant_r1 > b_mant_r1),
            mant_a_lt_b.eq(a_mant_r1 < b_mant_r1),
            mant_eq.eq(a_mant_r1 == b_mant_r1),
        ]

        mag_a_gt_b = Signal(name="mag_a_gt_b")
        m.d.comb += mag_a_gt_b.eq(exp_a_gt_b | (exp_eq & mant_a_gt_b))

        mag_a_lt_b = Signal(name="mag_a_lt_b")
        m.d.comb += mag_a_lt_b.eq(exp_a_lt_b | (exp_eq & mant_a_lt_b))

        mag_eq = Signal(name="mag_eq")
        m.d.comb += mag_eq.eq(exp_eq & mant_eq)

        r_lt = Signal(name="r_lt")
        r_eq = Signal(name="r_eq")
        r_gt = Signal(name="r_gt")
        r_unord = Signal(name="r_unord")

        m.d.comb += r_unord.eq(either_nan_r1)

        with m.If(either_nan_r1):
            m.d.comb += [r_lt.eq(0), r_eq.eq(0), r_gt.eq(0)]
        with m.Elif(both_zero_r1):
            m.d.comb += [r_lt.eq(0), r_eq.eq(1), r_gt.eq(0)]
        with m.Elif(a_is_zero_r1):
            with m.If(b_sign_r1):
                m.d.comb += [r_lt.eq(0), r_eq.eq(0), r_gt.eq(1)]
            with m.Else():
                m.d.comb += [r_lt.eq(1), r_eq.eq(0), r_gt.eq(0)]
        with m.Elif(b_is_zero_r1):
            with m.If(a_sign_r1):
                m.d.comb += [r_lt.eq(1), r_eq.eq(0), r_gt.eq(0)]
            with m.Else():
                m.d.comb += [r_lt.eq(0), r_eq.eq(0), r_gt.eq(1)]
        with m.Elif(a_sign_r1 != b_sign_r1):
            with m.If(a_sign_r1):
                m.d.comb += [r_lt.eq(1), r_eq.eq(0), r_gt.eq(0)]
            with m.Else():
                m.d.comb += [r_lt.eq(0), r_eq.eq(0), r_gt.eq(1)]
        with m.Else():
            with m.If(~a_sign_r1):
                with m.If(mag_a_gt_b):
                    m.d.comb += [r_lt.eq(0), r_eq.eq(0), r_gt.eq(1)]
                with m.Elif(mag_a_lt_b):
                    m.d.comb += [r_lt.eq(1), r_eq.eq(0), r_gt.eq(0)]
                with m.Else():
                    m.d.comb += [r_lt.eq(0), r_eq.eq(1), r_gt.eq(0)]
            with m.Else():
                with m.If(mag_a_gt_b):
                    m.d.comb += [r_lt.eq(1), r_eq.eq(0), r_gt.eq(0)]
                with m.Elif(mag_a_lt_b):
                    m.d.comb += [r_lt.eq(0), r_eq.eq(0), r_gt.eq(1)]
                with m.Else():
                    m.d.comb += [r_lt.eq(0), r_eq.eq(1), r_gt.eq(0)]

        # ── Stage 1 → 2 pipeline register (output) ──
        lt_r2 = Signal(name="lt_r2")
        eq_r2 = Signal(name="eq_r2")
        gt_r2 = Signal(name="gt_r2")
        unord_r2 = Signal(name="unord_r2")
        m.d.sync += [
            lt_r2.eq(r_lt),
            eq_r2.eq(r_eq),
            gt_r2.eq(r_gt),
            unord_r2.eq(r_unord),
        ]
        m.d.comb += [
            self.lt.eq(lt_r2),
            self.eq.eq(eq_r2),
            self.gt.eq(gt_r2),
            self.unordered.eq(unord_r2),
        ]

        return m
