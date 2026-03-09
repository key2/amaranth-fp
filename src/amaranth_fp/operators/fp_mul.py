"""Floating-point multiplier (pipelined, 5 stages).

Based on FloPoCo's FPMult algorithm, translated to Amaranth HDL.
Operates on the internal FloPoCo format:
    [exception(2) | sign(1) | exponent(we) | mantissa(wf)]
Exception encoding: 00=zero, 01=normal, 10=inf, 11=NaN.
"""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from ..building_blocks import RoundingUnit

__all__ = ["FPMul"]


class FPMul(PipelinedComponent):
    """Pipelined floating-point multiplier (5-cycle latency).

    Parameters
    ----------
    fmt : FPFormat
        Floating-point format (defines we, wf).

    Attributes
    ----------
    a : Signal(fmt.width), in
    b : Signal(fmt.width), in
    o : Signal(fmt.width), out
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.b = Signal(fmt.width, name="b")
        self.o = Signal(fmt.width, name="o")
        self.latency = 5

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we = fmt.we
        wf = fmt.wf

        # ==================================================================
        # Stage 0: unpack inputs, compute result sign
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

        result_sign = Signal(name="result_sign")
        m.d.comb += result_sign.eq(a_sign ^ b_sign)

        # ── Stage 0 → 1 pipeline register ──
        a_mant_r1 = Signal(wf, name="a_mant_r1")
        a_exp_r1 = Signal(we, name="a_exp_r1")
        a_exc_r1 = Signal(2, name="a_exc_r1")
        b_mant_r1 = Signal(wf, name="b_mant_r1")
        b_exp_r1 = Signal(we, name="b_exp_r1")
        b_exc_r1 = Signal(2, name="b_exc_r1")
        result_sign_r1 = Signal(name="result_sign_r1")
        m.d.sync += [
            a_mant_r1.eq(a_mant), a_exp_r1.eq(a_exp), a_exc_r1.eq(a_exc),
            b_mant_r1.eq(b_mant), b_exp_r1.eq(b_exp), b_exc_r1.eq(b_exc),
            result_sign_r1.eq(result_sign),
        ]
        for s in [a_mant_r1, a_exp_r1, a_exc_r1, b_mant_r1, b_exp_r1, b_exc_r1, result_sign_r1]:
            self.add_latency(s, 1)

        # ==================================================================
        # Stage 1: exception decode, exponent addition
        # ==================================================================
        exc_sel = Signal(4, name="exc_sel")
        m.d.comb += exc_sel.eq(Cat(b_exc_r1, a_exc_r1))

        exc_result = Signal(2, name="exc_result")
        with m.Switch(exc_sel):
            with m.Case(0b0000): m.d.comb += exc_result.eq(0b00)
            with m.Case(0b0001): m.d.comb += exc_result.eq(0b00)
            with m.Case(0b0100): m.d.comb += exc_result.eq(0b00)
            with m.Case(0b0101): m.d.comb += exc_result.eq(0b01)
            with m.Case(0b0110): m.d.comb += exc_result.eq(0b10)
            with m.Case(0b1001): m.d.comb += exc_result.eq(0b10)
            with m.Case(0b1010): m.d.comb += exc_result.eq(0b10)
            with m.Case(0b0010): m.d.comb += exc_result.eq(0b11)
            with m.Case(0b1000): m.d.comb += exc_result.eq(0b11)
            with m.Default():    m.d.comb += exc_result.eq(0b11)

        bias = fmt.bias
        exp_sum_pre = Signal(we + 2, name="exp_sum_pre")
        m.d.comb += exp_sum_pre.eq(
            Cat(a_exp_r1, Const(0, 2)) + Cat(b_exp_r1, Const(0, 2))
        )
        exp_sum = Signal(we + 2, name="exp_sum")
        m.d.comb += exp_sum.eq(exp_sum_pre - bias)

        sig_a = Signal(wf + 1, name="sig_a")
        sig_b = Signal(wf + 1, name="sig_b")
        m.d.comb += [
            sig_a.eq(Cat(a_mant_r1, Const(1, 1))),
            sig_b.eq(Cat(b_mant_r1, Const(1, 1))),
        ]

        # ── Stage 1 → 2 pipeline register ──
        exc_result_r2 = Signal(2, name="exc_result_r2")
        exp_sum_r2 = Signal(we + 2, name="exp_sum_r2")
        sig_a_r2 = Signal(wf + 1, name="sig_a_r2")
        sig_b_r2 = Signal(wf + 1, name="sig_b_r2")
        result_sign_r2 = Signal(name="result_sign_r2")
        m.d.sync += [
            exc_result_r2.eq(exc_result),
            exp_sum_r2.eq(exp_sum),
            sig_a_r2.eq(sig_a),
            sig_b_r2.eq(sig_b),
            result_sign_r2.eq(result_sign_r1),
        ]
        for s in [exc_result_r2, exp_sum_r2, sig_a_r2, sig_b_r2, result_sign_r2]:
            self.add_latency(s, 2)

        # ==================================================================
        # Stage 2: mantissa multiplication
        # ==================================================================
        prod_width = 2 * (wf + 1)
        sig_prod = Signal(prod_width, name="sig_prod")
        m.d.comb += sig_prod.eq(sig_a_r2 * sig_b_r2)

        # ── Stage 2 → 3 pipeline register ──
        sig_prod_r3 = Signal(prod_width, name="sig_prod_r3")
        exp_sum_r3 = Signal(we + 2, name="exp_sum_r3")
        exc_result_r3 = Signal(2, name="exc_result_r3")
        result_sign_r3 = Signal(name="result_sign_r3")
        m.d.sync += [
            sig_prod_r3.eq(sig_prod),
            exp_sum_r3.eq(exp_sum_r2),
            exc_result_r3.eq(exc_result_r2),
            result_sign_r3.eq(result_sign_r2),
        ]
        for s in [sig_prod_r3, exp_sum_r3, exc_result_r3, result_sign_r3]:
            self.add_latency(s, 3)

        # ==================================================================
        # Stage 3: normalization, G/R/S extraction, rounding
        # ==================================================================
        norm = Signal(name="norm")
        m.d.comb += norm.eq(sig_prod_r3[prod_width - 1])

        exp_post_norm = Signal(we + 2, name="exp_post_norm")
        m.d.comb += exp_post_norm.eq(exp_sum_r3 + norm)

        sig_prod_ext = Signal(prod_width, name="sig_prod_ext")
        with m.If(norm):
            m.d.comb += sig_prod_ext.eq(sig_prod_r3)
        with m.Else():
            m.d.comb += sig_prod_ext.eq(sig_prod_r3 << 1)

        round_mantissa = Signal(wf, name="round_mantissa")
        guard_bit = Signal(name="guard_bit")
        round_bit = Signal(name="round_bit")
        sticky_val = Signal(name="sticky_val")

        m.d.comb += [
            round_mantissa.eq(sig_prod_ext[prod_width - 1 - wf:prod_width - 1]),
            guard_bit.eq(sig_prod_ext[prod_width - 1 - wf - 1]),
            round_bit.eq(sig_prod_ext[prod_width - 1 - wf - 2]),
        ]

        sticky_start = prod_width - 1 - wf - 3
        if sticky_start >= 0:
            m.d.comb += sticky_val.eq(sig_prod_ext[:sticky_start + 1].any())
        else:
            m.d.comb += sticky_val.eq(0)

        round_in = Signal(wf + 3, name="round_in")
        m.d.comb += round_in.eq(Cat(sticky_val, round_bit, guard_bit, round_mantissa))

        rounder = RoundingUnit(wf)
        m.submodules.rounder = rounder
        m.d.comb += rounder.mantissa_in.eq(round_in)

        rounded_mant = rounder.mantissa_out
        round_overflow = rounder.overflow

        final_exp = Signal(we + 2, name="final_exp")
        m.d.comb += final_exp.eq(exp_post_norm + round_overflow)

        # ── Stage 3 → 4 pipeline register ──
        rounded_mant_r4 = Signal(wf, name="rounded_mant_r4")
        final_exp_r4 = Signal(we + 2, name="final_exp_r4")
        exc_result_r4 = Signal(2, name="exc_result_r4")
        result_sign_r4 = Signal(name="result_sign_r4")
        m.d.sync += [
            rounded_mant_r4.eq(rounded_mant),
            final_exp_r4.eq(final_exp),
            exc_result_r4.eq(exc_result_r3),
            result_sign_r4.eq(result_sign_r3),
        ]
        for s in [rounded_mant_r4, final_exp_r4, exc_result_r4, result_sign_r4]:
            self.add_latency(s, 4)

        # ==================================================================
        # Stage 4: overflow/underflow, final mux, pack
        # ==================================================================
        exp_top = Signal(2, name="exp_top")
        m.d.comb += exp_top.eq(final_exp_r4[we:we + 2])

        exc_post_norm = Signal(2, name="exc_post_norm")
        with m.Switch(exp_top):
            with m.Case(0b00): m.d.comb += exc_post_norm.eq(0b01)
            with m.Case(0b01): m.d.comb += exc_post_norm.eq(0b10)
            with m.Case(0b10): m.d.comb += exc_post_norm.eq(0b00)
            with m.Case(0b11): m.d.comb += exc_post_norm.eq(0b00)

        final_exc = Signal(2, name="final_exc")
        with m.If((exc_result_r4 == 0b11) | (exc_result_r4 == 0b10) | (exc_result_r4 == 0b00)):
            m.d.comb += final_exc.eq(exc_result_r4)
        with m.Else():
            m.d.comb += final_exc.eq(exc_post_norm)

        final_mant = Signal(wf, name="final_mant")
        final_e = Signal(we, name="final_e")

        with m.If(final_exc == 0b01):
            m.d.comb += [
                final_mant.eq(rounded_mant_r4),
                final_e.eq(final_exp_r4[:we]),
            ]
        with m.Else():
            m.d.comb += [
                final_mant.eq(0),
                final_e.eq(0),
            ]

        # ── Stage 4 → 5 pipeline register (output) ──
        o_r5 = Signal(fmt.width, name="o_r5")
        m.d.sync += o_r5.eq(Cat(final_mant, final_e, result_sign_r4, final_exc))
        m.d.comb += self.o.eq(o_r5)

        return m
