"""Fused multiply-add: o = a * b + c with a single rounding (pipelined, 9 stages).

Based on FloPoCo's IEEEFPFMA approach, translated to Amaranth HDL.
Operates on the internal FloPoCo format:
    [exception(2) | sign(1) | exponent(we) | mantissa(wf)]
Exception encoding: 00=zero, 01=normal, 10=inf, 11=NaN.
"""
from __future__ import annotations

from math import ceil, log2

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from ..building_blocks import LeadingZeroCounter, RoundingUnit

__all__ = ["FPFMA"]


class FPFMA(PipelinedComponent):
    """Pipelined fused multiply-add: o = a * b + c (9-cycle latency).

    Parameters
    ----------
    fmt : FPFormat
        Floating-point format (defines we, wf).

    Attributes
    ----------
    a : Signal(fmt.width), in
    b : Signal(fmt.width), in
    c : Signal(fmt.width), in
    o : Signal(fmt.width), out
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.b = Signal(fmt.width, name="b")
        self.c = Signal(fmt.width, name="c")
        self.o = Signal(fmt.width, name="o")
        self.latency = 9

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we = fmt.we
        wf = fmt.wf
        bias = fmt.bias

        # ==================================================================
        # Stage 0: Unpack a, b, c
        # ==================================================================
        def unpack(sig, prefix):
            mant = Signal(wf, name=f"{prefix}_mant")
            exp = Signal(we, name=f"{prefix}_exp")
            sign = Signal(name=f"{prefix}_sign")
            exc = Signal(2, name=f"{prefix}_exc")
            m.d.comb += [
                mant.eq(sig[:wf]),
                exp.eq(sig[wf:wf + we]),
                sign.eq(sig[wf + we]),
                exc.eq(sig[wf + we + 1:]),
            ]
            return mant, exp, sign, exc

        a_mant, a_exp, a_sign, a_exc = unpack(self.a, "a")
        b_mant, b_exp, b_sign, b_exc = unpack(self.b, "b")
        c_mant, c_exp, c_sign, c_exc = unpack(self.c, "c")

        # ── Stage 0 → 1 pipeline register ──
        a_mant_r1 = Signal(wf, name="a_mant_r1")
        a_exp_r1 = Signal(we, name="a_exp_r1")
        a_sign_r1 = Signal(name="a_sign_r1")
        a_exc_r1 = Signal(2, name="a_exc_r1")
        b_mant_r1 = Signal(wf, name="b_mant_r1")
        b_exp_r1 = Signal(we, name="b_exp_r1")
        b_sign_r1 = Signal(name="b_sign_r1")
        b_exc_r1 = Signal(2, name="b_exc_r1")
        c_mant_r1 = Signal(wf, name="c_mant_r1")
        c_exp_r1 = Signal(we, name="c_exp_r1")
        c_sign_r1 = Signal(name="c_sign_r1")
        c_exc_r1 = Signal(2, name="c_exc_r1")
        m.d.sync += [
            a_mant_r1.eq(a_mant), a_exp_r1.eq(a_exp), a_sign_r1.eq(a_sign), a_exc_r1.eq(a_exc),
            b_mant_r1.eq(b_mant), b_exp_r1.eq(b_exp), b_sign_r1.eq(b_sign), b_exc_r1.eq(b_exc),
            c_mant_r1.eq(c_mant), c_exp_r1.eq(c_exp), c_sign_r1.eq(c_sign), c_exc_r1.eq(c_exc),
        ]
        for s in [a_mant_r1, a_exp_r1, a_sign_r1, a_exc_r1,
                   b_mant_r1, b_exp_r1, b_sign_r1, b_exc_r1,
                   c_mant_r1, c_exp_r1, c_sign_r1, c_exc_r1]:
            self.add_latency(s, 1)

        # ==================================================================
        # Stage 1: Exception decode (prod_exc, fma_exc)
        # ==================================================================
        prod_exc = Signal(2, name="prod_exc")
        ab_exc_sel = Signal(4, name="ab_exc_sel")
        m.d.comb += ab_exc_sel.eq(Cat(b_exc_r1, a_exc_r1))

        with m.Switch(ab_exc_sel):
            with m.Case(0b0000): m.d.comb += prod_exc.eq(0b00)
            with m.Case(0b0001): m.d.comb += prod_exc.eq(0b00)
            with m.Case(0b0100): m.d.comb += prod_exc.eq(0b00)
            with m.Case(0b0101): m.d.comb += prod_exc.eq(0b01)
            with m.Case(0b0110): m.d.comb += prod_exc.eq(0b10)
            with m.Case(0b1001): m.d.comb += prod_exc.eq(0b10)
            with m.Case(0b1010): m.d.comb += prod_exc.eq(0b10)
            with m.Case(0b0010): m.d.comb += prod_exc.eq(0b11)
            with m.Case(0b1000): m.d.comb += prod_exc.eq(0b11)
            with m.Default():    m.d.comb += prod_exc.eq(0b11)

        prod_sign = Signal(name="prod_sign")
        m.d.comb += prod_sign.eq(a_sign_r1 ^ b_sign_r1)

        fma_exc_sel = Signal(4, name="fma_exc_sel")
        m.d.comb += fma_exc_sel.eq(Cat(c_exc_r1, prod_exc))

        fma_exc = Signal(2, name="fma_exc")
        fma_sign_special = Signal(name="fma_sign_special")
        eff_sub_special = Signal(name="eff_sub_special")
        m.d.comb += eff_sub_special.eq(prod_sign ^ c_sign_r1)

        with m.Switch(fma_exc_sel):
            with m.Case(0b0000):
                m.d.comb += [fma_exc.eq(0b00), fma_sign_special.eq(prod_sign & c_sign_r1)]
            with m.Case(0b0001):
                m.d.comb += [fma_exc.eq(0b01), fma_sign_special.eq(c_sign_r1)]
            with m.Case(0b0100):
                m.d.comb += [fma_exc.eq(0b01), fma_sign_special.eq(prod_sign)]
            with m.Case(0b0101):
                m.d.comb += [fma_exc.eq(0b01), fma_sign_special.eq(prod_sign)]
            with m.Case(0b0010):
                m.d.comb += [fma_exc.eq(0b10), fma_sign_special.eq(prod_sign)]
            with m.Case(0b1000):
                m.d.comb += [fma_exc.eq(0b10), fma_sign_special.eq(c_sign_r1)]
            with m.Case(0b0110):
                m.d.comb += [fma_exc.eq(0b10), fma_sign_special.eq(prod_sign)]
            with m.Case(0b1001):
                m.d.comb += [fma_exc.eq(0b10), fma_sign_special.eq(c_sign_r1)]
            with m.Case(0b1010):
                m.d.comb += [
                    fma_exc.eq(Mux(eff_sub_special, 0b11, 0b10)),
                    fma_sign_special.eq(prod_sign),
                ]
            with m.Default():
                m.d.comb += [fma_exc.eq(0b11), fma_sign_special.eq(0)]

        # ── Stage 1 → 2 pipeline register ──
        prod_exc_r2 = Signal(2, name="prod_exc_r2")
        prod_sign_r2 = Signal(name="prod_sign_r2")
        fma_exc_r2 = Signal(2, name="fma_exc_r2")
        fma_sign_special_r2 = Signal(name="fma_sign_special_r2")
        a_mant_r2 = Signal(wf, name="a_mant_r2")
        a_exp_r2 = Signal(we, name="a_exp_r2")
        b_mant_r2 = Signal(wf, name="b_mant_r2")
        b_exp_r2 = Signal(we, name="b_exp_r2")
        c_mant_r2 = Signal(wf, name="c_mant_r2")
        c_exp_r2 = Signal(we, name="c_exp_r2")
        c_sign_r2 = Signal(name="c_sign_r2")
        c_exc_r2 = Signal(2, name="c_exc_r2")
        m.d.sync += [
            prod_exc_r2.eq(prod_exc), prod_sign_r2.eq(prod_sign),
            fma_exc_r2.eq(fma_exc), fma_sign_special_r2.eq(fma_sign_special),
            a_mant_r2.eq(a_mant_r1), a_exp_r2.eq(a_exp_r1),
            b_mant_r2.eq(b_mant_r1), b_exp_r2.eq(b_exp_r1),
            c_mant_r2.eq(c_mant_r1), c_exp_r2.eq(c_exp_r1),
            c_sign_r2.eq(c_sign_r1), c_exc_r2.eq(c_exc_r1),
        ]
        for s in [prod_exc_r2, prod_sign_r2, fma_exc_r2, fma_sign_special_r2,
                   a_mant_r2, a_exp_r2, b_mant_r2, b_exp_r2,
                   c_mant_r2, c_exp_r2, c_sign_r2, c_exc_r2]:
            self.add_latency(s, 2)

        # ==================================================================
        # Stage 2: Product mantissa multiply (sig_a * sig_b)
        # ==================================================================
        sig_a = Signal(wf + 1, name="sig_a")
        sig_b = Signal(wf + 1, name="sig_b")
        m.d.comb += [
            sig_a.eq(Cat(a_mant_r2, Const(1, 1))),
            sig_b.eq(Cat(b_mant_r2, Const(1, 1))),
        ]

        prod_width = 2 * (wf + 1)
        sig_prod = Signal(prod_width, name="sig_prod")
        m.d.comb += sig_prod.eq(sig_a * sig_b)

        prod_exp = Signal(we + 2, name="prod_exp")
        m.d.comb += prod_exp.eq(
            Cat(a_exp_r2, Const(0, 2)) + Cat(b_exp_r2, Const(0, 2)) - bias
        )

        # ── Stage 2 → 3 pipeline register ──
        sig_prod_r3 = Signal(prod_width, name="sig_prod_r3")
        prod_exp_r3 = Signal(we + 2, name="prod_exp_r3")
        prod_sign_r3 = Signal(name="prod_sign_r3")
        prod_exc_r3 = Signal(2, name="prod_exc_r3")
        fma_exc_r3 = Signal(2, name="fma_exc_r3")
        fma_sign_special_r3 = Signal(name="fma_sign_special_r3")
        c_mant_r3 = Signal(wf, name="c_mant_r3")
        c_exp_r3 = Signal(we, name="c_exp_r3")
        c_sign_r3 = Signal(name="c_sign_r3")
        c_exc_r3 = Signal(2, name="c_exc_r3")
        m.d.sync += [
            sig_prod_r3.eq(sig_prod), prod_exp_r3.eq(prod_exp),
            prod_sign_r3.eq(prod_sign_r2), prod_exc_r3.eq(prod_exc_r2),
            fma_exc_r3.eq(fma_exc_r2), fma_sign_special_r3.eq(fma_sign_special_r2),
            c_mant_r3.eq(c_mant_r2), c_exp_r3.eq(c_exp_r2),
            c_sign_r3.eq(c_sign_r2), c_exc_r3.eq(c_exc_r2),
        ]
        for s in [sig_prod_r3, prod_exp_r3, prod_sign_r3, prod_exc_r3,
                   fma_exc_r3, fma_sign_special_r3,
                   c_mant_r3, c_exp_r3, c_sign_r3, c_exc_r3]:
            self.add_latency(s, 3)

        # ==================================================================
        # Stage 3: Product normalization, exponent adjustment
        # ==================================================================
        prod_norm = Signal(name="prod_norm")
        m.d.comb += prod_norm.eq(sig_prod_r3[prod_width - 1])

        prod_exp_adj = Signal(we + 2, name="prod_exp_adj")
        m.d.comb += prod_exp_adj.eq(prod_exp_r3 + prod_norm)

        prod_frac = Signal(prod_width, name="prod_frac")
        with m.If(prod_norm):
            m.d.comb += prod_frac.eq(sig_prod_r3)
        with m.Else():
            m.d.comb += prod_frac.eq(sig_prod_r3 << 1)

        # ── Stage 3 → 4 pipeline register ──
        prod_frac_r4 = Signal(prod_width, name="prod_frac_r4")
        prod_exp_adj_r4 = Signal(we + 2, name="prod_exp_adj_r4")
        prod_sign_r4 = Signal(name="prod_sign_r4")
        prod_exc_r4 = Signal(2, name="prod_exc_r4")
        fma_exc_r4 = Signal(2, name="fma_exc_r4")
        fma_sign_special_r4 = Signal(name="fma_sign_special_r4")
        c_mant_r4 = Signal(wf, name="c_mant_r4")
        c_exp_r4 = Signal(we, name="c_exp_r4")
        c_sign_r4 = Signal(name="c_sign_r4")
        c_exc_r4 = Signal(2, name="c_exc_r4")
        m.d.sync += [
            prod_frac_r4.eq(prod_frac), prod_exp_adj_r4.eq(prod_exp_adj),
            prod_sign_r4.eq(prod_sign_r3), prod_exc_r4.eq(prod_exc_r3),
            fma_exc_r4.eq(fma_exc_r3), fma_sign_special_r4.eq(fma_sign_special_r3),
            c_mant_r4.eq(c_mant_r3), c_exp_r4.eq(c_exp_r3),
            c_sign_r4.eq(c_sign_r3), c_exc_r4.eq(c_exc_r3),
        ]
        for s in [prod_frac_r4, prod_exp_adj_r4, prod_sign_r4, prod_exc_r4,
                   fma_exc_r4, fma_sign_special_r4,
                   c_mant_r4, c_exp_r4, c_sign_r4, c_exc_r4]:
            self.add_latency(s, 4)

        # ==================================================================
        # Stage 4: Alignment of c to product
        # ==================================================================
        sig_c = Signal(wf + 1, name="sig_c")
        m.d.comb += sig_c.eq(Cat(c_mant_r4, Const(1, 1)))

        c_exp_ext = Signal(we + 2, name="c_exp_ext")
        m.d.comb += c_exp_ext.eq(Cat(c_exp_r4, Const(0, 2)))

        align_width = prod_width + wf + 4
        exp_diff = Signal(signed(we + 3), name="exp_diff")
        m.d.comb += exp_diff.eq(prod_exp_adj_r4.as_signed() - c_exp_ext.as_signed())

        prod_aligned = Signal(align_width, name="prod_aligned")
        m.d.comb += prod_aligned.eq(prod_frac_r4 << (align_width - prod_width))

        c_scaled = Signal(align_width, name="c_scaled")
        m.d.comb += c_scaled.eq(sig_c << (align_width - 1 - wf))

        c_shifted = Signal(align_width, name="c_shifted")
        c_sticky = Signal(name="c_sticky")

        abs_diff = Signal(we + 3, name="abs_diff")
        diff_neg = Signal(name="diff_neg")
        m.d.comb += diff_neg.eq(exp_diff < 0)

        with m.If(diff_neg):
            m.d.comb += abs_diff.eq(-exp_diff)
        with m.Else():
            m.d.comb += abs_diff.eq(exp_diff)

        clamped_shift = Signal(range(align_width + 1), name="clamped_shift")
        with m.If(abs_diff > align_width):
            m.d.comb += clamped_shift.eq(align_width)
        with m.Else():
            m.d.comb += clamped_shift.eq(abs_diff)

        wide_c = Signal(2 * align_width, name="wide_c")
        shifted_wide_c = Signal(2 * align_width, name="shifted_wide_c")

        # We need prod_final and prod_sticky for both branches
        prod_final = Signal(align_width, name="prod_final")
        prod_sticky = Signal(name="prod_sticky")

        with m.If(diff_neg):
            m.d.comb += [
                wide_c.eq(c_scaled << align_width),
                shifted_wide_c.eq(wide_c),
                c_shifted.eq(shifted_wide_c[align_width:]),
                c_sticky.eq(0),
            ]
            prod_shifted = Signal(align_width, name="prod_shifted_neg")
            prod_wide = Signal(2 * align_width, name="prod_wide_neg")
            prod_shifted_wide = Signal(2 * align_width, name="prod_shifted_wide_neg")
            m.d.comb += [
                prod_wide.eq(prod_aligned << align_width),
                prod_shifted_wide.eq(prod_wide >> clamped_shift),
                prod_shifted.eq(prod_shifted_wide[align_width:]),
            ]
            m.d.comb += [
                prod_final.eq(prod_shifted),
                prod_sticky.eq(prod_shifted_wide[:align_width].any()),
            ]
        with m.Else():
            m.d.comb += [
                wide_c.eq(c_scaled << align_width),
                shifted_wide_c.eq(wide_c >> clamped_shift),
                c_shifted.eq(shifted_wide_c[align_width:]),
                c_sticky.eq(shifted_wide_c[:align_width].any()),
            ]
            m.d.comb += [
                prod_final.eq(prod_aligned),
                prod_sticky.eq(0),
            ]

        # ── Stage 4 → 5 pipeline register ──
        prod_final_r5 = Signal(align_width, name="prod_final_r5")
        c_shifted_r5 = Signal(align_width, name="c_shifted_r5")
        c_sticky_r5 = Signal(name="c_sticky_r5")
        prod_sticky_r5 = Signal(name="prod_sticky_r5")
        diff_neg_r5 = Signal(name="diff_neg_r5")
        prod_sign_r5 = Signal(name="prod_sign_r5")
        c_sign_r5 = Signal(name="c_sign_r5")
        prod_exc_r5 = Signal(2, name="prod_exc_r5")
        c_exc_r5 = Signal(2, name="c_exc_r5")
        c_mant_r5 = Signal(wf, name="c_mant_r5")
        c_exp_r5_s = Signal(we, name="c_exp_r5")
        fma_exc_r5 = Signal(2, name="fma_exc_r5")
        fma_sign_special_r5 = Signal(name="fma_sign_special_r5")
        prod_exp_adj_r5 = Signal(we + 2, name="prod_exp_adj_r5")
        c_exp_ext_r5 = Signal(we + 2, name="c_exp_ext_r5")
        m.d.sync += [
            prod_final_r5.eq(prod_final), c_shifted_r5.eq(c_shifted),
            c_sticky_r5.eq(c_sticky), prod_sticky_r5.eq(prod_sticky),
            diff_neg_r5.eq(diff_neg),
            prod_sign_r5.eq(prod_sign_r4), c_sign_r5.eq(c_sign_r4),
            prod_exc_r5.eq(prod_exc_r4), c_exc_r5.eq(c_exc_r4),
            c_mant_r5.eq(c_mant_r4), c_exp_r5_s.eq(c_exp_r4),
            fma_exc_r5.eq(fma_exc_r4), fma_sign_special_r5.eq(fma_sign_special_r4),
            prod_exp_adj_r5.eq(prod_exp_adj_r4), c_exp_ext_r5.eq(c_exp_ext),
        ]
        for s in [prod_final_r5, c_shifted_r5, c_sticky_r5, prod_sticky_r5,
                   diff_neg_r5, prod_sign_r5, c_sign_r5, prod_exc_r5, c_exc_r5,
                   c_mant_r5, c_exp_r5_s, fma_exc_r5, fma_sign_special_r5,
                   prod_exp_adj_r5, c_exp_ext_r5]:
            self.add_latency(s, 5)

        # ==================================================================
        # Stage 5: Add/subtract aligned mantissas
        # ==================================================================
        eff_sub = Signal(name="eff_sub")
        m.d.comb += eff_sub.eq(prod_sign_r5 ^ c_sign_r5)

        sum_width = align_width + 1
        frac_sum = Signal(sum_width, name="frac_sum")

        result_sign = Signal(name="result_sign")

        with m.If(eff_sub):
            with m.If(diff_neg_r5):
                m.d.comb += [
                    frac_sum.eq(c_shifted_r5 - prod_final_r5),
                    result_sign.eq(c_sign_r5),
                ]
            with m.Else():
                m.d.comb += [
                    frac_sum.eq(prod_final_r5 - c_shifted_r5),
                    result_sign.eq(prod_sign_r5),
                ]
        with m.Else():
            m.d.comb += [
                frac_sum.eq(prod_final_r5 + c_shifted_r5),
                result_sign.eq(prod_sign_r5),
            ]

        all_sticky = Signal(name="all_sticky")
        m.d.comb += all_sticky.eq(c_sticky_r5 | prod_sticky_r5)

        # ── Stage 5 → 6 pipeline register ──
        frac_sum_r6 = Signal(sum_width, name="frac_sum_r6")
        all_sticky_r6 = Signal(name="all_sticky_r6")
        result_sign_r6 = Signal(name="result_sign_r6")
        prod_sign_r6 = Signal(name="prod_sign_r6")
        c_sign_r6 = Signal(name="c_sign_r6")
        prod_exc_r6 = Signal(2, name="prod_exc_r6")
        c_exc_r6 = Signal(2, name="c_exc_r6")
        c_mant_r6 = Signal(wf, name="c_mant_r6")
        c_exp_r6 = Signal(we, name="c_exp_r6")
        fma_exc_r6 = Signal(2, name="fma_exc_r6")
        fma_sign_special_r6 = Signal(name="fma_sign_special_r6")
        diff_neg_r6 = Signal(name="diff_neg_r6")
        prod_exp_adj_r6 = Signal(we + 2, name="prod_exp_adj_r6")
        c_exp_ext_r6 = Signal(we + 2, name="c_exp_ext_r6")
        m.d.sync += [
            frac_sum_r6.eq(frac_sum), all_sticky_r6.eq(all_sticky),
            result_sign_r6.eq(result_sign),
            prod_sign_r6.eq(prod_sign_r5), c_sign_r6.eq(c_sign_r5),
            prod_exc_r6.eq(prod_exc_r5), c_exc_r6.eq(c_exc_r5),
            c_mant_r6.eq(c_mant_r5), c_exp_r6.eq(c_exp_r5_s),
            fma_exc_r6.eq(fma_exc_r5), fma_sign_special_r6.eq(fma_sign_special_r5),
            diff_neg_r6.eq(diff_neg_r5),
            prod_exp_adj_r6.eq(prod_exp_adj_r5), c_exp_ext_r6.eq(c_exp_ext_r5),
        ]
        for s in [frac_sum_r6, all_sticky_r6, result_sign_r6,
                   prod_sign_r6, c_sign_r6, prod_exc_r6, c_exc_r6,
                   c_mant_r6, c_exp_r6, fma_exc_r6, fma_sign_special_r6,
                   diff_neg_r6, prod_exp_adj_r6, c_exp_ext_r6]:
            self.add_latency(s, 6)

        # ==================================================================
        # Stage 6: LZC + normalization
        # ==================================================================
        norm_width = sum_width + 1
        norm_input = Signal(norm_width, name="norm_input")
        m.d.comb += norm_input.eq(Cat(all_sticky_r6, frac_sum_r6))

        lzc = LeadingZeroCounter(norm_width)
        m.submodules.lzc = lzc
        m.d.comb += lzc.i.eq(norm_input)

        shifted_frac = Signal(norm_width, name="shifted_frac")
        m.d.comb += shifted_frac.eq(norm_input << lzc.count)

        result_exp_base = Signal(we + 2, name="result_exp_base")
        with m.If(diff_neg_r6):
            m.d.comb += result_exp_base.eq(c_exp_ext_r6 + 1)
        with m.Else():
            m.d.comb += result_exp_base.eq(prod_exp_adj_r6 + 1)

        result_exp = Signal(we + 2, name="result_exp")
        m.d.comb += result_exp.eq(result_exp_base - lzc.count)

        is_zero_result = Signal(name="is_zero_result")
        m.d.comb += is_zero_result.eq(lzc.all_zeros)

        # ── Stage 6 → 7 pipeline register ──
        shifted_frac_r7 = Signal(norm_width, name="shifted_frac_r7")
        result_exp_r7 = Signal(we + 2, name="result_exp_r7")
        is_zero_result_r7 = Signal(name="is_zero_result_r7")
        result_sign_r7 = Signal(name="result_sign_r7")
        prod_sign_r7 = Signal(name="prod_sign_r7")
        c_sign_r7 = Signal(name="c_sign_r7")
        prod_exc_r7 = Signal(2, name="prod_exc_r7")
        c_exc_r7 = Signal(2, name="c_exc_r7")
        c_mant_r7 = Signal(wf, name="c_mant_r7")
        c_exp_r7 = Signal(we, name="c_exp_r7")
        fma_exc_r7 = Signal(2, name="fma_exc_r7")
        fma_sign_special_r7 = Signal(name="fma_sign_special_r7")
        m.d.sync += [
            shifted_frac_r7.eq(shifted_frac), result_exp_r7.eq(result_exp),
            is_zero_result_r7.eq(is_zero_result),
            result_sign_r7.eq(result_sign_r6),
            prod_sign_r7.eq(prod_sign_r6), c_sign_r7.eq(c_sign_r6),
            prod_exc_r7.eq(prod_exc_r6), c_exc_r7.eq(c_exc_r6),
            c_mant_r7.eq(c_mant_r6), c_exp_r7.eq(c_exp_r6),
            fma_exc_r7.eq(fma_exc_r6), fma_sign_special_r7.eq(fma_sign_special_r6),
        ]
        for s in [shifted_frac_r7, result_exp_r7, is_zero_result_r7,
                   result_sign_r7, prod_sign_r7, c_sign_r7,
                   prod_exc_r7, c_exc_r7, c_mant_r7, c_exp_r7,
                   fma_exc_r7, fma_sign_special_r7]:
            self.add_latency(s, 7)

        # ==================================================================
        # Stage 7: Rounding, exponent adjustment
        # ==================================================================
        round_in = Signal(wf + 3, name="round_in")
        r_mantissa = Signal(wf, name="r_mantissa")
        r_guard = Signal(name="r_guard")
        r_round = Signal(name="r_round")
        r_sticky = Signal(name="r_sticky")

        m.d.comb += [
            r_mantissa.eq(shifted_frac_r7[norm_width - 1 - wf:norm_width - 1]),
            r_guard.eq(shifted_frac_r7[norm_width - 1 - wf - 1] if norm_width - 1 - wf - 1 >= 0 else 0),
            r_round.eq(shifted_frac_r7[norm_width - 1 - wf - 2] if norm_width - 1 - wf - 2 >= 0 else 0),
        ]
        sticky_end = norm_width - 1 - wf - 3
        if sticky_end >= 0:
            m.d.comb += r_sticky.eq(shifted_frac_r7[:sticky_end + 1].any())
        else:
            m.d.comb += r_sticky.eq(0)

        m.d.comb += round_in.eq(Cat(r_sticky, r_round, r_guard, r_mantissa))

        rounder = RoundingUnit(wf)
        m.submodules.rounder = rounder
        m.d.comb += rounder.mantissa_in.eq(round_in)

        rounded_mant = rounder.mantissa_out
        round_overflow = rounder.overflow

        final_exp = Signal(we + 2, name="final_exp")
        m.d.comb += final_exp.eq(result_exp_r7 + round_overflow)

        # ── Stage 7 → 8 pipeline register ──
        rounded_mant_r8 = Signal(wf, name="rounded_mant_r8")
        final_exp_r8 = Signal(we + 2, name="final_exp_r8")
        is_zero_result_r8 = Signal(name="is_zero_result_r8")
        result_sign_r8 = Signal(name="result_sign_r8")
        prod_sign_r8 = Signal(name="prod_sign_r8")
        c_sign_r8 = Signal(name="c_sign_r8")
        prod_exc_r8 = Signal(2, name="prod_exc_r8")
        c_exc_r8 = Signal(2, name="c_exc_r8")
        c_mant_r8 = Signal(wf, name="c_mant_r8")
        c_exp_r8 = Signal(we, name="c_exp_r8")
        fma_exc_r8 = Signal(2, name="fma_exc_r8")
        fma_sign_special_r8 = Signal(name="fma_sign_special_r8")
        m.d.sync += [
            rounded_mant_r8.eq(rounded_mant), final_exp_r8.eq(final_exp),
            is_zero_result_r8.eq(is_zero_result_r7),
            result_sign_r8.eq(result_sign_r7),
            prod_sign_r8.eq(prod_sign_r7), c_sign_r8.eq(c_sign_r7),
            prod_exc_r8.eq(prod_exc_r7), c_exc_r8.eq(c_exc_r7),
            c_mant_r8.eq(c_mant_r7), c_exp_r8.eq(c_exp_r7),
            fma_exc_r8.eq(fma_exc_r7), fma_sign_special_r8.eq(fma_sign_special_r7),
        ]
        for s in [rounded_mant_r8, final_exp_r8, is_zero_result_r8,
                   result_sign_r8, prod_sign_r8, c_sign_r8,
                   prod_exc_r8, c_exc_r8, c_mant_r8, c_exp_r8,
                   fma_exc_r8, fma_sign_special_r8]:
            self.add_latency(s, 8)

        # ==================================================================
        # Stage 8: Overflow/underflow, final mux, pack output
        # ==================================================================
        exp_top = Signal(2, name="exp_top")
        m.d.comb += exp_top.eq(final_exp_r8[we:we + 2])

        computed_exc = Signal(2, name="computed_exc")

        with m.If(is_zero_result_r8):
            m.d.comb += computed_exc.eq(0b00)
        with m.Else():
            with m.Switch(exp_top):
                with m.Case(0b00): m.d.comb += computed_exc.eq(0b01)
                with m.Case(0b01): m.d.comb += computed_exc.eq(0b10)
                with m.Case(0b10): m.d.comb += computed_exc.eq(0b00)
                with m.Case(0b11): m.d.comb += computed_exc.eq(0b00)

        final_exc = Signal(2, name="final_exc")
        final_sign = Signal(name="final_sign")
        final_mant = Signal(wf, name="final_mant")
        final_e = Signal(we, name="final_e")

        both_inputs_computable = Signal(name="both_inputs_computable")
        m.d.comb += both_inputs_computable.eq(
            (prod_exc_r8 == 0b01) & (c_exc_r8 == 0b01)
        )

        prod_normal_c_zero = Signal(name="prod_normal_c_zero")
        prod_zero_c_normal = Signal(name="prod_zero_c_normal")
        m.d.comb += [
            prod_normal_c_zero.eq((prod_exc_r8 == 0b01) & (c_exc_r8 == 0b00)),
            prod_zero_c_normal.eq((prod_exc_r8 == 0b00) & (c_exc_r8 == 0b01)),
        ]

        with m.If(both_inputs_computable):
            m.d.comb += [
                final_exc.eq(computed_exc),
                final_sign.eq(Mux(is_zero_result_r8, 0, result_sign_r8)),
                final_mant.eq(rounded_mant_r8),
                final_e.eq(final_exp_r8[:we]),
            ]
        with m.Elif(prod_normal_c_zero):
            m.d.comb += [
                final_exc.eq(computed_exc),
                final_sign.eq(prod_sign_r8),
                final_mant.eq(rounded_mant_r8),
                final_e.eq(final_exp_r8[:we]),
            ]
        with m.Elif(prod_zero_c_normal):
            m.d.comb += [
                final_exc.eq(0b01),
                final_sign.eq(c_sign_r8),
                final_mant.eq(c_mant_r8),
                final_e.eq(c_exp_r8),
            ]
        with m.Else():
            m.d.comb += [
                final_exc.eq(fma_exc_r8),
                final_sign.eq(fma_sign_special_r8),
                final_mant.eq(0),
                final_e.eq(0),
            ]

        # ── Stage 8 → 9 pipeline register (output) ──
        o_r9 = Signal(fmt.width, name="o_r9")
        m.d.sync += o_r9.eq(Cat(final_mant, final_e, final_sign, final_exc))
        m.d.comb += self.o.eq(o_r9)

        return m
