"""Karatsuba-optimized FP multiplier (pipelined, 6 stages).

Uses 3 half-width multiplies instead of 1 full-width multiply.
"""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from ..building_blocks import RoundingUnit

__all__ = ["FPMultKaratsuba"]


class FPMultKaratsuba(PipelinedComponent):
    """Karatsuba FP multiplier (6-cycle latency).

    Parameters
    ----------
    fmt : FPFormat
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.b = Signal(fmt.width, name="b")
        self.o = Signal(fmt.width, name="o")
        self.latency = 6

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we = fmt.we
        wf = fmt.wf

        # Stage 0: Unpack
        a_mant = Signal(wf); a_exp = Signal(we); a_sign = Signal(); a_exc = Signal(2)
        b_mant = Signal(wf); b_exp = Signal(we); b_sign = Signal(); b_exc = Signal(2)
        m.d.comb += [
            a_mant.eq(self.a[:wf]), a_exp.eq(self.a[wf:wf+we]),
            a_sign.eq(self.a[wf+we]), a_exc.eq(self.a[wf+we+1:]),
            b_mant.eq(self.b[:wf]), b_exp.eq(self.b[wf:wf+we]),
            b_sign.eq(self.b[wf+we]), b_exc.eq(self.b[wf+we+1:]),
        ]

        rsign = Signal()
        m.d.comb += rsign.eq(a_sign ^ b_sign)

        # Significands with implicit 1
        sig_a = Cat(a_mant, Const(1, 1))  # wf+1 bits
        sig_b = Cat(b_mant, Const(1, 1))

        # Split into halves for Karatsuba
        hw = (wf + 1 + 1) // 2  # half-width
        fw = wf + 1  # full sig width

        a_lo = Signal(hw); a_hi = Signal(fw - hw)
        b_lo = Signal(hw); b_hi = Signal(fw - hw)
        m.d.comb += [
            a_lo.eq(sig_a[:hw]), a_hi.eq(sig_a[hw:]),
            b_lo.eq(sig_b[:hw]), b_hi.eq(sig_b[hw:]),
        ]

        # Pipeline 0→1
        a_lo1 = Signal(hw); a_hi1 = Signal(fw - hw)
        b_lo1 = Signal(hw); b_hi1 = Signal(fw - hw)
        a_exc1 = Signal(2); b_exc1 = Signal(2)
        rsign1 = Signal(); a_exp1 = Signal(we); b_exp1 = Signal(we)
        m.d.sync += [
            a_lo1.eq(a_lo), a_hi1.eq(a_hi),
            b_lo1.eq(b_lo), b_hi1.eq(b_hi),
            a_exc1.eq(a_exc), b_exc1.eq(b_exc),
            rsign1.eq(rsign), a_exp1.eq(a_exp), b_exp1.eq(b_exp),
        ]

        # Stage 1: Exception decode + prepare sums for Karatsuba
        exc_sel = Cat(b_exc1, a_exc1)
        exc_result = Signal(2)
        with m.Switch(exc_sel):
            with m.Case(0b0000): m.d.comb += exc_result.eq(0b00)
            with m.Case(0b0001): m.d.comb += exc_result.eq(0b00)
            with m.Case(0b0100): m.d.comb += exc_result.eq(0b00)
            with m.Case(0b0101): m.d.comb += exc_result.eq(0b01)
            with m.Case(0b0110): m.d.comb += exc_result.eq(0b10)
            with m.Case(0b1001): m.d.comb += exc_result.eq(0b10)
            with m.Case(0b1010): m.d.comb += exc_result.eq(0b10)
            with m.Default():    m.d.comb += exc_result.eq(0b11)

        bias = fmt.bias
        exp_sum = Signal(we + 2)
        m.d.comb += exp_sum.eq(Cat(a_exp1, Const(0, 2)) + Cat(b_exp1, Const(0, 2)) - bias)

        # Karatsuba sums
        a_sum = Signal(hw + 1)
        b_sum = Signal(hw + 1)
        m.d.comb += [a_sum.eq(a_hi1 + a_lo1), b_sum.eq(b_hi1 + b_lo1)]

        # 3 multiplies
        p_hh = Signal(2 * (fw - hw))
        p_ll = Signal(2 * hw)
        p_mid = Signal(2 * (hw + 1))
        m.d.comb += [
            p_hh.eq(a_hi1 * b_hi1),
            p_ll.eq(a_lo1 * b_lo1),
            p_mid.eq(a_sum * b_sum),
        ]

        # Pipeline 1→2
        p_hh2 = Signal(2 * (fw - hw)); p_ll2 = Signal(2 * hw)
        p_mid2 = Signal(2 * (hw + 1))
        exp_sum2 = Signal(we + 2); exc_result2 = Signal(2); rsign2 = Signal()
        m.d.sync += [
            p_hh2.eq(p_hh), p_ll2.eq(p_ll), p_mid2.eq(p_mid),
            exp_sum2.eq(exp_sum), exc_result2.eq(exc_result), rsign2.eq(rsign1),
        ]

        # Stage 2: Combine Karatsuba: product = p_hh << 2*hw + (p_mid - p_hh - p_ll) << hw + p_ll
        prod_w = 2 * fw
        cross = Signal(2 * (hw + 1) + 1)
        m.d.comb += cross.eq(p_mid2 - p_hh2 - p_ll2)

        product = Signal(prod_w + 2)
        m.d.comb += product.eq((p_hh2 << (2 * hw)) + (cross << hw) + p_ll2)

        sig_prod = Signal(prod_w)
        m.d.comb += sig_prod.eq(product[:prod_w])

        # Pipeline 2→3
        sig_prod3 = Signal(prod_w)
        exp_sum3 = Signal(we + 2); exc_result3 = Signal(2); rsign3 = Signal()
        m.d.sync += [
            sig_prod3.eq(sig_prod),
            exp_sum3.eq(exp_sum2), exc_result3.eq(exc_result2), rsign3.eq(rsign2),
        ]

        # Stage 3: Normalize + Round (same as FPMul)
        norm = Signal()
        m.d.comb += norm.eq(sig_prod3[prod_w - 1])

        exp_post = Signal(we + 2)
        m.d.comb += exp_post.eq(exp_sum3 + norm)

        sig_norm = Signal(prod_w)
        with m.If(norm):
            m.d.comb += sig_norm.eq(sig_prod3)
        with m.Else():
            m.d.comb += sig_norm.eq(sig_prod3 << 1)

        rmant = Signal(wf)
        m.d.comb += rmant.eq(sig_norm[prod_w - 1 - wf:prod_w - 1])

        g = Signal(); r = Signal(); s = Signal()
        m.d.comb += [
            g.eq(sig_norm[prod_w - 1 - wf - 1]),
            r.eq(sig_norm[prod_w - 1 - wf - 2]),
        ]
        sticky_start = prod_w - 1 - wf - 3
        if sticky_start >= 0:
            m.d.comb += s.eq(sig_norm[:sticky_start + 1].any())

        round_in = Signal(wf + 3)
        m.d.comb += round_in.eq(Cat(s, r, g, rmant))
        rounder = RoundingUnit(wf)
        m.submodules.rounder = rounder
        m.d.comb += rounder.mantissa_in.eq(round_in)

        final_exp = Signal(we + 2)
        m.d.comb += final_exp.eq(exp_post + rounder.overflow)

        # Pipeline 3→4
        rmant4 = Signal(wf); fexp4 = Signal(we + 2)
        exc4 = Signal(2); rsign4 = Signal()
        m.d.sync += [
            rmant4.eq(rounder.mantissa_out), fexp4.eq(final_exp),
            exc4.eq(exc_result3), rsign4.eq(rsign3),
        ]

        # Stage 4: Overflow/underflow + pack
        exp_top = fexp4[we:we + 2]
        exc_post = Signal(2)
        with m.Switch(exp_top):
            with m.Case(0b00): m.d.comb += exc_post.eq(0b01)
            with m.Case(0b01): m.d.comb += exc_post.eq(0b10)
            with m.Case(0b10): m.d.comb += exc_post.eq(0b00)
            with m.Case(0b11): m.d.comb += exc_post.eq(0b00)

        final_exc = Signal(2)
        with m.If((exc4 == 0b11) | (exc4 == 0b10) | (exc4 == 0b00)):
            m.d.comb += final_exc.eq(exc4)
        with m.Else():
            m.d.comb += final_exc.eq(exc_post)

        out_mant = Signal(wf); out_exp = Signal(we)
        with m.If(final_exc == 0b01):
            m.d.comb += [out_mant.eq(rmant4), out_exp.eq(fexp4[:we])]

        # Pipeline 4→5
        o_r = Signal(fmt.width)
        m.d.sync += o_r.eq(Cat(out_mant, out_exp, rsign4, final_exc))

        # Output
        m.d.comb += self.o.eq(o_r)
        return m
