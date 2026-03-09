"""Single-path floating-point adder (pipelined, 7 stages).

Based on FloPoCo's FPAddSinglePath algorithm, translated to Amaranth HDL.
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

__all__ = ["FPAdd"]


class FPAdd(PipelinedComponent):
    """Pipelined single-path floating-point adder (7-cycle latency).

    Parameters
    ----------
    fmt : FPFormat
        Floating-point format (defines we, wf).

    Attributes
    ----------
    a : Signal(fmt.width), in
        First operand in internal format.
    b : Signal(fmt.width), in
        Second operand in internal format.
    o : Signal(fmt.width), out
        Result in internal format.
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.b = Signal(fmt.width, name="b")
        self.o = Signal(fmt.width, name="o")
        self.latency = 7

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we = fmt.we
        wf = fmt.wf

        # ==================================================================
        # Stage 0: Unpack inputs, compute swap condition, select x/y
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

        a_mag = Signal(2 + we + wf, name="a_mag")
        b_mag = Signal(2 + we + wf, name="b_mag")
        m.d.comb += [
            a_mag.eq(Cat(a_mant, a_exp, a_exc)),
            b_mag.eq(Cat(b_mant, b_exp, b_exc)),
        ]

        swap = Signal(name="swap")
        m.d.comb += swap.eq(a_mag < b_mag)

        x_mant = Signal(wf, name="x_mant")
        x_exp = Signal(we, name="x_exp")
        x_sign = Signal(name="x_sign")
        x_exc = Signal(2, name="x_exc")

        y_mant = Signal(wf, name="y_mant")
        y_exp = Signal(we, name="y_exp")
        y_sign = Signal(name="y_sign")
        y_exc = Signal(2, name="y_exc")

        with m.If(swap):
            m.d.comb += [
                x_mant.eq(b_mant), x_exp.eq(b_exp), x_sign.eq(b_sign), x_exc.eq(b_exc),
                y_mant.eq(a_mant), y_exp.eq(a_exp), y_sign.eq(a_sign), y_exc.eq(a_exc),
            ]
        with m.Else():
            m.d.comb += [
                x_mant.eq(a_mant), x_exp.eq(a_exp), x_sign.eq(a_sign), x_exc.eq(a_exc),
                y_mant.eq(b_mant), y_exp.eq(b_exp), y_sign.eq(b_sign), y_exc.eq(b_exc),
            ]

        # ── Stage 0 → 1 pipeline register ──
        x_mant_r1 = Signal(wf, name="x_mant_r1")
        x_exp_r1 = Signal(we, name="x_exp_r1")
        x_sign_r1 = Signal(name="x_sign_r1")
        x_exc_r1 = Signal(2, name="x_exc_r1")
        y_mant_r1 = Signal(wf, name="y_mant_r1")
        y_exp_r1 = Signal(we, name="y_exp_r1")
        y_sign_r1 = Signal(name="y_sign_r1")
        y_exc_r1 = Signal(2, name="y_exc_r1")
        m.d.sync += [
            x_mant_r1.eq(x_mant), x_exp_r1.eq(x_exp), x_sign_r1.eq(x_sign), x_exc_r1.eq(x_exc),
            y_mant_r1.eq(y_mant), y_exp_r1.eq(y_exp), y_sign_r1.eq(y_sign), y_exc_r1.eq(y_exc),
        ]
        self.add_latency(x_mant_r1, 1)
        self.add_latency(x_exp_r1, 1)
        self.add_latency(x_sign_r1, 1)
        self.add_latency(x_exc_r1, 1)
        self.add_latency(y_mant_r1, 1)
        self.add_latency(y_exp_r1, 1)
        self.add_latency(y_sign_r1, 1)
        self.add_latency(y_exc_r1, 1)

        # ==================================================================
        # Stage 1: Exception decode, compute exp_diff, prepend implicit 1
        # ==================================================================
        eff_sub = Signal(name="eff_sub")
        m.d.comb += eff_sub.eq(x_sign_r1 ^ y_sign_r1)

        sXsYExnXY = Signal(6, name="sXsYExnXY")
        m.d.comb += sXsYExnXY.eq(Cat(y_exc_r1, x_exc_r1, y_sign_r1, x_sign_r1))

        exc_result = Signal(2, name="exc_result")
        sign_result = Signal(name="sign_result")

        sdExnXY = Signal(4, name="sdExnXY")
        m.d.comb += sdExnXY.eq(Cat(y_exc_r1, x_exc_r1))

        with m.Switch(sdExnXY):
            with m.Case(0b0000):
                m.d.comb += exc_result.eq(0b00)
            with m.Case(0b0001):
                m.d.comb += exc_result.eq(0b01)
            with m.Case(0b0100):
                m.d.comb += exc_result.eq(0b01)
            with m.Case(0b0101):
                m.d.comb += exc_result.eq(0b01)
            with m.Case(0b0010):
                m.d.comb += exc_result.eq(0b10)
            with m.Case(0b1000):
                m.d.comb += exc_result.eq(0b10)
            with m.Case(0b0110):
                m.d.comb += exc_result.eq(0b10)
            with m.Case(0b1001):
                m.d.comb += exc_result.eq(0b10)
            with m.Case(0b1010):
                m.d.comb += exc_result.eq(Mux(eff_sub, 0b11, 0b10))
            with m.Default():
                m.d.comb += exc_result.eq(0b11)

        with m.If((x_exc_r1 == 0b00) & (y_exc_r1 == 0b00)):
            m.d.comb += sign_result.eq(x_sign_r1 & y_sign_r1)
        with m.Else():
            m.d.comb += sign_result.eq(x_sign_r1)

        exp_diff = Signal(we, name="exp_diff")
        m.d.comb += exp_diff.eq(x_exp_r1 - y_exp_r1)

        frac_x = Signal(wf + 1, name="frac_x")
        frac_y = Signal(wf + 1, name="frac_y")
        m.d.comb += [
            frac_x.eq(Cat(x_mant_r1, x_exc_r1[0])),
            frac_y.eq(Cat(y_mant_r1, y_exc_r1[0])),
        ]

        # ── Stage 1 → 2 pipeline register ──
        eff_sub_r2 = Signal(name="eff_sub_r2")
        exc_result_r2 = Signal(2, name="exc_result_r2")
        sign_result_r2 = Signal(name="sign_result_r2")
        exp_diff_r2 = Signal(we, name="exp_diff_r2")
        frac_x_r2 = Signal(wf + 1, name="frac_x_r2")
        frac_y_r2 = Signal(wf + 1, name="frac_y_r2")
        x_exp_r2 = Signal(we, name="x_exp_r2")
        x_sign_r2 = Signal(name="x_sign_r2")
        x_exc_r2 = Signal(2, name="x_exc_r2")
        y_exc_r2 = Signal(2, name="y_exc_r2")
        x_mant_r2 = Signal(wf, name="x_mant_r2")
        m.d.sync += [
            eff_sub_r2.eq(eff_sub),
            exc_result_r2.eq(exc_result),
            sign_result_r2.eq(sign_result),
            exp_diff_r2.eq(exp_diff),
            frac_x_r2.eq(frac_x),
            frac_y_r2.eq(frac_y),
            x_exp_r2.eq(x_exp_r1),
            x_sign_r2.eq(x_sign_r1),
            x_exc_r2.eq(x_exc_r1),
            y_exc_r2.eq(y_exc_r1),
            x_mant_r2.eq(x_mant_r1),
        ]
        self.add_latency(eff_sub_r2, 2)
        self.add_latency(exc_result_r2, 2)
        self.add_latency(sign_result_r2, 2)
        self.add_latency(exp_diff_r2, 2)
        self.add_latency(frac_x_r2, 2)
        self.add_latency(frac_y_r2, 2)
        self.add_latency(x_exp_r2, 2)
        self.add_latency(x_sign_r2, 2)
        self.add_latency(x_exc_r2, 2)
        self.add_latency(y_exc_r2, 2)
        self.add_latency(x_mant_r2, 2)

        # ==================================================================
        # Stage 2: Alignment shift of fracY, compute sticky
        # ==================================================================
        shift_bits = ceil(log2(wf + 4))
        shift_val = Signal(shift_bits, name="shift_val")

        with m.If(exp_diff_r2 > wf + 3):
            m.d.comb += shift_val.eq(wf + 3)
        with m.Else():
            m.d.comb += shift_val.eq(exp_diff_r2[:shift_bits])

        frac_y_ext = Signal(wf + 4, name="frac_y_ext")
        m.d.comb += frac_y_ext.eq(frac_y_r2 << 3)

        shifted_frac_y = Signal(wf + 4, name="shifted_frac_y")
        sticky_bits = Signal(wf + 4, name="sticky_bits")

        wide_y = Signal(2 * (wf + 4), name="wide_y")
        m.d.comb += wide_y.eq(frac_y_ext << (wf + 4))
        shifted_wide = Signal(2 * (wf + 4), name="shifted_wide")
        m.d.comb += shifted_wide.eq(wide_y >> shift_val)
        m.d.comb += shifted_frac_y.eq(shifted_wide[wf + 4:])

        sticky_mask = Signal(2 * (wf + 4), name="sticky_mask")
        m.d.comb += sticky_mask.eq(shifted_wide[:wf + 4])
        sticky = Signal(name="sticky")
        m.d.comb += sticky.eq(sticky_mask.any())

        frac_y_pad = Signal(wf + 4, name="frac_y_pad")
        m.d.comb += frac_y_pad.eq(shifted_frac_y)

        frac_x_pad = Signal(wf + 4, name="frac_x_pad")
        m.d.comb += frac_x_pad.eq(frac_x_r2 << 3)

        # ── Stage 2 → 3 pipeline register ──
        frac_x_pad_r3 = Signal(wf + 4, name="frac_x_pad_r3")
        frac_y_pad_r3 = Signal(wf + 4, name="frac_y_pad_r3")
        sticky_r3 = Signal(name="sticky_r3")
        eff_sub_r3 = Signal(name="eff_sub_r3")
        x_exp_r3 = Signal(we, name="x_exp_r3")
        x_sign_r3 = Signal(name="x_sign_r3")
        x_exc_r3 = Signal(2, name="x_exc_r3")
        y_exc_r3 = Signal(2, name="y_exc_r3")
        exc_result_r3 = Signal(2, name="exc_result_r3")
        sign_result_r3 = Signal(name="sign_result_r3")
        x_mant_r3 = Signal(wf, name="x_mant_r3")
        m.d.sync += [
            frac_x_pad_r3.eq(frac_x_pad),
            frac_y_pad_r3.eq(frac_y_pad),
            sticky_r3.eq(sticky),
            eff_sub_r3.eq(eff_sub_r2),
            x_exp_r3.eq(x_exp_r2),
            x_sign_r3.eq(x_sign_r2),
            x_exc_r3.eq(x_exc_r2),
            y_exc_r3.eq(y_exc_r2),
            exc_result_r3.eq(exc_result_r2),
            sign_result_r3.eq(sign_result_r2),
            x_mant_r3.eq(x_mant_r2),
        ]
        self.add_latency(frac_x_pad_r3, 3)
        self.add_latency(frac_y_pad_r3, 3)
        self.add_latency(sticky_r3, 3)
        self.add_latency(eff_sub_r3, 3)
        self.add_latency(x_exp_r3, 3)
        self.add_latency(x_sign_r3, 3)
        self.add_latency(x_exc_r3, 3)
        self.add_latency(y_exc_r3, 3)
        self.add_latency(exc_result_r3, 3)
        self.add_latency(sign_result_r3, 3)
        self.add_latency(x_mant_r3, 3)

        # ==================================================================
        # Stage 3: Addition (frac_sum)
        # ==================================================================
        frac_y_effective = Signal(wf + 4, name="frac_y_effective")
        cin = Signal(name="cin")

        with m.If(eff_sub_r3):
            m.d.comb += [
                frac_y_effective.eq(~frac_y_pad_r3),
                cin.eq(~sticky_r3),
            ]
        with m.Else():
            m.d.comb += [
                frac_y_effective.eq(frac_y_pad_r3),
                cin.eq(0),
            ]

        frac_sum = Signal(wf + 5, name="frac_sum")
        m.d.comb += frac_sum.eq(frac_x_pad_r3 + frac_y_effective + cin)

        frac_for_norm = Signal(wf + 5, name="frac_for_norm")
        with m.If(eff_sub_r3):
            m.d.comb += frac_for_norm.eq(frac_sum[:wf + 4])
        with m.Else():
            m.d.comb += frac_for_norm.eq(frac_sum)

        frac_sticky = Signal(wf + 6, name="frac_sticky")
        m.d.comb += frac_sticky.eq(Cat(sticky_r3, frac_for_norm))

        # eq_diff_sign computed here (needs frac_sum and sticky)
        eq_diff_sign = Signal(name="eq_diff_sign")
        m.d.comb += eq_diff_sign.eq(~frac_sum[:wf + 4].any() & ~sticky_r3)

        # ── Stage 3 → 4 pipeline register ──
        frac_sticky_r4 = Signal(wf + 6, name="frac_sticky_r4")
        eq_diff_sign_r4 = Signal(name="eq_diff_sign_r4")
        eff_sub_r4 = Signal(name="eff_sub_r4")
        x_exp_r4 = Signal(we, name="x_exp_r4")
        x_sign_r4 = Signal(name="x_sign_r4")
        x_exc_r4 = Signal(2, name="x_exc_r4")
        y_exc_r4 = Signal(2, name="y_exc_r4")
        exc_result_r4 = Signal(2, name="exc_result_r4")
        sign_result_r4 = Signal(name="sign_result_r4")
        x_mant_r4 = Signal(wf, name="x_mant_r4")
        m.d.sync += [
            frac_sticky_r4.eq(frac_sticky),
            eq_diff_sign_r4.eq(eq_diff_sign),
            eff_sub_r4.eq(eff_sub_r3),
            x_exp_r4.eq(x_exp_r3),
            x_sign_r4.eq(x_sign_r3),
            x_exc_r4.eq(x_exc_r3),
            y_exc_r4.eq(y_exc_r3),
            exc_result_r4.eq(exc_result_r3),
            sign_result_r4.eq(sign_result_r3),
            x_mant_r4.eq(x_mant_r3),
        ]
        self.add_latency(frac_sticky_r4, 4)
        self.add_latency(eq_diff_sign_r4, 4)
        self.add_latency(eff_sub_r4, 4)
        self.add_latency(x_exp_r4, 4)
        self.add_latency(x_sign_r4, 4)
        self.add_latency(x_exc_r4, 4)
        self.add_latency(y_exc_r4, 4)
        self.add_latency(exc_result_r4, 4)
        self.add_latency(sign_result_r4, 4)
        self.add_latency(x_mant_r4, 4)

        # ==================================================================
        # Stage 4: LZC + normalization shift
        # ==================================================================
        norm_width = wf + 6
        lzc = LeadingZeroCounter(norm_width)
        m.submodules.lzc = lzc
        m.d.comb += lzc.i.eq(frac_sticky_r4)

        lzc_count = lzc.count

        shifted_frac = Signal(norm_width, name="shifted_frac")
        m.d.comb += shifted_frac.eq(frac_sticky_r4 << lzc_count)

        ext_exp_inc = Signal(we + 2, name="ext_exp_inc")
        m.d.comb += ext_exp_inc.eq(Cat(x_exp_r4, Const(0, 2)) + 1)

        updated_exp = Signal(we + 2, name="updated_exp")
        m.d.comb += updated_exp.eq(ext_exp_inc - lzc_count)

        # ── Stage 4 → 5 pipeline register ──
        shifted_frac_r5 = Signal(norm_width, name="shifted_frac_r5")
        updated_exp_r5 = Signal(we + 2, name="updated_exp_r5")
        eq_diff_sign_r5 = Signal(name="eq_diff_sign_r5")
        eff_sub_r5 = Signal(name="eff_sub_r5")
        x_sign_r5 = Signal(name="x_sign_r5")
        x_exc_r5 = Signal(2, name="x_exc_r5")
        y_exc_r5 = Signal(2, name="y_exc_r5")
        exc_result_r5 = Signal(2, name="exc_result_r5")
        sign_result_r5 = Signal(name="sign_result_r5")
        x_mant_r5 = Signal(wf, name="x_mant_r5")
        x_exp_r5 = Signal(we, name="x_exp_r5")
        m.d.sync += [
            shifted_frac_r5.eq(shifted_frac),
            updated_exp_r5.eq(updated_exp),
            eq_diff_sign_r5.eq(eq_diff_sign_r4),
            eff_sub_r5.eq(eff_sub_r4),
            x_sign_r5.eq(x_sign_r4),
            x_exc_r5.eq(x_exc_r4),
            y_exc_r5.eq(y_exc_r4),
            exc_result_r5.eq(exc_result_r4),
            sign_result_r5.eq(sign_result_r4),
            x_mant_r5.eq(x_mant_r4),
            x_exp_r5.eq(x_exp_r4),
        ]
        self.add_latency(shifted_frac_r5, 5)
        self.add_latency(updated_exp_r5, 5)
        self.add_latency(eq_diff_sign_r5, 5)
        self.add_latency(eff_sub_r5, 5)
        self.add_latency(x_sign_r5, 5)
        self.add_latency(x_exc_r5, 5)
        self.add_latency(y_exc_r5, 5)
        self.add_latency(exc_result_r5, 5)
        self.add_latency(sign_result_r5, 5)
        self.add_latency(x_mant_r5, 5)
        self.add_latency(x_exp_r5, 5)

        # ==================================================================
        # Stage 5: Rounding, exponent adjustment
        # ==================================================================
        round_in = Signal(wf + 3, name="round_in")
        round_sticky = Signal(name="round_sticky")
        top_bits = Signal(wf + 2, name="top_bits")
        m.d.comb += top_bits.eq(shifted_frac_r5[norm_width - 1 - wf - 2:norm_width - 1])
        m.d.comb += round_sticky.eq(shifted_frac_r5[:norm_width - 1 - wf - 2].any())
        m.d.comb += round_in.eq(Cat(round_sticky, top_bits))

        rounder = RoundingUnit(wf)
        m.submodules.rounder = rounder
        m.d.comb += rounder.mantissa_in.eq(round_in)

        rounded_mant = rounder.mantissa_out
        round_overflow = rounder.overflow

        final_exp = Signal(we + 2, name="final_exp")
        m.d.comb += final_exp.eq(updated_exp_r5 + round_overflow)

        # ── Stage 5 → 6 pipeline register ──
        rounded_mant_r6 = Signal(wf, name="rounded_mant_r6")
        final_exp_r6 = Signal(we + 2, name="final_exp_r6")
        eq_diff_sign_r6 = Signal(name="eq_diff_sign_r6")
        eff_sub_r6 = Signal(name="eff_sub_r6")
        x_sign_r6 = Signal(name="x_sign_r6")
        x_exc_r6 = Signal(2, name="x_exc_r6")
        y_exc_r6 = Signal(2, name="y_exc_r6")
        exc_result_r6 = Signal(2, name="exc_result_r6")
        sign_result_r6 = Signal(name="sign_result_r6")
        x_mant_r6 = Signal(wf, name="x_mant_r6")
        x_exp_r6 = Signal(we, name="x_exp_r6")
        m.d.sync += [
            rounded_mant_r6.eq(rounded_mant),
            final_exp_r6.eq(final_exp),
            eq_diff_sign_r6.eq(eq_diff_sign_r5),
            eff_sub_r6.eq(eff_sub_r5),
            x_sign_r6.eq(x_sign_r5),
            x_exc_r6.eq(x_exc_r5),
            y_exc_r6.eq(y_exc_r5),
            exc_result_r6.eq(exc_result_r5),
            sign_result_r6.eq(sign_result_r5),
            x_mant_r6.eq(x_mant_r5),
            x_exp_r6.eq(x_exp_r5),
        ]
        self.add_latency(rounded_mant_r6, 6)
        self.add_latency(final_exp_r6, 6)
        self.add_latency(eq_diff_sign_r6, 6)
        self.add_latency(eff_sub_r6, 6)
        self.add_latency(x_sign_r6, 6)
        self.add_latency(x_exc_r6, 6)
        self.add_latency(y_exc_r6, 6)
        self.add_latency(exc_result_r6, 6)
        self.add_latency(sign_result_r6, 6)
        self.add_latency(x_mant_r6, 6)
        self.add_latency(x_exp_r6, 6)

        # ==================================================================
        # Stage 6: Overflow/underflow detection, final mux, pack output
        # ==================================================================
        exp_overflow = Signal(name="exp_overflow")
        exp_underflow = Signal(name="exp_underflow")

        max_exp = (1 << we) - 1
        m.d.comb += [
            exp_overflow.eq((~final_exp_r6[we + 1]) & (final_exp_r6[:we + 1] >= max_exp)),
            exp_underflow.eq(final_exp_r6[we + 1]),
        ]

        computed_exc = Signal(2, name="computed_exc")
        with m.If(eq_diff_sign_r6 & eff_sub_r6):
            m.d.comb += computed_exc.eq(0b00)
        with m.Elif(exp_overflow):
            m.d.comb += computed_exc.eq(0b10)
        with m.Elif(exp_underflow):
            m.d.comb += computed_exc.eq(0b00)
        with m.Else():
            m.d.comb += computed_exc.eq(0b01)

        computed_sign = Signal(name="computed_sign")
        with m.If(eq_diff_sign_r6 & eff_sub_r6):
            m.d.comb += computed_sign.eq(0)
        with m.Else():
            m.d.comb += computed_sign.eq(x_sign_r6)

        final_exc = Signal(2, name="final_exc")
        final_sign = Signal(name="final_sign")
        final_mant = Signal(wf, name="final_mant")
        final_e = Signal(we, name="final_e")

        both_normal = Signal(name="both_normal")
        m.d.comb += both_normal.eq((x_exc_r6 == 0b01) & (y_exc_r6 == 0b01))

        with m.If(both_normal):
            m.d.comb += [
                final_exc.eq(computed_exc),
                final_sign.eq(computed_sign),
                final_mant.eq(rounded_mant_r6),
                final_e.eq(final_exp_r6[:we]),
            ]
        with m.Elif((x_exc_r6 == 0b01) & (y_exc_r6 == 0b00)):
            m.d.comb += [
                final_exc.eq(0b01),
                final_sign.eq(x_sign_r6),
                final_mant.eq(x_mant_r6),
                final_e.eq(x_exp_r6),
            ]
        with m.Elif((x_exc_r6 == 0b00) & (y_exc_r6 == 0b00)):
            m.d.comb += [
                final_exc.eq(0b00),
                final_sign.eq(sign_result_r6),
                final_mant.eq(0),
                final_e.eq(0),
            ]
        with m.Else():
            m.d.comb += [
                final_exc.eq(exc_result_r6),
                final_sign.eq(sign_result_r6),
                final_mant.eq(0),
                final_e.eq(0),
            ]

        # ── Stage 6 → 7 pipeline register (output) ──
        o_r7 = Signal(fmt.width, name="o_r7")
        m.d.sync += o_r7.eq(Cat(final_mant, final_e, final_sign, final_exc))
        m.d.comb += self.o.eq(o_r7)

        return m
