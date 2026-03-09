"""Dual-path floating-point adder (pipelined, 6 stages).

Near path (|exp_diff| <= 1): no alignment shift, LZC+normalize.
Far path (|exp_diff| > 1): alignment shift, no LZC needed.
"""
from __future__ import annotations

from math import ceil, log2

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from ..building_blocks import LeadingZeroCounter, RoundingUnit

__all__ = ["FPAddDualPath"]


class FPAddDualPath(PipelinedComponent):
    """Dual-path FP adder (6-cycle latency).

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

        # Stage 0: Unpack, swap so |x| >= |y|
        a_mant = Signal(wf); a_exp = Signal(we); a_sign = Signal(); a_exc = Signal(2)
        b_mant = Signal(wf); b_exp = Signal(we); b_sign = Signal(); b_exc = Signal(2)
        m.d.comb += [
            a_mant.eq(self.a[:wf]), a_exp.eq(self.a[wf:wf+we]),
            a_sign.eq(self.a[wf+we]), a_exc.eq(self.a[wf+we+1:]),
            b_mant.eq(self.b[:wf]), b_exp.eq(self.b[wf:wf+we]),
            b_sign.eq(self.b[wf+we]), b_exc.eq(self.b[wf+we+1:]),
        ]

        a_mag = Cat(a_mant, a_exp, a_exc)
        b_mag = Cat(b_mant, b_exp, b_exc)
        swap = Signal()
        m.d.comb += swap.eq(a_mag < b_mag)

        x_mant = Signal(wf); x_exp = Signal(we); x_sign = Signal(); x_exc = Signal(2)
        y_mant = Signal(wf); y_exp = Signal(we); y_sign = Signal(); y_exc = Signal(2)
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

        exp_diff = Signal(we)
        m.d.comb += exp_diff.eq(x_exp - y_exp)
        is_near = Signal()
        m.d.comb += is_near.eq(exp_diff <= 1)
        eff_sub = Signal()
        m.d.comb += eff_sub.eq(x_sign ^ y_sign)

        # Pipeline reg 0→1
        x_m1 = Signal(wf); x_e1 = Signal(we); x_s1 = Signal(); x_x1 = Signal(2)
        y_m1 = Signal(wf); y_e1 = Signal(we); y_s1 = Signal(); y_x1 = Signal(2)
        ed1 = Signal(we); near1 = Signal(); esub1 = Signal()
        m.d.sync += [
            x_m1.eq(x_mant), x_e1.eq(x_exp), x_s1.eq(x_sign), x_x1.eq(x_exc),
            y_m1.eq(y_mant), y_e1.eq(y_exp), y_s1.eq(y_sign), y_x1.eq(y_exc),
            ed1.eq(exp_diff), near1.eq(is_near), esub1.eq(eff_sub),
        ]

        # Stage 1: Near path: subtract mantissas; Far path: align shift
        frac_x = Cat(x_m1, x_x1[0])  # wf+1 bits
        frac_y = Cat(y_m1, y_x1[0])

        # Near: just subtract, no shift
        near_diff = Signal(wf + 2)
        near_shifted_y = Signal(wf + 1)
        with m.If(ed1 == 1):
            m.d.comb += near_shifted_y.eq(frac_y >> 1)
        with m.Else():
            m.d.comb += near_shifted_y.eq(frac_y)
        with m.If(esub1):
            m.d.comb += near_diff.eq(frac_x - near_shifted_y)
        with m.Else():
            m.d.comb += near_diff.eq(frac_x + near_shifted_y)

        # Far: full alignment
        far_y_ext = Signal(wf + 4)
        m.d.comb += far_y_ext.eq(frac_y << 3)
        far_shifted = Signal(wf + 4)
        sh = Signal(we)
        with m.If(ed1 > wf + 3):
            m.d.comb += sh.eq(wf + 3)
        with m.Else():
            m.d.comb += sh.eq(ed1)
        m.d.comb += far_shifted.eq(far_y_ext >> sh)

        far_x_ext = Signal(wf + 4)
        m.d.comb += far_x_ext.eq(frac_x << 3)
        far_sum = Signal(wf + 5)
        with m.If(esub1):
            m.d.comb += far_sum.eq(far_x_ext - far_shifted)
        with m.Else():
            m.d.comb += far_sum.eq(far_x_ext + far_shifted)

        # Pipeline reg 1→2
        near_diff2 = Signal(wf + 2); far_sum2 = Signal(wf + 5); near2 = Signal()
        x_e2 = Signal(we); x_s2 = Signal(); x_x2 = Signal(2); y_x2 = Signal(2)
        esub2 = Signal()
        m.d.sync += [
            near_diff2.eq(near_diff), far_sum2.eq(far_sum), near2.eq(near1),
            x_e2.eq(x_e1), x_s2.eq(x_s1), x_x2.eq(x_x1), y_x2.eq(y_x1),
            esub2.eq(esub1),
        ]

        # Stage 2: Near path LZC; Far path normalize-by-1
        lzc_w = wf + 2
        lzc = LeadingZeroCounter(lzc_w)
        m.submodules.lzc = lzc
        m.d.comb += lzc.i.eq(near_diff2)
        near_shifted = Signal(lzc_w)
        m.d.comb += near_shifted.eq(near_diff2 << lzc.count)

        far_norm = Signal(name="far_norm")
        m.d.comb += far_norm.eq(far_sum2[wf + 4])
        far_frac = Signal(wf + 4)
        with m.If(far_norm):
            m.d.comb += far_frac.eq(far_sum2[1:wf + 5])
        with m.Else():
            m.d.comb += far_frac.eq(far_sum2[:wf + 4])

        # Pipeline reg 2→3
        near_sh3 = Signal(lzc_w); far_frac3 = Signal(wf + 4); near3 = Signal()
        lzc_cnt3 = Signal(range(lzc_w + 1)); far_norm3 = Signal()
        x_e3 = Signal(we); x_s3 = Signal(); x_x3 = Signal(2); y_x3 = Signal(2)
        esub3 = Signal()
        m.d.sync += [
            near_sh3.eq(near_shifted), far_frac3.eq(far_frac), near3.eq(near2),
            lzc_cnt3.eq(lzc.count), far_norm3.eq(far_norm),
            x_e3.eq(x_e2), x_s3.eq(x_s2), x_x3.eq(x_x2), y_x3.eq(y_x2),
            esub3.eq(esub2),
        ]

        # Stage 3: Select path, compute exponent
        mux_frac = Signal(wf + 4)
        new_exp = Signal(we + 2)
        with m.If(near3):
            m.d.comb += [
                mux_frac.eq(near_sh3 << 2),
                new_exp.eq(Cat(x_e3, Const(0, 2)) + 1 - lzc_cnt3),
            ]
        with m.Else():
            m.d.comb += [
                mux_frac.eq(far_frac3),
                new_exp.eq(Cat(x_e3, Const(0, 2)) + far_norm3),
            ]

        # Pipeline reg 3→4
        mux_frac4 = Signal(wf + 4); new_exp4 = Signal(we + 2)
        x_s4 = Signal(); x_x4 = Signal(2); y_x4 = Signal(2); esub4 = Signal()
        m.d.sync += [
            mux_frac4.eq(mux_frac), new_exp4.eq(new_exp),
            x_s4.eq(x_s3), x_x4.eq(x_x3), y_x4.eq(y_x3), esub4.eq(esub3),
        ]

        # Stage 4: Rounding
        rounder = RoundingUnit(wf)
        m.submodules.rounder = rounder
        round_in = Signal(wf + 3)
        top = mux_frac4[4 - 1:]  # wf+1 bits from top
        sticky = mux_frac4[:3].any()
        m.d.comb += round_in.eq(Cat(sticky, mux_frac4[1:wf + 3]))
        m.d.comb += rounder.mantissa_in.eq(round_in)

        final_exp = Signal(we + 2)
        m.d.comb += final_exp.eq(new_exp4 + rounder.overflow)

        # Pipeline reg 4→5
        rmant5 = Signal(wf); fexp5 = Signal(we + 2)
        x_s5 = Signal(); x_x5 = Signal(2); y_x5 = Signal(2); esub5 = Signal()
        m.d.sync += [
            rmant5.eq(rounder.mantissa_out), fexp5.eq(final_exp),
            x_s5.eq(x_s4), x_x5.eq(x_x4), y_x5.eq(y_x4), esub5.eq(esub4),
        ]

        # Stage 5: Exception handling, pack
        max_exp = (1 << we) - 1
        ov = Signal(); uf = Signal()
        m.d.comb += [
            ov.eq((~fexp5[we + 1]) & (fexp5[:we + 1] >= max_exp)),
            uf.eq(fexp5[we + 1]),
        ]

        out_exc = Signal(2); out_sign = Signal()
        out_mant = Signal(wf); out_exp = Signal(we)

        both_normal = (x_x5 == 0b01) & (y_x5 == 0b01)
        with m.If(both_normal):
            with m.If(ov):
                m.d.comb += out_exc.eq(0b10)
            with m.Elif(uf):
                m.d.comb += out_exc.eq(0b00)
            with m.Else():
                m.d.comb += [out_exc.eq(0b01), out_mant.eq(rmant5), out_exp.eq(fexp5[:we])]
            m.d.comb += out_sign.eq(x_s5)
        with m.Else():
            # Special cases use same logic as FPAdd
            sdExnXY = Cat(y_x5, x_x5)
            with m.Switch(sdExnXY):
                with m.Case(0b0000): m.d.comb += out_exc.eq(0b00)
                with m.Case(0b0001): m.d.comb += out_exc.eq(0b01)
                with m.Case(0b0100): m.d.comb += out_exc.eq(0b01)
                with m.Case(0b1010):
                    m.d.comb += out_exc.eq(Mux(esub5, 0b11, 0b10))
                with m.Default(): m.d.comb += out_exc.eq(0b11)
            m.d.comb += out_sign.eq(x_s5)

        o_r = Signal(fmt.width)
        m.d.sync += o_r.eq(Cat(out_mant, out_exp, out_sign, out_exc))
        m.d.comb += self.o.eq(o_r)

        return m
