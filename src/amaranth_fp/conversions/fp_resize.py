"""Floating-point precision conversion (pipelined, 2 stages).

Converts between different FP precisions in internal FloPoCo format:
    [exception(2) | sign(1) | exponent(we) | mantissa(wf)]
"""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from ..building_blocks import RoundingUnit

__all__ = ["FPResize"]


class FPResize(PipelinedComponent):
    """Pipelined floating-point precision converter (2-cycle latency).

    Parameters
    ----------
    fmt_in : FPFormat
        Source format.
    fmt_out : FPFormat
        Target format.

    Attributes
    ----------
    fp_in : Signal(fmt_in.width), in
    fp_out : Signal(fmt_out.width), out
    """

    def __init__(self, fmt_in: FPFormat, fmt_out: FPFormat) -> None:
        super().__init__()
        self.fmt_in = fmt_in
        self.fmt_out = fmt_out
        self.fp_in = Signal(fmt_in.width, name="fp_in")
        self.fp_out = Signal(fmt_out.width, name="fp_out")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        fi = self.fmt_in
        fo = self.fmt_out

        # ==================================================================
        # Stage 0: unpack, exponent bias adjustment
        # ==================================================================
        mant_in = Signal(fi.wf, name="mant_in")
        exp_in = Signal(fi.we, name="exp_in")
        sign_in = Signal(name="sign_in")
        exc_in = Signal(2, name="exc_in")

        m.d.comb += [
            mant_in.eq(self.fp_in[:fi.wf]),
            exp_in.eq(self.fp_in[fi.wf:fi.wf + fi.we]),
            sign_in.eq(self.fp_in[fi.wf + fi.we]),
            exc_in.eq(self.fp_in[fi.wf + fi.we + 1:]),
        ]

        bias_diff = fo.bias - fi.bias
        new_exp = Signal(max(fi.we, fo.we) + 2, name="new_exp")
        m.d.comb += new_exp.eq(exp_in + bias_diff)

        # ── Stage 0 → 1 pipeline register ──
        mant_in_r1 = Signal(fi.wf, name="mant_in_r1")
        new_exp_r1 = Signal(max(fi.we, fo.we) + 2, name="new_exp_r1")
        sign_in_r1 = Signal(name="sign_in_r1")
        exc_in_r1 = Signal(2, name="exc_in_r1")
        m.d.sync += [
            mant_in_r1.eq(mant_in),
            new_exp_r1.eq(new_exp),
            sign_in_r1.eq(sign_in),
            exc_in_r1.eq(exc_in),
        ]
        for s in [mant_in_r1, new_exp_r1, sign_in_r1, exc_in_r1]:
            self.add_latency(s, 1)

        # ==================================================================
        # Stage 1: mantissa resize + rounding, pack
        # ==================================================================
        mant_out = Signal(fo.wf, name="mant_out")
        new_exp_final = new_exp_r1

        if fo.wf >= fi.wf:
            # Widening: zero-extend LSBs
            m.d.comb += mant_out.eq(mant_in_r1 << (fo.wf - fi.wf))
        else:
            # Narrowing: truncate with rounding
            rounder = RoundingUnit(fo.wf)
            m.submodules.rounder = rounder
            diff = fi.wf - fo.wf
            top_bits = Signal(fo.wf, name="top_bits")
            m.d.comb += top_bits.eq(mant_in_r1[diff:])

            guard = Signal(name="guard")
            round_bit = Signal(name="round_bit")
            sticky = Signal(name="sticky")

            if diff >= 1:
                m.d.comb += guard.eq(mant_in_r1[diff - 1])
            else:
                m.d.comb += guard.eq(0)
            if diff >= 2:
                m.d.comb += round_bit.eq(mant_in_r1[diff - 2])
            else:
                m.d.comb += round_bit.eq(0)
            if diff >= 3:
                m.d.comb += sticky.eq(mant_in_r1[:diff - 2].any())
            else:
                m.d.comb += sticky.eq(0)

            round_in = Signal(fo.wf + 3, name="round_in")
            m.d.comb += round_in.eq(Cat(sticky, round_bit, guard, top_bits))
            m.d.comb += rounder.mantissa_in.eq(round_in)

            m.d.comb += mant_out.eq(rounder.mantissa_out)
            new_exp_adj = Signal(max(fi.we, fo.we) + 2, name="new_exp_adj")
            m.d.comb += new_exp_adj.eq(new_exp_r1 + rounder.overflow)
            new_exp_final = new_exp_adj

        # Check exponent overflow/underflow
        exp_out = Signal(fo.we, name="exp_out")
        exc_out = Signal(2, name="exc_out")
        final_mant = Signal(fo.wf, name="final_mant")

        with m.If(exc_in_r1 != 0b01):
            m.d.comb += [
                exc_out.eq(exc_in_r1),
                exp_out.eq(0),
            ]
            if fo.wf >= fi.wf:
                m.d.comb += mant_out.eq(0)
        with m.Elif(new_exp_final[max(fi.we, fo.we) + 1]):
            m.d.comb += [exc_out.eq(0b00), exp_out.eq(0)]
        with m.Elif(new_exp_final[fo.we:max(fi.we, fo.we) + 1].any()):
            m.d.comb += [exc_out.eq(0b10), exp_out.eq(0)]
        with m.Else():
            m.d.comb += [
                exc_out.eq(0b01),
                exp_out.eq(new_exp_final[:fo.we]),
            ]

        with m.If(exc_out == 0b01):
            m.d.comb += final_mant.eq(mant_out)
        with m.Else():
            m.d.comb += final_mant.eq(0)

        # ── Stage 1 → 2 pipeline register (output) ──
        fp_out_r2 = Signal(fo.width, name="fp_out_r2")
        m.d.sync += fp_out_r2.eq(Cat(final_mant, exp_out, sign_in_r1, exc_out))
        m.d.comb += self.fp_out.eq(fp_out_r2)

        return m
