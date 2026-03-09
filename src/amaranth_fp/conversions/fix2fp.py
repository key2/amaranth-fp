"""Fixed-point to floating-point conversion (pipelined, 3 stages).

Converts a fixed-point number to internal FloPoCo FP format:
    [exception(2) | sign(1) | exponent(we) | mantissa(wf)]
"""
from __future__ import annotations

from math import ceil, log2

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from ..building_blocks import LeadingZeroCounter

__all__ = ["Fix2FP"]


class Fix2FP(PipelinedComponent):
    """Pipelined fixed-point to floating-point converter (3-cycle latency).

    Parameters
    ----------
    int_width : int
        Integer part width (including sign bit if signed).
    frac_width : int
        Fractional part width.
    signed : bool
        Whether the fixed-point input is signed (2's complement).
    fmt : FPFormat
        Target floating-point format.

    Attributes
    ----------
    fix_in : Signal(int_width + frac_width), in
    fp_out : Signal(fmt.width), out
    """

    def __init__(self, int_width: int, frac_width: int, signed: bool, fmt: FPFormat) -> None:
        super().__init__()
        self.int_width = int_width
        self.frac_width = frac_width
        self.is_signed = signed
        self.fmt = fmt
        total = int_width + frac_width
        self.fix_in = Signal(total, name="fix_in")
        self.fp_out = Signal(fmt.width, name="fp_out")
        self.latency = 3

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we = fmt.we
        wf = fmt.wf
        bias = fmt.bias
        total = self.int_width + self.frac_width

        # ==================================================================
        # Stage 0: sign handling, absolute value
        # ==================================================================
        sign = Signal(name="sign")
        abs_val = Signal(total, name="abs_val")

        if self.is_signed:
            m.d.comb += sign.eq(self.fix_in[total - 1])
            with m.If(sign):
                m.d.comb += abs_val.eq(-self.fix_in)
            with m.Else():
                m.d.comb += abs_val.eq(self.fix_in)
        else:
            m.d.comb += [
                sign.eq(0),
                abs_val.eq(self.fix_in),
            ]

        is_zero = Signal(name="is_zero")
        m.d.comb += is_zero.eq(abs_val == 0)

        # ── Stage 0 → 1 pipeline register ──
        sign_r1 = Signal(name="sign_r1")
        abs_val_r1 = Signal(total, name="abs_val_r1")
        is_zero_r1 = Signal(name="is_zero_r1")
        m.d.sync += [
            sign_r1.eq(sign),
            abs_val_r1.eq(abs_val),
            is_zero_r1.eq(is_zero),
        ]
        for s in [sign_r1, abs_val_r1, is_zero_r1]:
            self.add_latency(s, 1)

        # ==================================================================
        # Stage 1: LZC + normalization shift
        # ==================================================================
        lzc = LeadingZeroCounter(total)
        m.submodules.lzc = lzc
        m.d.comb += lzc.i.eq(abs_val_r1)

        shifted = Signal(total, name="shifted")
        m.d.comb += shifted.eq(abs_val_r1 << lzc.count)

        exp_val = Signal(we + 2, name="exp_val")
        m.d.comb += exp_val.eq(
            (total - 1 + bias) - lzc.count - self.frac_width
        )

        # ── Stage 1 → 2 pipeline register ──
        sign_r2 = Signal(name="sign_r2")
        shifted_r2 = Signal(total, name="shifted_r2")
        exp_val_r2 = Signal(we + 2, name="exp_val_r2")
        is_zero_r2 = Signal(name="is_zero_r2")
        m.d.sync += [
            sign_r2.eq(sign_r1),
            shifted_r2.eq(shifted),
            exp_val_r2.eq(exp_val),
            is_zero_r2.eq(is_zero_r1),
        ]
        for s in [sign_r2, shifted_r2, exp_val_r2, is_zero_r2]:
            self.add_latency(s, 2)

        # ==================================================================
        # Stage 2: exponent computation, pack
        # ==================================================================
        mantissa = Signal(wf, name="mantissa")
        if total - 1 >= wf:
            m.d.comb += mantissa.eq(shifted_r2[total - 1 - wf:total - 1])
        else:
            m.d.comb += mantissa.eq(shifted_r2[:total - 1] << (wf - (total - 1)))

        exc = Signal(2, name="exc")
        final_exp = Signal(we, name="final_exp")
        final_mant = Signal(wf, name="final_mant")

        with m.If(is_zero_r2):
            m.d.comb += [
                exc.eq(0b00), final_exp.eq(0), final_mant.eq(0),
            ]
        with m.Elif(exp_val_r2[we + 1]):  # negative exponent => underflow
            m.d.comb += [
                exc.eq(0b00), final_exp.eq(0), final_mant.eq(0),
            ]
        with m.Elif(exp_val_r2[we:we + 1].any()):  # overflow
            m.d.comb += [
                exc.eq(0b10), final_exp.eq(0), final_mant.eq(0),
            ]
        with m.Else():
            m.d.comb += [
                exc.eq(0b01),
                final_exp.eq(exp_val_r2[:we]),
                final_mant.eq(mantissa),
            ]

        # ── Stage 2 → 3 pipeline register (output) ──
        fp_out_r3 = Signal(fmt.width, name="fp_out_r3")
        m.d.sync += fp_out_r3.eq(Cat(final_mant, final_exp, sign_r2, exc))
        m.d.comb += self.fp_out.eq(fp_out_r3)

        return m
