"""Floating-point to fixed-point conversion (pipelined, 3 stages).

Converts internal FloPoCo FP format to fixed-point:
    [exception(2) | sign(1) | exponent(we) | mantissa(wf)]
"""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent

__all__ = ["FP2Fix"]


class FP2Fix(PipelinedComponent):
    """Pipelined floating-point to fixed-point converter (3-cycle latency).

    Parameters
    ----------
    fmt : FPFormat
        Source floating-point format.
    int_width : int
        Integer part width of output (including sign bit if signed).
    frac_width : int
        Fractional part width of output.
    signed : bool
        Whether the output is signed (2's complement).

    Attributes
    ----------
    fp_in : Signal(fmt.width), in
    fix_out : Signal(int_width + frac_width), out
    overflow : Signal(1), out
    """

    def __init__(self, fmt: FPFormat, int_width: int, frac_width: int, signed: bool) -> None:
        super().__init__()
        self.fmt = fmt
        self.int_width = int_width
        self.frac_width = frac_width
        self.is_signed = signed
        total = int_width + frac_width
        self.fp_in = Signal(fmt.width, name="fp_in")
        self.fix_out = Signal(total, name="fix_out")
        self.overflow = Signal(name="overflow")
        self.latency = 3

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we = fmt.we
        wf = fmt.wf
        bias = fmt.bias
        total = self.int_width + self.frac_width

        # ==================================================================
        # Stage 0: Unpack, compute shift direction and amounts
        # ==================================================================
        mant = Signal(wf, name="mant")
        exp = Signal(we, name="exp")
        sign = Signal(name="sign")
        exc = Signal(2, name="exc")

        m.d.comb += [
            mant.eq(self.fp_in[:wf]),
            exp.eq(self.fp_in[wf:wf + we]),
            sign.eq(self.fp_in[wf + we]),
            exc.eq(self.fp_in[wf + we + 1:]),
        ]

        sig = Signal(wf + 1, name="sig")
        m.d.comb += sig.eq(Cat(mant, Const(1, 1)))

        shift_offset = bias + wf - self.frac_width

        shift_left_amt = Signal(we + 1, name="shift_left_amt")
        shift_right_amt = Signal(we + 1, name="shift_right_amt")
        do_shift_left = Signal(name="do_shift_left")

        m.d.comb += do_shift_left.eq(exp >= shift_offset)

        with m.If(do_shift_left):
            m.d.comb += shift_left_amt.eq(exp - shift_offset)
        with m.Else():
            m.d.comb += shift_right_amt.eq(shift_offset - exp)

        # ── Stage 0 → 1 pipeline register ──
        sig_r1 = Signal(wf + 1, name="sig_r1")
        sign_r1 = Signal(name="sign_r1")
        exc_r1 = Signal(2, name="exc_r1")
        do_shift_left_r1 = Signal(name="do_shift_left_r1")
        shift_left_amt_r1 = Signal(we + 1, name="shift_left_amt_r1")
        shift_right_amt_r1 = Signal(we + 1, name="shift_right_amt_r1")
        m.d.sync += [
            sig_r1.eq(sig),
            sign_r1.eq(sign),
            exc_r1.eq(exc),
            do_shift_left_r1.eq(do_shift_left),
            shift_left_amt_r1.eq(shift_left_amt),
            shift_right_amt_r1.eq(shift_right_amt),
        ]
        for s in [sig_r1, sign_r1, exc_r1, do_shift_left_r1,
                   shift_left_amt_r1, shift_right_amt_r1]:
            self.add_latency(s, 1)

        # ==================================================================
        # Stage 1: Wide shift (left or right)
        # ==================================================================
        wide_width = total + wf + 2
        raw_result = Signal(total, name="raw_result")
        ovf = Signal(name="ovf")

        with m.If(do_shift_left_r1):
            wide_result = Signal(wide_width, name="wide_result_l")
            m.d.comb += wide_result.eq(sig_r1 << shift_left_amt_r1[:6])
            m.d.comb += raw_result.eq(wide_result[:total])
            if wide_width > total:
                m.d.comb += ovf.eq(wide_result[total:].any())
            else:
                m.d.comb += ovf.eq(0)
        with m.Else():
            m.d.comb += raw_result.eq(sig_r1 >> shift_right_amt_r1[:6])
            m.d.comb += ovf.eq(0)

        # ── Stage 1 → 2 pipeline register ──
        raw_result_r2 = Signal(total, name="raw_result_r2")
        ovf_r2 = Signal(name="ovf_r2")
        sign_r2 = Signal(name="sign_r2")
        exc_r2 = Signal(2, name="exc_r2")
        m.d.sync += [
            raw_result_r2.eq(raw_result),
            ovf_r2.eq(ovf),
            sign_r2.eq(sign_r1),
            exc_r2.eq(exc_r1),
        ]
        for s in [raw_result_r2, ovf_r2, sign_r2, exc_r2]:
            self.add_latency(s, 2)

        # ==================================================================
        # Stage 2: Sign application, exception mux, pack output
        # ==================================================================
        signed_result = Signal(total, name="signed_result")
        if self.is_signed:
            with m.If(sign_r2):
                m.d.comb += signed_result.eq(-raw_result_r2)
            with m.Else():
                m.d.comb += signed_result.eq(raw_result_r2)
        else:
            m.d.comb += signed_result.eq(raw_result_r2)

        fix_out_comb = Signal(total, name="fix_out_comb")
        overflow_comb = Signal(name="overflow_comb")

        with m.If(exc_r2 == 0b00):  # zero
            m.d.comb += [
                fix_out_comb.eq(0),
                overflow_comb.eq(0),
            ]
        with m.Elif(exc_r2 == 0b01):  # normal
            m.d.comb += [
                fix_out_comb.eq(signed_result),
                overflow_comb.eq(ovf_r2),
            ]
        with m.Else():  # inf or NaN -> overflow
            m.d.comb += [
                fix_out_comb.eq(0),
                overflow_comb.eq(1),
            ]

        # ── Stage 2 → 3 pipeline register (output) ──
        fix_out_r3 = Signal(total, name="fix_out_r3")
        overflow_r3 = Signal(name="overflow_r3")
        m.d.sync += [
            fix_out_r3.eq(fix_out_comb),
            overflow_r3.eq(overflow_comb),
        ]
        m.d.comb += [
            self.fix_out.eq(fix_out_r3),
            self.overflow.eq(overflow_r3),
        ]

        return m
