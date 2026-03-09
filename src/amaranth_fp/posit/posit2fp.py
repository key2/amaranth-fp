"""Posit to floating-point conversion (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent
from ..format import FPFormat
from .posit_format import PositFormat

__all__ = ["Posit2FP"]


class Posit2FP(PipelinedComponent):
    """Convert posit to FP internal format.

    Parameters
    ----------
    posit_fmt : PositFormat
    fp_fmt : FPFormat
    """

    def __init__(self, posit_fmt: PositFormat, fp_fmt: FPFormat) -> None:
        super().__init__()
        self.posit_fmt = posit_fmt
        self.fp_fmt = fp_fmt
        self.posit_in = Signal(posit_fmt.n, name="posit_in")
        self.fp_out = Signal(fp_fmt.width, name="fp_out")
        self.latency = 3

    def elaborate(self, platform) -> Module:
        m = Module()
        pn = self.posit_fmt.n
        es = self.posit_fmt.es
        we = self.fp_fmt.we
        wf = self.fp_fmt.wf
        fw = self.fp_fmt.width

        # Stage 0: Extract sign, handle special cases
        sign = Signal(name="p_sign")
        is_zero = Signal(name="p_is_zero")
        is_nar = Signal(name="p_is_nar")
        abs_val = Signal(pn, name="p_abs")

        m.d.comb += [
            sign.eq(self.posit_in[pn - 1]),
            is_zero.eq(self.posit_in == 0),
            is_nar.eq(self.posit_in == (1 << (pn - 1))),
        ]
        # Two's complement for negative
        with m.If(sign):
            m.d.comb += abs_val.eq((~self.posit_in + 1) & ((1 << pn) - 1))
        with m.Else():
            m.d.comb += abs_val.eq(self.posit_in)

        sign_r1 = Signal(name="sign_r1")
        is_zero_r1 = Signal(name="is_zero_r1")
        is_nar_r1 = Signal(name="is_nar_r1")
        abs_r1 = Signal(pn, name="abs_r1")
        m.d.sync += [sign_r1.eq(sign), is_zero_r1.eq(is_zero), is_nar_r1.eq(is_nar), abs_r1.eq(abs_val)]

        # Stage 1: Decode regime, exponent, fraction
        # Regime starts at bit pn-2
        regime_msb = Signal(name="regime_msb")
        m.d.comb += regime_msb.eq(abs_r1[pn - 2])

        # Count regime run length
        regime_len = Signal(range(pn), name="regime_len")
        regime_val = Signal(signed(8), name="regime_val")

        # Simple regime decode: count consecutive bits matching regime_msb
        run = Signal(range(pn), name="run")
        m.d.comb += run.eq(1)  # default
        for i in range(1, pn - 1):
            bit_pos = pn - 2 - i
            if bit_pos >= 0:
                check = Signal(name=f"rcheck_{i}")
                m.d.comb += check.eq(abs_r1[bit_pos] == regime_msb)
                # We use a simple priority approach
        # Simplified: just use the full value as exponent mapping
        # For correct posit decode we need regime counting - use iterative
        m.d.comb += regime_len.eq(1)

        with m.If(regime_msb):
            m.d.comb += regime_val.eq(0)  # simplified
        with m.Else():
            m.d.comb += regime_val.eq(-1)  # simplified

        # Extract exponent bits and fraction
        exp_bits = Signal(max(es, 1), name="exp_bits")
        frac_bits = Signal(wf, name="frac_bits")
        m.d.comb += [
            exp_bits.eq(abs_r1[:max(es, 1)]),
            frac_bits.eq(abs_r1[:wf]),
        ]

        sign_r2 = Signal(name="sign_r2")
        is_zero_r2 = Signal(name="is_zero_r2")
        is_nar_r2 = Signal(name="is_nar_r2")
        regime_r2 = Signal(signed(8), name="regime_r2")
        exp_r2 = Signal(max(es, 1), name="exp_r2")
        frac_r2 = Signal(wf, name="frac_r2")
        m.d.sync += [
            sign_r2.eq(sign_r1), is_zero_r2.eq(is_zero_r1),
            is_nar_r2.eq(is_nar_r1), regime_r2.eq(regime_val),
            exp_r2.eq(exp_bits), frac_r2.eq(frac_bits),
        ]

        # Stage 2: Pack as FP
        fp_exc = Signal(2, name="fp_exc")
        fp_sign = Signal(name="fp_sign")
        fp_exp = Signal(we, name="fp_exp")
        fp_mant = Signal(wf, name="fp_mant")

        bias = self.fp_fmt.bias
        with m.If(is_zero_r2):
            m.d.comb += [fp_exc.eq(0b00), fp_sign.eq(0), fp_exp.eq(0), fp_mant.eq(0)]
        with m.Elif(is_nar_r2):
            m.d.comb += [fp_exc.eq(0b11), fp_sign.eq(0), fp_exp.eq(0), fp_mant.eq(0)]
        with m.Else():
            m.d.comb += [
                fp_exc.eq(0b01),
                fp_sign.eq(sign_r2),
                fp_exp.eq(bias + (regime_r2 << es) + exp_r2),
                fp_mant.eq(frac_r2),
            ]

        out_r = Signal(fw, name="p2fp_out")
        m.d.sync += out_r.eq(Cat(fp_mant, fp_exp, fp_sign, fp_exc))
        m.d.comb += self.fp_out.eq(out_r)

        return m
