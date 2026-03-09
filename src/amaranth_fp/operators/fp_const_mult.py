"""Floating-point constant multiplier (pipelined, 3 stages).

Multiplies an FP input by a compile-time constant. Simpler than general FPMul
since one operand is known at elaboration time.
"""
from __future__ import annotations

import math
import struct

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent

__all__ = ["FPConstMult"]


class FPConstMult(PipelinedComponent):
    """Multiply FP value by compile-time constant.

    Parameters
    ----------
    fmt : FPFormat
    constant : float
    """

    def __init__(self, fmt: FPFormat, constant: float) -> None:
        super().__init__()
        self.fmt = fmt
        self.constant = constant
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")
        self.latency = 3

        # Decompose constant
        if constant == 0.0:
            self._const_sign = 0
            self._const_exp = 0
            self._const_mant = 0
            self._const_exc = 0b00
        elif math.isinf(constant):
            self._const_sign = 1 if constant < 0 else 0
            self._const_exp = 0
            self._const_mant = 0
            self._const_exc = 0b10
        elif math.isnan(constant):
            self._const_sign = 0
            self._const_exp = 0
            self._const_mant = 0
            self._const_exc = 0b11
        else:
            self._const_sign = 1 if constant < 0 else 0
            abs_c = abs(constant)
            # Compute exponent and mantissa
            if abs_c >= 2.0 ** (fmt.bias + 1):
                self._const_exc = 0b10  # overflow to inf
                self._const_exp = 0
                self._const_mant = 0
            else:
                exp_val = int(math.floor(math.log2(abs_c)))
                biased_exp = exp_val + fmt.bias
                if biased_exp <= 0:
                    self._const_exc = 0b00  # underflow to zero
                    self._const_exp = 0
                    self._const_mant = 0
                else:
                    self._const_exc = 0b01
                    self._const_exp = biased_exp
                    sig = abs_c / (2.0 ** exp_val)  # 1.xxx
                    self._const_mant = int(round((sig - 1.0) * (1 << fmt.wf))) & ((1 << fmt.wf) - 1)

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we = fmt.we
        wf = fmt.wf
        bias = fmt.bias

        # ==================================================================
        # Stage 0: Unpack input
        # ==================================================================
        a_mant = Signal(wf, name="a_mant")
        a_exp = Signal(we, name="a_exp")
        a_sign = Signal(name="a_sign")
        a_exc = Signal(2, name="a_exc")
        m.d.comb += [
            a_mant.eq(self.a[:wf]),
            a_exp.eq(self.a[wf:wf + we]),
            a_sign.eq(self.a[wf + we]),
            a_exc.eq(self.a[wf + we + 1:]),
        ]

        result_sign = Signal(name="result_sign")
        m.d.comb += result_sign.eq(a_sign ^ self._const_sign)

        # Exception logic
        exc_result = Signal(2, name="exc_result")
        c_exc = self._const_exc
        with m.Switch(Cat(Const(c_exc, 2), a_exc)):
            with m.Case(0b0000): m.d.comb += exc_result.eq(0b00)  # 0*0=0
            with m.Case(0b0001): m.d.comb += exc_result.eq(0b00)  # 0*normal=0
            with m.Case(0b0100): m.d.comb += exc_result.eq(0b00)  # normal*0=0
            with m.Case(0b0101): m.d.comb += exc_result.eq(0b01)  # normal*normal
            with m.Case(0b0110): m.d.comb += exc_result.eq(0b10)  # normal*inf
            with m.Case(0b1001): m.d.comb += exc_result.eq(0b10)  # inf*normal
            with m.Case(0b1010): m.d.comb += exc_result.eq(0b10)  # inf*inf
            with m.Default():    m.d.comb += exc_result.eq(0b11)  # NaN

        # Exponent add
        exp_sum = Signal(we + 2, name="exp_sum")
        m.d.comb += exp_sum.eq(a_exp + self._const_exp - bias)

        # Mantissa multiply
        sig_a = Signal(wf + 1, name="sig_a")
        m.d.comb += sig_a.eq(Cat(a_mant, Const(1, 1)))
        const_sig = (1 << wf) | self._const_mant

        prod_w = 2 * (wf + 1)
        sig_prod = Signal(prod_w, name="sig_prod")
        m.d.comb += sig_prod.eq(sig_a * const_sig)

        # Stage 0 → 1
        exc_r1 = Signal(2, name="exc_r1")
        sign_r1 = Signal(name="sign_r1")
        exp_r1 = Signal(we + 2, name="exp_r1")
        prod_r1 = Signal(prod_w, name="prod_r1")
        m.d.sync += [
            exc_r1.eq(exc_result), sign_r1.eq(result_sign),
            exp_r1.eq(exp_sum), prod_r1.eq(sig_prod),
        ]

        # ==================================================================
        # Stage 1: Normalize product
        # ==================================================================
        norm = Signal(name="norm")
        m.d.comb += norm.eq(prod_r1[prod_w - 1])

        exp_post = Signal(we + 2, name="exp_post")
        m.d.comb += exp_post.eq(exp_r1 + norm)

        result_mant = Signal(wf, name="result_mant")
        with m.If(norm):
            m.d.comb += result_mant.eq(prod_r1[prod_w - 1 - wf:prod_w - 1])
        with m.Else():
            m.d.comb += result_mant.eq(prod_r1[prod_w - 2 - wf:prod_w - 2])

        # Stage 1 → 2
        exc_r2 = Signal(2, name="exc_r2")
        sign_r2 = Signal(name="sign_r2")
        exp_r2 = Signal(we + 2, name="exp_r2")
        mant_r2 = Signal(wf, name="mant_r2")
        m.d.sync += [
            exc_r2.eq(exc_r1), sign_r2.eq(sign_r1),
            exp_r2.eq(exp_post), mant_r2.eq(result_mant),
        ]

        # ==================================================================
        # Stage 2: Overflow/underflow, pack
        # ==================================================================
        final_exc = Signal(2, name="final_exc")
        final_sign = Signal(name="final_sign")
        final_mant = Signal(wf, name="final_mant")
        final_exp = Signal(we, name="final_exp")

        max_exp = (1 << we) - 1

        with m.If((exc_r2 != 0b01)):
            m.d.comb += [
                final_exc.eq(exc_r2), final_sign.eq(sign_r2),
                final_mant.eq(0), final_exp.eq(0),
            ]
        with m.Elif(exp_r2[we + 1]):  # negative = underflow
            m.d.comb += [
                final_exc.eq(0b00), final_sign.eq(0),
                final_mant.eq(0), final_exp.eq(0),
            ]
        with m.Elif(exp_r2[:we + 1] >= max_exp):
            m.d.comb += [
                final_exc.eq(0b10), final_sign.eq(sign_r2),
                final_mant.eq(0), final_exp.eq(0),
            ]
        with m.Else():
            m.d.comb += [
                final_exc.eq(0b01), final_sign.eq(sign_r2),
                final_mant.eq(mant_r2), final_exp.eq(exp_r2[:we]),
            ]

        o_r3 = Signal(fmt.width, name="o_r3")
        m.d.sync += o_r3.eq(Cat(final_mant, final_exp, final_sign, final_exc))
        m.d.comb += self.o.eq(o_r3)

        return m
