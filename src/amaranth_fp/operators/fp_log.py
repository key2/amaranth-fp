"""Floating-point natural logarithm (pipelined, 8 stages).

Algorithm:
  1. Unpack, handle exceptions
  2. Decompose: x = 2^e * 1.f → log(x) = e*ln2 + log(1.f)
  3. Compute log(1.f) via table lookup
  4. Add e*ln2
  5. Pack result
"""
from __future__ import annotations

import math

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from ..functions.table import Table

__all__ = ["FPLog"]


class FPLog(PipelinedComponent):
    """Pipelined floating-point natural logarithm (8-cycle latency).

    Parameters
    ----------
    fmt : FPFormat
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")
        self.latency = 8

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we = fmt.we
        wf = fmt.wf
        bias = fmt.bias

        frac_bits = wf + 4
        ln2_fp = int(round(math.log(2) * (1 << frac_bits)))

        # Table for log(1 + i/table_size) scaled
        table_bits = min(wf, 8)
        table_size = 1 << table_bits
        log_table_vals = []
        for i in range(table_size):
            x = 1.0 + i / table_size
            val = int(round(math.log(x) * (1 << frac_bits)))
            log_table_vals.append(val & ((1 << (frac_bits + 2)) - 1))

        # ==================================================================
        # Stage 0: Unpack
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

        a_mant_r1 = Signal(wf, name="a_mant_r1")
        a_exp_r1 = Signal(we, name="a_exp_r1")
        a_sign_r1 = Signal(name="a_sign_r1")
        a_exc_r1 = Signal(2, name="a_exc_r1")
        m.d.sync += [
            a_mant_r1.eq(a_mant), a_exp_r1.eq(a_exp),
            a_sign_r1.eq(a_sign), a_exc_r1.eq(a_exc),
        ]

        # ==================================================================
        # Stage 1: Exception handling
        # ==================================================================
        exc_out = Signal(2, name="exc_out_s1")
        sign_out = Signal(name="sign_out_s1")
        is_special = Signal(name="is_special")

        with m.If(a_exc_r1 == 0b11):  # NaN
            m.d.comb += [exc_out.eq(0b11), sign_out.eq(0), is_special.eq(1)]
        with m.Elif(a_exc_r1 == 0b10):  # inf
            with m.If(a_sign_r1):  # log(-inf) = NaN
                m.d.comb += [exc_out.eq(0b11), sign_out.eq(0), is_special.eq(1)]
            with m.Else():  # log(+inf) = +inf
                m.d.comb += [exc_out.eq(0b10), sign_out.eq(0), is_special.eq(1)]
        with m.Elif(a_exc_r1 == 0b00):  # zero: log(0) = -inf
            m.d.comb += [exc_out.eq(0b10), sign_out.eq(1), is_special.eq(1)]
        with m.Elif(a_sign_r1):  # negative normal: log(negative) = NaN
            m.d.comb += [exc_out.eq(0b11), sign_out.eq(0), is_special.eq(1)]
        with m.Else():
            m.d.comb += [exc_out.eq(0b01), sign_out.eq(0), is_special.eq(0)]

        # e = exponent - bias (signed)
        e_val = Signal(signed(we + 2), name="e_val")
        m.d.comb += e_val.eq(a_exp_r1 - bias)

        # Stage 1 → 2
        exc_r2 = Signal(2, name="exc_r2")
        sign_r2 = Signal(name="sign_r2")
        is_special_r2 = Signal(name="is_special_r2")
        e_r2 = Signal(signed(we + 2), name="e_r2")
        mant_r2 = Signal(wf, name="mant_r2")
        m.d.sync += [
            exc_r2.eq(exc_out), sign_r2.eq(sign_out),
            is_special_r2.eq(is_special), e_r2.eq(e_val),
            mant_r2.eq(a_mant_r1),
        ]

        # ==================================================================
        # Stage 2: Table lookup for log(1.f)
        # ==================================================================
        tbl_addr = Signal(table_bits, name="log_tbl_addr")
        if wf >= table_bits:
            m.d.comb += tbl_addr.eq(mant_r2[wf - table_bits:wf])
        else:
            m.d.comb += tbl_addr.eq(mant_r2[:table_bits])

        log_tbl = Table(table_bits, frac_bits + 2, log_table_vals)
        m.submodules.log_tbl = log_tbl
        m.d.comb += log_tbl.addr.eq(tbl_addr)

        # Stage 2 → 3
        exc_r3 = Signal(2, name="exc_r3")
        sign_r3 = Signal(name="sign_r3")
        is_special_r3 = Signal(name="is_special_r3")
        e_r3 = Signal(signed(we + 2), name="e_r3")
        m.d.sync += [
            exc_r3.eq(exc_r2), sign_r3.eq(sign_r2),
            is_special_r3.eq(is_special_r2), e_r3.eq(e_r2),
        ]

        # ==================================================================
        # Stage 3: Read table (memory latency)
        # ==================================================================
        log1f = Signal(frac_bits + 2, name="log1f")
        m.d.comb += log1f.eq(log_tbl.data)

        exc_r4 = Signal(2, name="exc_r4")
        sign_r4 = Signal(name="sign_r4")
        is_special_r4 = Signal(name="is_special_r4")
        e_r4 = Signal(signed(we + 2), name="e_r4")
        log1f_r4 = Signal(frac_bits + 2, name="log1f_r4")
        m.d.sync += [
            exc_r4.eq(exc_r3), sign_r4.eq(sign_r3),
            is_special_r4.eq(is_special_r3), e_r4.eq(e_r3),
            log1f_r4.eq(log1f),
        ]

        # ==================================================================
        # Stage 4: Compute e * ln2
        # ==================================================================
        e_ln2 = Signal(signed(we + 2 + frac_bits + 1), name="e_ln2")
        m.d.comb += e_ln2.eq(e_r4 * ln2_fp)

        exc_r5 = Signal(2, name="exc_r5")
        sign_r5 = Signal(name="sign_r5")
        is_special_r5 = Signal(name="is_special_r5")
        e_ln2_r5 = Signal(signed(we + 2 + frac_bits + 1), name="e_ln2_r5")
        log1f_r5 = Signal(frac_bits + 2, name="log1f_r5")
        m.d.sync += [
            exc_r5.eq(exc_r4), sign_r5.eq(sign_r4),
            is_special_r5.eq(is_special_r4),
            e_ln2_r5.eq(e_ln2), log1f_r5.eq(log1f_r4),
        ]

        # ==================================================================
        # Stage 5: Add log(1.f) + e*ln2 = log(x) in fixed-point
        # ==================================================================
        log_result_w = we + 2 + frac_bits + 2
        log_result = Signal(signed(log_result_w), name="log_result")
        m.d.comb += log_result.eq(e_ln2_r5 + log1f_r5)

        # Determine sign of result
        res_negative = Signal(name="res_negative")
        m.d.comb += res_negative.eq(log_result[log_result_w - 1])

        abs_log = Signal(log_result_w, name="abs_log")
        with m.If(res_negative):
            m.d.comb += abs_log.eq(-log_result)
        with m.Else():
            m.d.comb += abs_log.eq(log_result)

        exc_r6 = Signal(2, name="exc_r6")
        sign_r6_s = Signal(name="sign_r6_s")
        is_special_r6 = Signal(name="is_special_r6")
        abs_log_r6 = Signal(log_result_w, name="abs_log_r6")
        res_neg_r6 = Signal(name="res_neg_r6")
        m.d.sync += [
            exc_r6.eq(exc_r5), sign_r6_s.eq(sign_r5),
            is_special_r6.eq(is_special_r5),
            abs_log_r6.eq(abs_log), res_neg_r6.eq(res_negative),
        ]

        # ==================================================================
        # Stage 6: Convert fixed-point log to FP (find leading 1, extract mantissa)
        # ==================================================================
        # Find position of leading 1 in abs_log_r6
        lzc_val = Signal(range(log_result_w + 1), name="lzc_val")
        # Simple priority encoder
        m.d.comb += lzc_val.eq(0)
        for i in range(log_result_w):
            with m.If(abs_log_r6[i]):
                m.d.comb += lzc_val.eq(i)

        # Exponent of the result FP: the leading bit position minus frac_bits gives
        # the true exponent, then add bias
        result_exp_pre = Signal(signed(we + 4), name="result_exp_pre")
        m.d.comb += result_exp_pre.eq(lzc_val - frac_bits + bias)

        result_mant = Signal(wf, name="result_mant")
        # Shift abs_log so leading 1 is at bit lzc_val, extract wf bits below it
        shifted = Signal(log_result_w, name="log_shifted")
        shift_amt = Signal(range(log_result_w + 1), name="log_shift_amt")
        m.d.comb += shift_amt.eq(lzc_val)
        # We want bits [lzc_val-1 : lzc_val-wf] from abs_log_r6
        # Shift left so bit lzc_val becomes MSB, then take top wf bits
        left_shift = Signal(range(log_result_w), name="log_left_shift")
        m.d.comb += left_shift.eq(log_result_w - 1 - shift_amt)
        m.d.comb += shifted.eq(abs_log_r6 << left_shift)
        m.d.comb += result_mant.eq(shifted[log_result_w - 1 - wf:log_result_w - 1])

        final_exc = Signal(2, name="final_exc")
        final_sign = Signal(name="final_sign")
        final_mant = Signal(wf, name="final_mant")
        final_exp = Signal(we, name="final_exp")

        with m.If(is_special_r6):
            m.d.comb += [
                final_exc.eq(exc_r6), final_sign.eq(sign_r6_s),
                final_mant.eq(0), final_exp.eq(0),
            ]
        with m.Elif(abs_log_r6 == 0):  # log(1) = 0
            m.d.comb += [
                final_exc.eq(0b00), final_sign.eq(0),
                final_mant.eq(0), final_exp.eq(0),
            ]
        with m.Elif((result_exp_pre <= 0) | (result_exp_pre >= ((1 << we) - 1))):
            # Overflow/underflow
            with m.If(result_exp_pre >= ((1 << we) - 1)):
                m.d.comb += [
                    final_exc.eq(0b10), final_sign.eq(res_neg_r6),
                    final_mant.eq(0), final_exp.eq(0),
                ]
            with m.Else():
                m.d.comb += [
                    final_exc.eq(0b00), final_sign.eq(0),
                    final_mant.eq(0), final_exp.eq(0),
                ]
        with m.Else():
            m.d.comb += [
                final_exc.eq(0b01), final_sign.eq(res_neg_r6),
                final_mant.eq(result_mant), final_exp.eq(result_exp_pre[:we]),
            ]

        # Stage 6 → 7
        o_r7 = Signal(fmt.width, name="o_r7")
        m.d.sync += o_r7.eq(Cat(final_mant, final_exp, final_sign, final_exc))

        # Stage 7 → 8
        o_r8 = Signal(fmt.width, name="o_r8")
        m.d.sync += o_r8.eq(o_r7)
        m.d.comb += self.o.eq(o_r8)

        return m
