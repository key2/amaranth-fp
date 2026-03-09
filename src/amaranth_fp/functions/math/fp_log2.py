"""Floating-point log2(x) (pipelined, 7-cycle latency).

Algorithm:
  1. Unpack, handle exceptions (x <= 0, NaN, inf)
  2. Extract exponent e: log2(x) = e + log2(1.f)
  3. Approximate log2(1 + f) via table on [1, 2)
  4. Combine e + log2(1.f) in fixed-point, normalize to FP
  5. Pack result
"""
from __future__ import annotations

import math

from amaranth import *

from ...format import FPFormat, float_to_flopoco
from ...pipelined import PipelinedComponent
from ..table import Table

__all__ = ["FPLog2"]


class FPLog2(PipelinedComponent):
    """Pipelined floating-point log2(x).

    Parameters
    ----------
    fmt : FPFormat
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")
        self.latency = 7

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we, wf, bias = fmt.we, fmt.wf, fmt.bias

        # Pre-encode all possible results as FloPoCo FP
        # For each (exponent, mantissa_top_bits) pair, compute log2(x) and encode
        # Result table: indexed by (e_biased, mant_top_bits)
        table_bits = min(wf, 8)
        table_size = 1 << table_bits
        frac_bits = wf + 4

        # Fixed-point table for log2(1 + i/table_size)
        log2_table = []
        for i in range(table_size):
            f = 1.0 + i / table_size
            val = math.log2(f)  # in [0, 1)
            log2_table.append(int(round(val * (1 << frac_bits))) & ((1 << (frac_bits + 2)) - 1))

        tbl = Table(table_bits, frac_bits + 2, log2_table)
        m.submodules.log2_tbl = tbl

        # Stage 0: Unpack
        a_mant = Signal(wf); a_exp = Signal(we); a_sign = Signal(); a_exc = Signal(2)
        m.d.comb += [
            a_mant.eq(self.a[:wf]), a_exp.eq(self.a[wf:wf + we]),
            a_sign.eq(self.a[wf + we]), a_exc.eq(self.a[wf + we + 1:]),
        ]

        s1_mant = Signal(wf); s1_exp = Signal(we); s1_sign = Signal(); s1_exc = Signal(2)
        m.d.sync += [s1_mant.eq(a_mant), s1_exp.eq(a_exp), s1_sign.eq(a_sign), s1_exc.eq(a_exc)]

        # Stage 1: Exception handling
        s1_is_special = Signal(); s1_exc_out = Signal(2); s1_sign_out = Signal()
        with m.If(s1_exc == 0b11):
            m.d.comb += [s1_exc_out.eq(0b11), s1_sign_out.eq(0), s1_is_special.eq(1)]
        with m.Elif(s1_exc == 0b00):
            m.d.comb += [s1_exc_out.eq(0b10), s1_sign_out.eq(1), s1_is_special.eq(1)]
        with m.Elif(s1_sign):
            m.d.comb += [s1_exc_out.eq(0b11), s1_sign_out.eq(0), s1_is_special.eq(1)]
        with m.Elif(s1_exc == 0b10):
            m.d.comb += [s1_exc_out.eq(0b10), s1_sign_out.eq(0), s1_is_special.eq(1)]
        with m.Else():
            m.d.comb += [s1_exc_out.eq(0b01), s1_sign_out.eq(0), s1_is_special.eq(0)]

        tbl_addr = Signal(table_bits)
        m.d.comb += tbl_addr.eq(s1_mant >> (wf - table_bits) if wf >= table_bits else s1_mant)
        m.d.comb += tbl.addr.eq(tbl_addr)

        e_val = Signal(signed(we + 2))
        m.d.comb += e_val.eq(s1_exp - bias)

        s2_exc = Signal(2); s2_sign = Signal(); s2_special = Signal(); s2_e = Signal(signed(we + 2))
        m.d.sync += [s2_exc.eq(s1_exc_out), s2_sign.eq(s1_sign_out), s2_special.eq(s1_is_special), s2_e.eq(e_val)]

        s3_exc = Signal(2); s3_sign = Signal(); s3_special = Signal(); s3_e = Signal(signed(we + 2))
        m.d.sync += [s3_exc.eq(s2_exc), s3_sign.eq(s2_sign), s3_special.eq(s2_special), s3_e.eq(s2_e)]

        # Stage 3: Table value ready
        tbl_val = Signal(frac_bits + 2)
        m.d.comb += tbl_val.eq(tbl.data)

        # Combine e + frac in fixed-point, then convert to FP at elaboration time
        # is too complex with variable e. Instead, compute the full result value
        # and encode at elaboration using a second lookup.
        #
        # Alternative approach: compute total = e + frac/2^frac_bits as a Python float
        # for each possible (e, frac) combination, but e has too many values.
        #
        # Simpler approach: use fixed-point arithmetic then normalize
        # total_fixed = e * 2^frac_bits + tbl_val (signed fixed-point with frac_bits fractional bits)
        int_bits = we + 2
        total_bits = int_bits + frac_bits
        total_fixed = Signal(signed(total_bits + 1))
        m.d.comb += total_fixed.eq((s3_e << frac_bits) + tbl_val)

        abs_fixed = Signal(total_bits + 1)
        result_sign = Signal()
        with m.If(total_fixed < 0):
            m.d.comb += [abs_fixed.eq(-total_fixed), result_sign.eq(1)]
        with m.Else():
            m.d.comb += [abs_fixed.eq(total_fixed), result_sign.eq(0)]

        # Find leading one: iterate LOW to HIGH so last match (highest bit) wins
        lead_pos = Signal(range(total_bits + 2))
        for i in range(total_bits + 1):
            with m.If(abs_fixed[i]):
                m.d.comb += lead_pos.eq(i)

        result_mant = Signal(wf)
        result_exp = Signal(we)
        result_exc = Signal(2)

        with m.If(s3_special):
            m.d.comb += [result_exc.eq(s3_exc), result_mant.eq(0), result_exp.eq(0)]
        with m.Elif(total_fixed == 0):
            m.d.comb += [result_exc.eq(0b00), result_mant.eq(0), result_exp.eq(0)]
        with m.Else():
            # FP exponent: value = abs_fixed / 2^frac_bits
            # If leading bit at position p, value ≈ 2^(p - frac_bits)
            # FP: exp_unbiased = p - frac_bits
            fp_exp_biased = Signal(signed(we + 4))
            m.d.comb += fp_exp_biased.eq(lead_pos - frac_bits + bias)

            # Extract wf mantissa bits below the leading 1
            rshift = Signal(range(total_bits + 2))
            lshift = Signal(range(wf + 1))
            m.d.comb += rshift.eq(lead_pos - wf)
            m.d.comb += lshift.eq(wf - lead_pos)
            with m.If(lead_pos > wf):
                m.d.comb += result_mant.eq((abs_fixed >> rshift) & ((1 << wf) - 1))
            with m.Elif(lead_pos == wf):
                m.d.comb += result_mant.eq(abs_fixed[:wf])
            with m.Else():
                m.d.comb += result_mant.eq((abs_fixed << lshift) & ((1 << wf) - 1))

            with m.If(fp_exp_biased >= (1 << we) - 1):
                m.d.comb += [result_exc.eq(0b10), result_exp.eq(0)]
            with m.Elif(fp_exp_biased < 1):
                m.d.comb += [result_exc.eq(0b00), result_exp.eq(0)]
            with m.Else():
                m.d.comb += [result_exc.eq(0b01), result_exp.eq(fp_exp_biased[:we])]

        # Stage 3 → 4
        s4_exc = Signal(2); s4_sign = Signal(); s4_mant = Signal(wf); s4_exp = Signal(we)
        m.d.sync += [s4_exc.eq(result_exc),
                     s4_sign.eq(Mux(s3_special, s3_sign, result_sign)),
                     s4_mant.eq(result_mant), s4_exp.eq(result_exp)]

        # Stage 4 → 5
        s5 = Signal(fmt.width)
        m.d.sync += s5.eq(Cat(s4_mant, s4_exp, s4_sign, s4_exc))

        # Stage 5 → 6
        s6 = Signal(fmt.width)
        m.d.sync += s6.eq(s5)

        # Stage 6 → 7 (output)
        s7 = Signal(fmt.width)
        m.d.sync += s7.eq(s6)
        m.d.comb += self.o.eq(s7)

        return m
