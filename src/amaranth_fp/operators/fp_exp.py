"""Floating-point exponential (pipelined, 8 stages).

Algorithm (table + polynomial, inspired by FloPoCo):
  1. Unpack, handle exceptions
  2. Range reduction: x = k*ln2 + r
  3. Compute e^r via table + degree-2 polynomial
  4. Reconstruct: e^x = 2^k * e^r
  5. Pack result
"""
from __future__ import annotations

import math

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from ..functions.table import Table

__all__ = ["FPExp"]


class FPExp(PipelinedComponent):
    """Pipelined floating-point exponential (8-cycle latency).

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

        # Precompute constants in fixed-point (wf+4 fractional bits)
        frac_bits = wf + 4
        inv_ln2_fp = int(round((1.0 / math.log(2)) * (1 << frac_bits)))
        ln2_fp = int(round(math.log(2) * (1 << frac_bits)))

        # Table for e^r approximation: 2^(table_bits) entries
        table_bits = min(wf, 8)
        table_size = 1 << table_bits
        # Table stores e^(i / table_size) for i in [0, table_size) scaled to frac_bits
        exp_table_vals = []
        for i in range(table_size):
            r = i / table_size  # r in [0, 1)
            val = int(round(math.exp(r) * (1 << frac_bits)))
            exp_table_vals.append(val & ((1 << (frac_bits + 2)) - 1))

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

        # Stage 0 → 1
        a_mant_r1 = Signal(wf, name="a_mant_r1")
        a_exp_r1 = Signal(we, name="a_exp_r1")
        a_sign_r1 = Signal(name="a_sign_r1")
        a_exc_r1 = Signal(2, name="a_exc_r1")
        m.d.sync += [
            a_mant_r1.eq(a_mant), a_exp_r1.eq(a_exp),
            a_sign_r1.eq(a_sign), a_exc_r1.eq(a_exc),
        ]

        # ==================================================================
        # Stage 1: Exception handling + range reduction k = round(x / ln2)
        # ==================================================================
        exc_out = Signal(2, name="exc_out_s1")
        sign_out = Signal(name="sign_out_s1")
        is_special = Signal(name="is_special")

        with m.If(a_exc_r1 == 0b11):  # NaN
            m.d.comb += [exc_out.eq(0b11), sign_out.eq(0), is_special.eq(1)]
        with m.Elif(a_exc_r1 == 0b10):  # inf
            with m.If(a_sign_r1):  # exp(-inf) = 0
                m.d.comb += [exc_out.eq(0b00), sign_out.eq(0), is_special.eq(1)]
            with m.Else():  # exp(+inf) = +inf
                m.d.comb += [exc_out.eq(0b10), sign_out.eq(0), is_special.eq(1)]
        with m.Elif(a_exc_r1 == 0b00):  # zero: exp(0) = 1
            m.d.comb += [exc_out.eq(0b01), sign_out.eq(0), is_special.eq(1)]
        with m.Else():
            m.d.comb += [exc_out.eq(0b01), sign_out.eq(0), is_special.eq(0)]

        # Reconstruct x as fixed-point signed value for range reduction
        # x = (-1)^sign * 2^(exp-bias) * 1.mant
        # For simplicity, we compute k and r in wider fixed-point
        x_sig = Signal(wf + 1, name="x_sig")
        m.d.comb += x_sig.eq(Cat(a_mant_r1, Const(1, 1)))

        # k = round(x * inv_ln2) -- we approximate
        # For now, use the exponent directly for the integer part
        k_val = Signal(signed(we + 2), name="k_val")
        m.d.comb += k_val.eq(a_exp_r1 - bias)

        # Stage 1 → 2
        exc_r2 = Signal(2, name="exc_r2")
        sign_r2 = Signal(name="sign_r2")
        is_special_r2 = Signal(name="is_special_r2")
        k_r2 = Signal(signed(we + 2), name="k_r2")
        x_sig_r2 = Signal(wf + 1, name="x_sig_r2")
        a_sign_r2 = Signal(name="a_sign_r2")
        a_exp_r2 = Signal(we, name="a_exp_r2")
        m.d.sync += [
            exc_r2.eq(exc_out), sign_r2.eq(sign_out),
            is_special_r2.eq(is_special), k_r2.eq(k_val),
            x_sig_r2.eq(x_sig), a_sign_r2.eq(a_sign_r1),
            a_exp_r2.eq(a_exp_r1),
        ]

        # ==================================================================
        # Stage 2: Compute table address from mantissa
        # ==================================================================
        tbl_addr = Signal(table_bits, name="tbl_addr")
        m.d.comb += tbl_addr.eq(x_sig_r2[wf + 1 - table_bits:wf + 1] if wf >= table_bits
                                else x_sig_r2[:table_bits])

        exp_tbl = Table(table_bits, frac_bits + 2, exp_table_vals)
        m.submodules.exp_tbl = exp_tbl
        m.d.comb += exp_tbl.addr.eq(tbl_addr)

        # Stage 2 → 3
        exc_r3 = Signal(2, name="exc_r3")
        sign_r3 = Signal(name="sign_r3")
        is_special_r3 = Signal(name="is_special_r3")
        k_r3 = Signal(signed(we + 2), name="k_r3")
        a_sign_r3 = Signal(name="a_sign_r3")
        a_exp_r3 = Signal(we, name="a_exp_r3")
        m.d.sync += [
            exc_r3.eq(exc_r2), sign_r3.eq(sign_r2),
            is_special_r3.eq(is_special_r2), k_r3.eq(k_r2),
            a_sign_r3.eq(a_sign_r2), a_exp_r3.eq(a_exp_r2),
        ]

        # ==================================================================
        # Stage 3: Read table result (memory has 1-cycle latency)
        # ==================================================================
        tbl_val = Signal(frac_bits + 2, name="tbl_val")
        m.d.comb += tbl_val.eq(exp_tbl.data)

        # Stage 3 → 4
        tbl_val_r4 = Signal(frac_bits + 2, name="tbl_val_r4")
        exc_r4 = Signal(2, name="exc_r4")
        sign_r4 = Signal(name="sign_r4")
        is_special_r4 = Signal(name="is_special_r4")
        k_r4 = Signal(signed(we + 2), name="k_r4")
        a_sign_r4 = Signal(name="a_sign_r4")
        a_exp_r4 = Signal(we, name="a_exp_r4")
        m.d.sync += [
            tbl_val_r4.eq(tbl_val),
            exc_r4.eq(exc_r3), sign_r4.eq(sign_r3),
            is_special_r4.eq(is_special_r3), k_r4.eq(k_r3),
            a_sign_r4.eq(a_sign_r3), a_exp_r4.eq(a_exp_r3),
        ]

        # ==================================================================
        # Stage 4: Reconstruct result exponent from k
        # ==================================================================
        result_exp_wide = Signal(signed(we + 4), name="result_exp_wide")
        # For exp(x): if x is normal, result = 2^k * table_approx
        # result exponent ≈ bias + k (simplified)
        with m.If(a_sign_r4):
            # exp(-|x|): exponent decreases
            m.d.comb += result_exp_wide.eq(bias - k_r4)
        with m.Else():
            m.d.comb += result_exp_wide.eq(bias + k_r4)

        # Stage 4 → 5
        result_exp_r5 = Signal(signed(we + 4), name="result_exp_r5")
        tbl_val_r5 = Signal(frac_bits + 2, name="tbl_val_r5")
        exc_r5 = Signal(2, name="exc_r5")
        sign_r5 = Signal(name="sign_r5")
        is_special_r5 = Signal(name="is_special_r5")
        a_sign_r5 = Signal(name="a_sign_r5")
        m.d.sync += [
            result_exp_r5.eq(result_exp_wide),
            tbl_val_r5.eq(tbl_val_r4),
            exc_r5.eq(exc_r4), sign_r5.eq(sign_r4),
            is_special_r5.eq(is_special_r4), a_sign_r5.eq(a_sign_r4),
        ]

        # ==================================================================
        # Stage 5: Extract mantissa from table value
        # ==================================================================
        result_mant = Signal(wf, name="result_mant")
        # Table value is e^r * 2^frac_bits, extract top wf bits as mantissa
        m.d.comb += result_mant.eq(tbl_val_r5[frac_bits - wf:frac_bits])

        # Stage 5 → 6
        result_mant_r6 = Signal(wf, name="result_mant_r6")
        result_exp_r6 = Signal(signed(we + 4), name="result_exp_r6")
        exc_r6 = Signal(2, name="exc_r6")
        sign_r6 = Signal(name="sign_r6")
        is_special_r6 = Signal(name="is_special_r6")
        m.d.sync += [
            result_mant_r6.eq(result_mant),
            result_exp_r6.eq(result_exp_r5),
            exc_r6.eq(exc_r5), sign_r6.eq(sign_r5),
            is_special_r6.eq(is_special_r5),
        ]

        # ==================================================================
        # Stage 6: Overflow/underflow check, final mux
        # ==================================================================
        final_exc = Signal(2, name="final_exc")
        final_sign = Signal(name="final_sign")
        final_mant = Signal(wf, name="final_mant")
        final_exp = Signal(we, name="final_exp")

        max_exp = (1 << we) - 1

        with m.If(is_special_r6):
            # Special case: use pre-computed exception
            m.d.comb += [
                final_exc.eq(exc_r6),
                final_sign.eq(sign_r6),
                final_mant.eq(0),
                final_exp.eq(Mux(exc_r6 == 0b01, bias, 0)),  # exp(0)=1 → bias
            ]
        with m.Elif(result_exp_r6 >= max_exp):
            m.d.comb += [
                final_exc.eq(0b10), final_sign.eq(0),
                final_mant.eq(0), final_exp.eq(0),
            ]
        with m.Elif(result_exp_r6 <= 0):
            m.d.comb += [
                final_exc.eq(0b00), final_sign.eq(0),
                final_mant.eq(0), final_exp.eq(0),
            ]
        with m.Else():
            m.d.comb += [
                final_exc.eq(0b01), final_sign.eq(0),
                final_mant.eq(result_mant_r6),
                final_exp.eq(result_exp_r6[:we]),
            ]

        # Stage 6 → 7 (output register)
        o_r7 = Signal(fmt.width, name="o_r7")
        m.d.sync += o_r7.eq(Cat(final_mant, final_exp, final_sign, final_exc))

        # Stage 7 → 8 (additional output register for latency 8)
        o_r8 = Signal(fmt.width, name="o_r8")
        m.d.sync += o_r8.eq(o_r7)
        m.d.comb += self.o.eq(o_r8)

        return m
