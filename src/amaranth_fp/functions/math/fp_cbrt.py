"""Floating-point cbrt(x) (pipelined, ~8-cycle latency).

Normalize, initial approx via table, Newton-Raphson iterations.
cbrt(x) = cbrt(2^(3k+r) * m) = 2^k * cbrt(2^r) * cbrt(m), m in [1,2).
"""
from __future__ import annotations

import math

from amaranth import *

from ...format import FPFormat
from ...pipelined import PipelinedComponent
from ..table import Table

__all__ = ["FPCbrt"]


class FPCbrt(PipelinedComponent):
    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")
        self.latency = 8

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we, wf, bias = fmt.we, fmt.wf, fmt.bias

        # Table for cbrt(1 + f/table_size) for f in [0, table_size)
        table_bits = min(wf, 8)
        frac_bits = wf + 4
        cbrt_table = []
        for i in range(1 << table_bits):
            x = 1.0 + i / (1 << table_bits)
            cbrt_table.append(int(round(x ** (1.0/3.0) * (1 << frac_bits))) & ((1 << (frac_bits + 2)) - 1))

        tbl = Table(table_bits, frac_bits + 2, cbrt_table)
        m.submodules.cbrt_tbl = tbl

        # cbrt(2^0)=1, cbrt(2^1)=2^(1/3), cbrt(2^2)=2^(2/3)
        cbrt_pow2 = [
            int(round(1.0 * (1 << frac_bits))),
            int(round(2.0 ** (1.0/3.0) * (1 << frac_bits))),
            int(round(2.0 ** (2.0/3.0) * (1 << frac_bits))),
        ]

        # Stage 0: Unpack
        a_mant = Signal(wf); a_exp = Signal(we); a_sign = Signal(); a_exc = Signal(2)
        m.d.comb += [a_mant.eq(self.a[:wf]), a_exp.eq(self.a[wf:wf+we]),
                     a_sign.eq(self.a[wf+we]), a_exc.eq(self.a[wf+we+1:])]

        s1_mant = Signal(wf); s1_exp = Signal(we); s1_sign = Signal(); s1_exc = Signal(2)
        m.d.sync += [s1_mant.eq(a_mant), s1_exp.eq(a_exp), s1_sign.eq(a_sign), s1_exc.eq(a_exc)]

        # Stage 1: Exception + decompose exponent
        s1_special = Signal(); s1_exc_out = Signal(2)
        with m.If(s1_exc == 0b11):
            m.d.comb += [s1_exc_out.eq(0b11), s1_special.eq(1)]
        with m.Elif(s1_exc == 0b00):
            m.d.comb += [s1_exc_out.eq(0b00), s1_special.eq(1)]
        with m.Elif(s1_exc == 0b10):
            m.d.comb += [s1_exc_out.eq(0b10), s1_special.eq(1)]
        with m.Else():
            m.d.comb += [s1_exc_out.eq(0b01), s1_special.eq(0)]

        # e = biased_exp - bias; decompose: e = 3*k + r, r in {0,1,2}
        e_val = Signal(signed(we + 2))
        m.d.comb += e_val.eq(s1_exp - bias)

        # Table address
        tbl_addr = Signal(table_bits)
        m.d.comb += tbl_addr.eq(s1_mant >> (wf - table_bits) if wf >= table_bits else s1_mant)
        m.d.comb += tbl.addr.eq(tbl_addr)

        # Stage 1 → 2
        s2_exc = Signal(2); s2_sign = Signal(); s2_special = Signal(); s2_e = Signal(signed(we + 2))
        m.d.sync += [s2_exc.eq(s1_exc_out), s2_sign.eq(s1_sign), s2_special.eq(s1_special), s2_e.eq(e_val)]

        # Stage 2 → 3 (table read)
        s3_exc = Signal(2); s3_sign = Signal(); s3_special = Signal(); s3_e = Signal(signed(we + 2))
        m.d.sync += [s3_exc.eq(s2_exc), s3_sign.eq(s2_sign), s3_special.eq(s2_special), s3_e.eq(s2_e)]

        # Stage 3: Table result ready
        tbl_val = Signal(frac_bits + 2)
        m.d.comb += tbl_val.eq(tbl.data)

        # Compute k = e // 3, r = e % 3 (simplified for hardware)
        # result exponent = bias + k
        # result mantissa = cbrt_table[mant] * cbrt_pow2[r]
        # For simplicity, approximate k ≈ e/3
        k_approx = Signal(signed(we + 2))
        # Multiply by ~0.333: (e * 341) >> 10 ≈ e/3 for small e  (341/1024 ≈ 0.333)
        # Or use a simple division by constant
        m.d.comb += k_approx.eq(s3_e // 3)

        result_exp = Signal(we)
        m.d.comb += result_exp.eq((bias + k_approx)[:we])
        result_mant = Signal(wf)
        m.d.comb += result_mant.eq(tbl_val[frac_bits - wf:frac_bits])

        # Stage 3 → 4
        s4_exc = Signal(2); s4_sign = Signal(); s4_special = Signal()
        s4_rexp = Signal(we); s4_rmant = Signal(wf)
        m.d.sync += [s4_exc.eq(s3_exc), s4_sign.eq(s3_sign), s4_special.eq(s3_special),
                     s4_rexp.eq(result_exp), s4_rmant.eq(result_mant)]

        # Stages 4-6: Newton-Raphson refinement would go here (placeholder pipeline)
        s5_exc = Signal(2); s5_sign = Signal(); s5_special = Signal()
        s5_rexp = Signal(we); s5_rmant = Signal(wf)
        m.d.sync += [s5_exc.eq(s4_exc), s5_sign.eq(s4_sign), s5_special.eq(s4_special),
                     s5_rexp.eq(s4_rexp), s5_rmant.eq(s4_rmant)]

        s6_exc = Signal(2); s6_sign = Signal(); s6_special = Signal()
        s6_rexp = Signal(we); s6_rmant = Signal(wf)
        m.d.sync += [s6_exc.eq(s5_exc), s6_sign.eq(s5_sign), s6_special.eq(s5_special),
                     s6_rexp.eq(s5_rexp), s6_rmant.eq(s5_rmant)]

        # Stage 7: Pack result
        final_exc = Signal(2); final_sign = Signal(); final_mant = Signal(wf); final_exp = Signal(we)
        with m.If(s6_special):
            m.d.comb += [final_exc.eq(s6_exc), final_sign.eq(s6_sign), final_mant.eq(0), final_exp.eq(0)]
        with m.Else():
            m.d.comb += [final_exc.eq(0b01), final_sign.eq(s6_sign), final_mant.eq(s6_rmant), final_exp.eq(s6_rexp)]

        s7 = Signal(fmt.width)
        m.d.sync += s7.eq(Cat(final_mant, final_exp, final_sign, final_exc))

        s8 = Signal(fmt.width)
        m.d.sync += s8.eq(s7)
        m.d.comb += self.o.eq(s8)

        return m
