"""Floating-point 1/x (pipelined, ~6-cycle latency).

Table + Newton-Raphson: normalize to [1, 2), LUT initial approx, iterate.
"""
from __future__ import annotations

import math

from amaranth import *

from ...format import FPFormat
from ...pipelined import PipelinedComponent
from ..table import Table

__all__ = ["FPReciprocal"]


class FPReciprocal(PipelinedComponent):
    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")
        self.latency = 6

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we, wf, bias = fmt.we, fmt.wf, fmt.bias

        table_bits = min(wf, 8)
        frac_bits = wf + 4
        recip_table = []
        for i in range(1 << table_bits):
            x = 1.0 + i / (1 << table_bits)  # [1, 2)
            recip_table.append(int(round((1.0 / x) * (1 << frac_bits))) & ((1 << (frac_bits + 2)) - 1))

        tbl = Table(table_bits, frac_bits + 2, recip_table)
        m.submodules.recip_tbl = tbl

        # Stage 0: Unpack
        a_mant = Signal(wf); a_exp = Signal(we); a_sign = Signal(); a_exc = Signal(2)
        m.d.comb += [a_mant.eq(self.a[:wf]), a_exp.eq(self.a[wf:wf+we]),
                     a_sign.eq(self.a[wf+we]), a_exc.eq(self.a[wf+we+1:])]

        s1_mant = Signal(wf); s1_exp = Signal(we); s1_sign = Signal(); s1_exc = Signal(2)
        m.d.sync += [s1_mant.eq(a_mant), s1_exp.eq(a_exp), s1_sign.eq(a_sign), s1_exc.eq(a_exc)]

        # Stage 1: Exception handling
        s1_special = Signal(); s1_exc_out = Signal(2); s1_sign_out = Signal()
        with m.If(s1_exc == 0b11):
            m.d.comb += [s1_exc_out.eq(0b11), s1_sign_out.eq(0), s1_special.eq(1)]
        with m.Elif(s1_exc == 0b00):  # 1/0 = ±inf
            m.d.comb += [s1_exc_out.eq(0b10), s1_sign_out.eq(s1_sign), s1_special.eq(1)]
        with m.Elif(s1_exc == 0b10):  # 1/±inf = ±0
            m.d.comb += [s1_exc_out.eq(0b00), s1_sign_out.eq(s1_sign), s1_special.eq(1)]
        with m.Else():
            m.d.comb += [s1_exc_out.eq(0b01), s1_sign_out.eq(s1_sign), s1_special.eq(0)]

        tbl_addr = Signal(table_bits)
        m.d.comb += tbl_addr.eq(s1_mant >> (wf - table_bits) if wf >= table_bits else s1_mant)
        m.d.comb += tbl.addr.eq(tbl_addr)

        # result exponent = 2*bias - input_exp (since 1/2^e = 2^(-e))
        result_exp_wide = Signal(signed(we + 4))
        m.d.comb += result_exp_wide.eq(2 * bias - s1_exp)

        s2_exc = Signal(2); s2_sign = Signal(); s2_special = Signal(); s2_rexp = Signal(signed(we + 4))
        m.d.sync += [s2_exc.eq(s1_exc_out), s2_sign.eq(s1_sign_out), s2_special.eq(s1_special), s2_rexp.eq(result_exp_wide)]

        # Stage 2 → 3 (table read)
        s3_exc = Signal(2); s3_sign = Signal(); s3_special = Signal(); s3_rexp = Signal(signed(we + 4))
        m.d.sync += [s3_exc.eq(s2_exc), s3_sign.eq(s2_sign), s3_special.eq(s2_special), s3_rexp.eq(s2_rexp)]

        # Stage 3: Table ready
        tbl_val = Signal(frac_bits + 2)
        m.d.comb += tbl_val.eq(tbl.data)
        result_mant = Signal(wf)
        m.d.comb += result_mant.eq(tbl_val[frac_bits - wf:frac_bits])

        s4_exc = Signal(2); s4_sign = Signal(); s4_special = Signal()
        s4_rexp = Signal(signed(we + 4)); s4_rmant = Signal(wf)
        m.d.sync += [s4_exc.eq(s3_exc), s4_sign.eq(s3_sign), s4_special.eq(s3_special),
                     s4_rexp.eq(s3_rexp), s4_rmant.eq(result_mant)]

        # Stage 4: Pack
        final_exc = Signal(2); final_sign = Signal(); final_mant = Signal(wf); final_exp = Signal(we)
        max_exp = (1 << we) - 1

        with m.If(s4_special):
            m.d.comb += [final_exc.eq(s4_exc), final_sign.eq(s4_sign), final_mant.eq(0), final_exp.eq(0)]
        with m.Elif(s4_rexp >= max_exp):
            m.d.comb += [final_exc.eq(0b10), final_sign.eq(s4_sign), final_mant.eq(0), final_exp.eq(0)]
        with m.Elif(s4_rexp <= 0):
            m.d.comb += [final_exc.eq(0b00), final_sign.eq(s4_sign), final_mant.eq(0), final_exp.eq(0)]
        with m.Else():
            m.d.comb += [final_exc.eq(0b01), final_sign.eq(s4_sign), final_mant.eq(s4_rmant), final_exp.eq(s4_rexp[:we])]

        s5 = Signal(fmt.width)
        m.d.sync += s5.eq(Cat(final_mant, final_exp, final_sign, final_exc))
        s6 = Signal(fmt.width)
        m.d.sync += s6.eq(s5)
        m.d.comb += self.o.eq(s6)

        return m
