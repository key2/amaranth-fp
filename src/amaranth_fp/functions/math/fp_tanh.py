"""Floating-point tanh(x) (pipelined, ~8-cycle latency).

Small |x|: table lookup on [0, 4). Large |x| > 4: saturate to ±1.
BranchMux for piecewise evaluation.
"""
from __future__ import annotations

import math

from amaranth import *

from ...format import FPFormat, float_to_flopoco
from ...pipelined import PipelinedComponent
from ...building_blocks.branch_mux import BranchMux
from ..table import Table

__all__ = ["FPTanh"]


class FPTanh(PipelinedComponent):
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

        table_bits = min(wf, 8)
        table_size = 1 << table_bits
        RANGE = 4.0
        RANGE_EXP = 2

        tanh_table = []
        for i in range(table_size):
            x = i / table_size * RANGE
            val = math.tanh(x)
            tanh_table.append(float_to_flopoco(val, we, wf, bias) & ((1 << fmt.width) - 1))

        tbl = Table(table_bits, fmt.width, tanh_table)
        m.submodules.tanh_tbl = tbl

        one_enc = float_to_flopoco(1.0, we, wf, bias)

        a_mant = Signal(wf); a_exp = Signal(we); a_sign = Signal(); a_exc = Signal(2)
        m.d.comb += [a_mant.eq(self.a[:wf]), a_exp.eq(self.a[wf:wf+we]),
                     a_sign.eq(self.a[wf+we]), a_exc.eq(self.a[wf+we+1:])]

        s1_mant = Signal(wf); s1_exp = Signal(we); s1_sign = Signal(); s1_exc = Signal(2)
        m.d.sync += [s1_mant.eq(a_mant), s1_exp.eq(a_exp), s1_sign.eq(a_sign), s1_exc.eq(a_exc)]

        s1_special = Signal(); s1_exc_out = Signal(2)
        with m.If(s1_exc == 0b11):
            m.d.comb += [s1_exc_out.eq(0b11), s1_special.eq(1)]
        with m.Elif(s1_exc == 0b00):
            m.d.comb += [s1_exc_out.eq(0b00), s1_special.eq(1)]
        with m.Elif(s1_exc == 0b10):  # tanh(±inf) = ±1
            m.d.comb += [s1_exc_out.eq(0b01), s1_special.eq(1)]
        with m.Else():
            m.d.comb += [s1_exc_out.eq(0b01), s1_special.eq(0)]

        branch_large = Signal()
        m.d.comb += branch_large.eq(s1_exp >= bias + RANGE_EXP)

        # Table address computation
        significand = Signal(wf + 1)
        m.d.comb += significand.eq(Cat(s1_mant, Const(1, 1)))
        e_val = Signal(signed(we + 1))
        m.d.comb += e_val.eq(s1_exp - bias)
        BASE_SHIFT = table_bits - wf - RANGE_EXP
        total_shift = Signal(signed(we + 2))
        m.d.comb += total_shift.eq(BASE_SHIFT + e_val)
        raw_addr = Signal(wf + table_bits + 2)
        with m.If(total_shift >= 0):
            m.d.comb += raw_addr.eq(significand << total_shift[:5])
        with m.Else():
            neg_shift = Signal(we + 1)
            m.d.comb += neg_shift.eq(-total_shift)
            m.d.comb += raw_addr.eq(significand >> neg_shift)
        tbl_addr = Signal(table_bits)
        with m.If(raw_addr >= table_size):
            m.d.comb += tbl_addr.eq(table_size - 1)
        with m.Else():
            m.d.comb += tbl_addr.eq(raw_addr[:table_bits])
        m.d.comb += tbl.addr.eq(tbl_addr)

        branch_mux = BranchMux(width=fmt.width, latency_a=3, latency_b=3)
        m.submodules.branch_mux = branch_mux

        s2_exc = Signal(2); s2_sign = Signal(); s2_special = Signal(); s2_large = Signal()
        m.d.sync += [s2_exc.eq(s1_exc_out), s2_sign.eq(s1_sign), s2_special.eq(s1_special), s2_large.eq(branch_large)]

        s3_exc = Signal(2); s3_sign = Signal(); s3_special = Signal(); s3_large = Signal()
        m.d.sync += [s3_exc.eq(s2_exc), s3_sign.eq(s2_sign), s3_special.eq(s2_special), s3_large.eq(s2_large)]

        tbl_val = Signal(fmt.width)
        m.d.comb += tbl_val.eq(tbl.data)

        sign_pos = we + wf
        small_result = Signal(fmt.width)
        m.d.comb += small_result.eq((tbl_val & ~(1 << sign_pos)) | (s3_sign << sign_pos))

        large_result = Signal(fmt.width)
        m.d.comb += large_result.eq((one_enc & ~(1 << sign_pos)) | (s3_sign << sign_pos))

        m.d.comb += [
            branch_mux.cond.eq(s3_large),
            branch_mux.branch_a.eq(small_result),
            branch_mux.branch_b.eq(large_result),
        ]

        sp_chain = [Signal(fmt.width, name=f"sp_{i}") for i in range(branch_mux.latency)]
        sp_sel = [Signal(name=f"sp_sel_{i}") for i in range(branch_mux.latency)]
        special_enc = Signal(fmt.width)
        with m.If(s3_exc == 0b01):
            m.d.comb += special_enc.eq((one_enc & ~(1 << sign_pos)) | (s3_sign << sign_pos))
        with m.Else():
            m.d.comb += special_enc.eq(Cat(Const(0, wf), Const(0, we), s3_sign, s3_exc))

        if branch_mux.latency > 0:
            m.d.sync += [sp_chain[0].eq(special_enc), sp_sel[0].eq(s3_special)]
            for i in range(1, branch_mux.latency):
                m.d.sync += [sp_chain[i].eq(sp_chain[i-1]), sp_sel[i].eq(sp_sel[i-1])]
            s_out = Signal(fmt.width)
            m.d.comb += s_out.eq(Mux(sp_sel[-1], sp_chain[-1], branch_mux.o))
        else:
            s_out = Signal(fmt.width)
            m.d.comb += s_out.eq(branch_mux.o)

        m.d.sync += self.o.eq(s_out)
        return m
