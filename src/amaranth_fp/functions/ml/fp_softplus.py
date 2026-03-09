"""Floating-point softplus(x) = log(1 + exp(x)) (pipelined, ~8-cycle latency).

Large x → x, small x → table lookup on [0, 8). BranchMux.
"""
from __future__ import annotations

import math

from amaranth import *

from ...format import FPFormat, float_to_flopoco
from ...pipelined import PipelinedComponent
from ...building_blocks.branch_mux import BranchMux
from ..table import Table

__all__ = ["FPSoftplus"]


class FPSoftplus(PipelinedComponent):
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
        # Table covers [-4, 4) mapped as offset: index = (x + 4) / 8 * table_size
        # But since we store all pre-encoded, the domain range for unsigned access is [0, 8)
        # with offset: we'll address by the unsigned value |x| and handle sign separately
        # Actually, softplus is asymmetric: softplus(x) → x for large x, → 0 for large -x
        # For the small branch, use a table covering [0, 8) storing softplus(x - 4)
        # i.e., table[i] = softplus(i/table_size * 8 - 4)
        softplus_table = []
        for i in range(table_size):
            x = i / table_size * 8.0 - 4.0
            val = math.log1p(math.exp(x))
            softplus_table.append(float_to_flopoco(val, we, wf, bias) & ((1 << fmt.width) - 1))

        tbl = Table(table_bits, fmt.width, softplus_table)
        m.submodules.sp_tbl = tbl

        ln2_enc = float_to_flopoco(math.log(2), we, wf, bias)

        a_mant = Signal(wf); a_exp = Signal(we); a_sign = Signal(); a_exc = Signal(2)
        m.d.comb += [a_mant.eq(self.a[:wf]), a_exp.eq(self.a[wf:wf+we]),
                     a_sign.eq(self.a[wf+we]), a_exc.eq(self.a[wf+we+1:])]

        s1_mant = Signal(wf); s1_exp = Signal(we); s1_sign = Signal(); s1_exc = Signal(2)
        m.d.sync += [s1_mant.eq(a_mant), s1_exp.eq(a_exp), s1_sign.eq(a_sign), s1_exc.eq(a_exc)]

        s1_special = Signal(); s1_exc_out = Signal(2)
        with m.If(s1_exc == 0b11):
            m.d.comb += [s1_exc_out.eq(0b11), s1_special.eq(1)]
        with m.Elif(s1_exc == 0b10):
            with m.If(s1_sign):  # softplus(-inf) = 0
                m.d.comb += [s1_exc_out.eq(0b00), s1_special.eq(1)]
            with m.Else():  # softplus(+inf) = +inf
                m.d.comb += [s1_exc_out.eq(0b10), s1_special.eq(1)]
        with m.Elif(s1_exc == 0b00):  # softplus(0) = ln(2)
            m.d.comb += [s1_exc_out.eq(0b01), s1_special.eq(1)]
        with m.Else():
            m.d.comb += [s1_exc_out.eq(0b01), s1_special.eq(0)]

        # Large positive: x > 4 → softplus ≈ x
        branch_large = Signal()
        m.d.comb += branch_large.eq((s1_exp > bias + 1) & (~s1_sign))

        # Table address: the table covers [-4, 4) with offset
        # index = (x + 4) / 8 * table_size
        # For x in FP, we need to compute this. Since the range is [-4, 4),
        # we convert x to fixed-point, add 4, scale by table_size/8
        # x + 4 has range [0, 8) for our domain
        # index = (x + 4) / 8 * 256 = (x + 4) * 32
        # For x = 0: index = 4 * 32 = 128 → softplus(0) = ln(2) ✓
        # For x = 1: index = 5 * 32 = 160 → softplus(1) ≈ 1.313
        # For x = -4: index = 0 → softplus(-4) ≈ 0.018

        # Convert x to fixed-point (signed, with 3 integer bits for range [-4, 4))
        # significand with sign
        significand = Signal(wf + 1)
        m.d.comb += significand.eq(Cat(s1_mant, Const(1, 1)))
        e_val = Signal(signed(we + 1))
        m.d.comb += e_val.eq(s1_exp - bias)

        # x in fixed point with frac_bits = wf fractional bits
        # x_fixed = significand << e (or >> -e), then negate if sign
        x_fixed = Signal(signed(wf + we + 2))
        with m.If(e_val >= 0):
            m.d.comb += x_fixed.eq(significand << e_val[:we])
        with m.Else():
            neg_e = Signal(we + 1)
            m.d.comb += neg_e.eq(-e_val)
            m.d.comb += x_fixed.eq(significand >> neg_e)

        x_signed = Signal(signed(wf + we + 2))
        with m.If(s1_sign):
            m.d.comb += x_signed.eq(-x_fixed)
        with m.Else():
            m.d.comb += x_signed.eq(x_fixed)

        # Add offset: (x + 4) in fixed point. 4 in fixed point = 4 << wf
        offset = 4 << wf
        x_offset = Signal(wf + we + 3)
        m.d.comb += x_offset.eq(x_signed + offset)

        # Scale: index = x_offset / 8 * table_size = x_offset * table_size / (8 << wf)
        # = x_offset >> (3 + wf - table_bits)
        shift_amount = 3 + wf - table_bits
        raw_addr = Signal(wf + we + 3)
        m.d.comb += raw_addr.eq(x_offset >> shift_amount)

        tbl_addr = Signal(table_bits)
        with m.If(raw_addr >= table_size):
            m.d.comb += tbl_addr.eq(table_size - 1)
        with m.Elif(raw_addr < 0):
            m.d.comb += tbl_addr.eq(0)
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

        # Softplus output is always positive
        small_result = Signal(fmt.width)
        m.d.comb += small_result.eq(tbl_val)

        # Large: passthrough x (delayed)
        x_d1 = Signal(fmt.width); m.d.sync += x_d1.eq(self.a)
        x_d2 = Signal(fmt.width); m.d.sync += x_d2.eq(x_d1)
        x_d3 = Signal(fmt.width); m.d.sync += x_d3.eq(x_d2)
        large_result = Signal(fmt.width)
        m.d.comb += large_result.eq(x_d3)

        m.d.comb += [
            branch_mux.cond.eq(s3_large),
            branch_mux.branch_a.eq(small_result),
            branch_mux.branch_b.eq(large_result),
        ]

        sp_chain = [Signal(fmt.width, name=f"sp_{i}") for i in range(branch_mux.latency)]
        sp_sel = [Signal(name=f"sp_sel_{i}") for i in range(branch_mux.latency)]
        # softplus(0) = ln(2)
        special_enc = Signal(fmt.width)
        with m.If(s3_exc == 0b01):
            m.d.comb += special_enc.eq(ln2_enc)
        with m.Else():
            m.d.comb += special_enc.eq(Cat(Const(0, wf), Const(0, we), Const(0, 1), s3_exc))
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
