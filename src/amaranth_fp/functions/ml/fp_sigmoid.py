"""Floating-point sigmoid(x) = 1/(1+exp(-x)) (pipelined, ~8-cycle latency).

Piecewise: |x| > 8 → 0/1, else table lookup. BranchMux.
"""
from __future__ import annotations

import math

from amaranth import *

from ...format import FPFormat, float_to_flopoco
from ...pipelined import PipelinedComponent
from ...building_blocks.branch_mux import BranchMux
from ..table import Table

__all__ = ["FPSigmoid"]


class FPSigmoid(PipelinedComponent):
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
        # Table stores sigmoid(x) for x in [0, 8), pre-encoded as FloPoCo FP
        # For negative x: sigmoid(-x) = 1 - sigmoid(x)
        RANGE = 8.0
        RANGE_EXP = 3

        sigmoid_table = []
        for i in range(table_size):
            x = i / table_size * RANGE
            val = 1.0 / (1.0 + math.exp(-x))
            sigmoid_table.append(float_to_flopoco(val, we, wf, bias) & ((1 << fmt.width) - 1))

        tbl = Table(table_bits, fmt.width, sigmoid_table)
        m.submodules.sig_tbl = tbl

        one_enc = float_to_flopoco(1.0, we, wf, bias)
        half_enc = float_to_flopoco(0.5, we, wf, bias)

        a_mant = Signal(wf); a_exp = Signal(we); a_sign = Signal(); a_exc = Signal(2)
        m.d.comb += [a_mant.eq(self.a[:wf]), a_exp.eq(self.a[wf:wf+we]),
                     a_sign.eq(self.a[wf+we]), a_exc.eq(self.a[wf+we+1:])]

        s1_mant = Signal(wf); s1_exp = Signal(we); s1_sign = Signal(); s1_exc = Signal(2)
        m.d.sync += [s1_mant.eq(a_mant), s1_exp.eq(a_exp), s1_sign.eq(a_sign), s1_exc.eq(a_exc)]

        s1_special = Signal(); s1_exc_out = Signal(2)
        with m.If(s1_exc == 0b11):
            m.d.comb += [s1_exc_out.eq(0b11), s1_special.eq(1)]
        with m.Elif(s1_exc == 0b00):  # sigmoid(0) = 0.5
            m.d.comb += [s1_exc_out.eq(0b01), s1_special.eq(1)]
        with m.Elif(s1_exc == 0b10):  # sigmoid(±inf) = 1 or 0
            m.d.comb += [s1_exc_out.eq(Mux(s1_sign, 0b00, 0b01)), s1_special.eq(1)]
        with m.Else():
            m.d.comb += [s1_exc_out.eq(0b01), s1_special.eq(0)]

        # |x| >= 8 → saturate
        branch_large = Signal()
        m.d.comb += branch_large.eq(s1_exp >= bias + RANGE_EXP)

        # Table address computation (for |x|)
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

        # Small branch: table gives sigmoid(|x|).
        # For negative x: sigmoid(-x) = 1 - sigmoid(x)
        # We store sigmoid(|x|) which is in [0.5, 1) for x >= 0.
        # For x < 0, we need 1 - sigmoid(|x|).
        # Simplified: for positive x, use table directly. For negative x, use 1 - table.
        # But 1 - sigmoid(|x|) needs a subtraction. For simplicity in this table-based approach,
        # build a second table for 1 - sigmoid(x), or approximate.
        # Actually: sigmoid(-x) = 1 - sigmoid(x). So we can just store both:
        # For x >= 0: result = table[addr]
        # For x < 0: result = 1 - table[addr]
        # But we don't have a subtractor here. Let's build a neg_table.
        neg_sigmoid_table = []
        for i in range(table_size):
            x = i / table_size * RANGE
            val = 1.0 - 1.0 / (1.0 + math.exp(-x))  # = sigmoid(-x) for x >= 0
            neg_sigmoid_table.append(float_to_flopoco(val, we, wf, bias) & ((1 << fmt.width) - 1))

        neg_tbl = Table(table_bits, fmt.width, neg_sigmoid_table)
        m.submodules.neg_sig_tbl = neg_tbl
        m.d.comb += neg_tbl.addr.eq(tbl_addr)

        # Need to delay negative table read too
        neg_tbl_val = Signal(fmt.width)
        m.d.comb += neg_tbl_val.eq(neg_tbl.data)

        small_result = Signal(fmt.width)
        with m.If(s3_sign):
            m.d.comb += small_result.eq(neg_tbl_val)
        with m.Else():
            m.d.comb += small_result.eq(tbl_val)

        # Large: saturate to 0 (negative) or 1 (positive)
        large_result = Signal(fmt.width)
        with m.If(s3_sign):
            m.d.comb += large_result.eq(0)  # zero encoding
        with m.Else():
            m.d.comb += large_result.eq(one_enc)

        m.d.comb += [
            branch_mux.cond.eq(s3_large),
            branch_mux.branch_a.eq(small_result),
            branch_mux.branch_b.eq(large_result),
        ]

        # Special handling
        sp_chain = [Signal(fmt.width, name=f"sp_{i}") for i in range(branch_mux.latency)]
        sp_sel = [Signal(name=f"sp_sel_{i}") for i in range(branch_mux.latency)]
        # sigmoid(0) = 0.5, sigmoid(+inf) = 1, sigmoid(-inf) = 0
        special_enc = Signal(fmt.width)
        with m.If(s3_exc == 0b01):
            m.d.comb += special_enc.eq(half_enc)
        with m.Elif(s3_exc == 0b00):
            m.d.comb += special_enc.eq(0)  # zero
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
