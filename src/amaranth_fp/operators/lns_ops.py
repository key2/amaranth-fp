"""Logarithmic Number System operations (pipelined).

LNS format: sign(1) | log_value(width-1) in fixed-point.
- LNSMul: add log values (latency 1)
- LNSAdd: table lookup for log(1 + 2^d) (latency 3)
"""
from __future__ import annotations

import math

from amaranth import *

from ..pipelined import PipelinedComponent
from ..functions.table import Table

__all__ = ["LNSMul", "LNSAdd"]


class LNSMul(PipelinedComponent):
    """LNS multiplication: add log values.

    Parameters
    ----------
    width : int
        Total LNS word width (sign + log_value).
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.a = Signal(width, name="a")
        self.b = Signal(width, name="b")
        self.o = Signal(width, name="o")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        lw = w - 1  # log value width

        a_sign = Signal(name="a_sign")
        b_sign = Signal(name="b_sign")
        a_log = Signal(lw, name="a_log")
        b_log = Signal(lw, name="b_log")

        m.d.comb += [
            a_sign.eq(self.a[w - 1]),
            b_sign.eq(self.b[w - 1]),
            a_log.eq(self.a[:lw]),
            b_log.eq(self.b[:lw]),
        ]

        result_sign = Signal(name="result_sign")
        result_log = Signal(lw, name="result_log")
        m.d.comb += [
            result_sign.eq(a_sign ^ b_sign),
            result_log.eq(a_log + b_log),
        ]

        o_r = Signal(w, name="o_r")
        m.d.sync += o_r.eq(Cat(result_log, result_sign))
        m.d.comb += self.o.eq(o_r)

        return m


class LNSAdd(PipelinedComponent):
    """LNS addition using table lookup for log(1 + 2^d).

    Parameters
    ----------
    width : int
        Total LNS word width (sign + log_value).
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.a = Signal(width, name="a")
        self.b = Signal(width, name="b")
        self.o = Signal(width, name="o")
        self.latency = 3

        lw = width - 1
        # Table for f(d) = log2(1 + 2^d) for d in [-2^(lw-1), 2^(lw-1))
        # We index with lw bits (the difference)
        table_bits = min(lw, 8)
        table_size = 1 << table_bits
        frac_bits = lw
        tbl_vals = []
        for i in range(table_size):
            # Map i to d: d = i - table_size/2 (signed)
            d = i - table_size // 2
            # f(d) = log2(1 + 2^d) in fixed-point
            try:
                val = math.log2(1.0 + 2.0 ** d) * (1 << frac_bits)
                tbl_vals.append(int(round(val)) & ((1 << (frac_bits + 2)) - 1))
            except (OverflowError, ValueError):
                tbl_vals.append(0)

        self._table_bits = table_bits
        self._tbl_vals = tbl_vals
        self._frac_bits = frac_bits

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        lw = w - 1
        tb = self._table_bits
        fb = self._frac_bits

        a_sign = Signal(name="as")
        b_sign = Signal(name="bs")
        a_log = Signal(lw, name="al")
        b_log = Signal(lw, name="bl")
        m.d.comb += [
            a_sign.eq(self.a[w - 1]), b_sign.eq(self.b[w - 1]),
            a_log.eq(self.a[:lw]), b_log.eq(self.b[:lw]),
        ]

        # Stage 0: Compute difference d = a_log - b_log, determine larger
        larger_log = Signal(lw, name="larger_log")
        diff = Signal(lw, name="diff")
        result_sign = Signal(name="rs")

        with m.If(a_log >= b_log):
            m.d.comb += [larger_log.eq(a_log), diff.eq(a_log - b_log), result_sign.eq(a_sign)]
        with m.Else():
            m.d.comb += [larger_log.eq(b_log), diff.eq(b_log - a_log), result_sign.eq(b_sign)]

        larger_r1 = Signal(lw, name="larger_r1")
        diff_r1 = Signal(lw, name="diff_r1")
        rs_r1 = Signal(name="rs_r1")
        m.d.sync += [larger_r1.eq(larger_log), diff_r1.eq(diff), rs_r1.eq(result_sign)]

        # Stage 1: Table lookup
        tbl = Table(tb, fb + 2, self._tbl_vals)
        m.submodules.lns_tbl = tbl

        # Use top bits of diff as table address (offset by table_size/2 for signed)
        tbl_idx = Signal(tb, name="tbl_idx")
        if lw >= tb:
            m.d.comb += tbl_idx.eq(diff_r1[lw - tb:lw])
        else:
            m.d.comb += tbl_idx.eq(diff_r1[:tb])
        m.d.comb += tbl.addr.eq(tbl_idx)

        larger_r2 = Signal(lw, name="larger_r2")
        rs_r2 = Signal(name="rs_r2")
        m.d.sync += [larger_r2.eq(larger_r1), rs_r2.eq(rs_r1)]

        # Stage 2: Add table result to larger log
        f_d = Signal(fb + 2, name="f_d")
        m.d.comb += f_d.eq(tbl.data)

        result_log = Signal(lw, name="result_log")
        m.d.comb += result_log.eq(larger_r2 + f_d[:lw])

        o_r = Signal(w, name="lns_add_o")
        m.d.sync += o_r.eq(Cat(result_log, rs_r2))
        m.d.comb += self.o.eq(o_r)

        return m
