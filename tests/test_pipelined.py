"""Tests for pipelined FP operators (FPAdd, FPSub, FPFMA, FP2Fix).

These components use m.d.sync and require clocked simulation.
"""

import pytest
from amaranth.sim import Simulator

from amaranth_fp.format import FPFormat
from amaranth_fp.operators import FPAdd, FPSub, FPFMA
from amaranth_fp.conversions import FP2Fix
from conftest import (
    fp_zero, fp_inf, fp_nan, fp_one, fp_normal,
    decode_exc, decode_sign, decode_exp, decode_mant,
)

FMT = FPFormat.half()
BIAS = FMT.bias  # 15

ONE_POS = fp_one(FMT, sign=0)
ONE_NEG = fp_one(FMT, sign=1)
TWO = fp_normal(FMT, 0, 16, 0)
THREE = fp_normal(FMT, 0, 16, 0b1000000000)
NEG_TWO = fp_normal(FMT, 1, 16, 0)
ZERO_POS = fp_zero(FMT, 0)
ZERO_NEG = fp_zero(FMT, 1)
INF_POS = fp_inf(FMT, 0)
INF_NEG = fp_inf(FMT, 1)
NAN_VAL = fp_nan(FMT)
SEVEN = fp_normal(FMT, 0, 17, 0b1100000000)


def _run_clocked(dut, testbench):
    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_pipelined.vcd"):
        sim.run()


# ---------------------------------------------------------------------------
# FPAdd (pipelined, latency=7)
# ---------------------------------------------------------------------------

class TestPipelinedFPAdd:
    def test_one_plus_one(self):
        dut = FPAdd(FMT)

        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b01, f"exc={decode_exc(FMT, result):#04b}"
            assert decode_sign(FMT, result) == 0
            assert decode_exp(FMT, result) == 16  # 2.0
            assert decode_mant(FMT, result) == 0

        _run_clocked(dut, bench)

    def test_one_plus_neg_one(self):
        dut = FPAdd(FMT)

        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, ONE_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b00  # zero

        _run_clocked(dut, bench)

    def test_inf_plus_inf(self):
        dut = FPAdd(FMT)

        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, INF_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b10  # inf

        _run_clocked(dut, bench)

    def test_inf_plus_neg_inf(self):
        dut = FPAdd(FMT)

        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, INF_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b11  # NaN

        _run_clocked(dut, bench)

    def test_nan_plus_x(self):
        dut = FPAdd(FMT)

        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b11

        _run_clocked(dut, bench)


# ---------------------------------------------------------------------------
# FPSub (pipelined via FPAdd, latency=7)
# ---------------------------------------------------------------------------

class TestPipelinedFPSub:
    def test_three_minus_one(self):
        dut = FPSub(FMT)
        # latency is set during elaborate, need to elaborate first
        # We know it's 7 from FPAdd

        async def bench(ctx):
            ctx.set(dut.a, THREE)
            ctx.set(dut.b, ONE_POS)
            for _ in range(7):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b01
            assert decode_sign(FMT, result) == 0
            assert decode_exp(FMT, result) == 16  # 2.0
            assert decode_mant(FMT, result) == 0

        _run_clocked(dut, bench)

    def test_one_minus_one(self):
        dut = FPSub(FMT)

        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, ONE_POS)
            for _ in range(7):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b00  # zero

        _run_clocked(dut, bench)

    def test_nan_sub(self):
        dut = FPSub(FMT)

        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            ctx.set(dut.b, ONE_POS)
            for _ in range(7):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b11

        _run_clocked(dut, bench)


# ---------------------------------------------------------------------------
# FPFMA (pipelined, latency=9)
# ---------------------------------------------------------------------------

class TestPipelinedFPFMA:
    def test_two_times_three_plus_one(self):
        """2.0 * 3.0 + 1.0 = 7.0"""
        dut = FPFMA(FMT)

        async def bench(ctx):
            ctx.set(dut.a, TWO)
            ctx.set(dut.b, THREE)
            ctx.set(dut.c, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b01, f"exc={decode_exc(FMT, result):#04b}"
            assert decode_sign(FMT, result) == 0
            assert decode_exp(FMT, result) == 17, f"exp={decode_exp(FMT, result)}"
            assert decode_mant(FMT, result) == 0b1100000000, f"mant={decode_mant(FMT, result):#012b}"

        _run_clocked(dut, bench)

    def test_nan_fma(self):
        dut = FPFMA(FMT)

        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            ctx.set(dut.b, TWO)
            ctx.set(dut.c, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b11

        _run_clocked(dut, bench)


# ---------------------------------------------------------------------------
# FP2Fix (pipelined, latency=3)
# ---------------------------------------------------------------------------

class TestPipelinedFP2Fix:
    def test_two_point_five(self):
        """Convert 2.5 to fixed-point with 8 int bits and 4 frac bits."""
        fmt = FPFormat.half()
        dut = FP2Fix(fmt=fmt, int_width=8, frac_width=4, signed=True)

        val_2_5 = fp_normal(fmt, 0, 16, 0b0100000000)

        async def bench(ctx):
            ctx.set(dut.fp_in, val_2_5)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.fix_out)
            assert result == 40, f"fix_out={result}"
            assert ctx.get(dut.overflow) == 0

        _run_clocked(dut, bench)

    def test_zero(self):
        fmt = FPFormat.half()
        dut = FP2Fix(fmt=fmt, int_width=8, frac_width=4, signed=True)

        async def bench(ctx):
            ctx.set(dut.fp_in, fp_zero(fmt))
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.fix_out)
            assert result == 0

        _run_clocked(dut, bench)
