"""Tests for pipelined FP operators and conversions."""

import pytest
from amaranth.sim import Simulator

from amaranth_fp.format import FPFormat
from amaranth_fp.operators import FPSquare, FPAbs, FPNeg, FPMin, FPMax
from amaranth_fp.conversions import Fix2FP, FPResize
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
INF_POS = fp_inf(FMT, 0)
NAN_VAL = fp_nan(FMT)

NINE = fp_normal(FMT, 0, 18, 0b0010000000)
SEVEN = fp_normal(FMT, 0, 17, 0b1100000000)


def _run_clocked(dut, testbench):
    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_new_op.vcd"):
        sim.run()


# ---------------------------------------------------------------------------
# FPSquare tests (pipelined, latency=5)
# ---------------------------------------------------------------------------

class TestFPSquare:
    def test_three_squared(self):
        dut = FPSquare(FMT)

        async def bench(ctx):
            ctx.set(dut.a, THREE)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b01
            assert decode_sign(FMT, result) == 0
            assert decode_exp(FMT, result) == 18
            assert decode_mant(FMT, result) == 0b0010000000

        _run_clocked(dut, bench)


# ---------------------------------------------------------------------------
# FPAbs tests (pipelined, latency=1)
# ---------------------------------------------------------------------------

class TestFPAbs:
    def test_abs_negative(self):
        dut = FPAbs(FMT)

        async def bench(ctx):
            ctx.set(dut.a, NEG_TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b01
            assert decode_sign(FMT, result) == 0
            assert decode_exp(FMT, result) == 16
            assert decode_mant(FMT, result) == 0

        _run_clocked(dut, bench)

    def test_abs_positive(self):
        dut = FPAbs(FMT)

        async def bench(ctx):
            ctx.set(dut.a, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert result == TWO

        _run_clocked(dut, bench)


# ---------------------------------------------------------------------------
# FPNeg tests (pipelined, latency=1)
# ---------------------------------------------------------------------------

class TestFPNeg:
    def test_neg_positive(self):
        dut = FPNeg(FMT)

        async def bench(ctx):
            ctx.set(dut.a, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b01
            assert decode_sign(FMT, result) == 1
            assert decode_exp(FMT, result) == 16

        _run_clocked(dut, bench)

    def test_neg_negative(self):
        dut = FPNeg(FMT)

        async def bench(ctx):
            ctx.set(dut.a, NEG_TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b01
            assert decode_sign(FMT, result) == 0
            assert decode_exp(FMT, result) == 16

        _run_clocked(dut, bench)


# ---------------------------------------------------------------------------
# FPMin / FPMax tests (pipelined, latency=3)
# ---------------------------------------------------------------------------

class TestFPMinMax:
    def test_min_one_two(self):
        dut = FPMin(FMT)

        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert result == ONE_POS

        _run_clocked(dut, bench)

    def test_max_one_two(self):
        dut = FPMax(FMT)

        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert result == TWO

        _run_clocked(dut, bench)

    def test_min_nan(self):
        dut = FPMin(FMT)

        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b11

        _run_clocked(dut, bench)

    def test_max_nan(self):
        dut = FPMax(FMT)

        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, NAN_VAL)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b11

        _run_clocked(dut, bench)


# ---------------------------------------------------------------------------
# Fix2FP tests (pipelined, latency=3)
# ---------------------------------------------------------------------------

class TestFix2FP:
    def test_integer_three(self):
        """Convert integer 3 (unsigned, 8-bit int, 0-bit frac) to FP 3.0"""
        fmt = FPFormat.half()
        dut = Fix2FP(int_width=8, frac_width=0, signed=False, fmt=fmt)

        async def bench(ctx):
            ctx.set(dut.fix_in, 3)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.fp_out)
            assert decode_exc(fmt, result) == 0b01
            assert decode_sign(fmt, result) == 0
            assert decode_exp(fmt, result) == 16, f"exp={decode_exp(fmt, result)}"
            assert decode_mant(fmt, result) == 0b1000000000, f"mant={decode_mant(fmt, result):#012b}"

        _run_clocked(dut, bench)

    def test_zero(self):
        fmt = FPFormat.half()
        dut = Fix2FP(int_width=8, frac_width=0, signed=False, fmt=fmt)

        async def bench(ctx):
            ctx.set(dut.fix_in, 0)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.fp_out)
            assert decode_exc(fmt, result) == 0b00

        _run_clocked(dut, bench)


# ---------------------------------------------------------------------------
# FPResize tests (pipelined, latency=2)
# ---------------------------------------------------------------------------

class TestFPResize:
    def test_half_to_single(self):
        """Convert half-precision 2.0 to single-precision."""
        fmt_half = FPFormat.half()
        fmt_single = FPFormat.single()
        dut = FPResize(fmt_half, fmt_single)

        two_half = fp_normal(fmt_half, 0, 16, 0)

        async def bench(ctx):
            ctx.set(dut.fp_in, two_half)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.fp_out)
            assert decode_exc(fmt_single, result) == 0b01
            assert decode_sign(fmt_single, result) == 0
            assert decode_exp(fmt_single, result) == 128, f"exp={decode_exp(fmt_single, result)}"
            assert decode_mant(fmt_single, result) == 0

        _run_clocked(dut, bench)

    def test_nan_preserved(self):
        fmt_half = FPFormat.half()
        fmt_single = FPFormat.single()
        dut = FPResize(fmt_half, fmt_single)

        async def bench(ctx):
            ctx.set(dut.fp_in, fp_nan(fmt_half))
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.fp_out)
            assert decode_exc(fmt_single, result) == 0b11

        _run_clocked(dut, bench)
