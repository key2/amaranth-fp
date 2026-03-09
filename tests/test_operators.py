"""Tests for pipelined FP operators using Amaranth simulator."""

import pytest
from amaranth.sim import Simulator

from amaranth_fp.format import FPFormat
from amaranth_fp.operators import FPMul, FPDiv, FPSqrt, FPComparator
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
FOUR = fp_normal(FMT, 0, 17, 0)
SIX = fp_normal(FMT, 0, 17, 0b1000000000)
ZERO_POS = fp_zero(FMT, 0)
ZERO_NEG = fp_zero(FMT, 1)
INF_POS = fp_inf(FMT, 0)
INF_NEG = fp_inf(FMT, 1)
NAN_VAL = fp_nan(FMT)


def _run_clocked(dut, testbench):
    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_op.vcd"):
        sim.run()


# ---------------------------------------------------------------------------
# FPMul tests (pipelined, latency=5)
# ---------------------------------------------------------------------------

class TestFPMul:
    def test_two_times_three(self):
        dut = FPMul(FMT)

        async def bench(ctx):
            ctx.set(dut.a, TWO)
            ctx.set(dut.b, THREE)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b01
            assert decode_sign(FMT, result) == 0
            assert decode_exp(FMT, result) == 17
            assert decode_mant(FMT, result) == 0b1000000000

        _run_clocked(dut, bench)

    def test_one_times_zero(self):
        dut = FPMul(FMT)

        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b00

        _run_clocked(dut, bench)

    def test_inf_times_zero(self):
        dut = FPMul(FMT)

        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b11

        _run_clocked(dut, bench)

    def test_nan_times_x(self):
        dut = FPMul(FMT)

        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            ctx.set(dut.b, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b11

        _run_clocked(dut, bench)


# ---------------------------------------------------------------------------
# FPDiv tests (pipelined, latency=6)
# ---------------------------------------------------------------------------

class TestFPDiv:
    def test_six_div_two(self):
        dut = FPDiv(FMT)

        async def bench(ctx):
            ctx.set(dut.a, SIX)
            ctx.set(dut.b, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b01
            assert decode_sign(FMT, result) == 0
            assert decode_exp(FMT, result) == 16
            assert decode_mant(FMT, result) == 0b1000000000

        _run_clocked(dut, bench)

    def test_one_div_zero(self):
        dut = FPDiv(FMT)

        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b10

        _run_clocked(dut, bench)

    def test_zero_div_zero(self):
        dut = FPDiv(FMT)

        async def bench(ctx):
            ctx.set(dut.a, ZERO_POS)
            ctx.set(dut.b, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b11

        _run_clocked(dut, bench)


# ---------------------------------------------------------------------------
# FPSqrt tests (pipelined, latency=5)
# ---------------------------------------------------------------------------

class TestFPSqrt:
    def test_sqrt_four(self):
        dut = FPSqrt(FMT)

        async def bench(ctx):
            ctx.set(dut.a, FOUR)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b01
            assert decode_sign(FMT, result) == 0
            assert decode_exp(FMT, result) == 16
            assert decode_mant(FMT, result) == 0

        _run_clocked(dut, bench)

    def test_sqrt_zero(self):
        dut = FPSqrt(FMT)

        async def bench(ctx):
            ctx.set(dut.a, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b00

        _run_clocked(dut, bench)

    def test_sqrt_negative(self):
        dut = FPSqrt(FMT)

        async def bench(ctx):
            ctx.set(dut.a, ONE_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b11

        _run_clocked(dut, bench)

    def test_sqrt_inf(self):
        dut = FPSqrt(FMT)

        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.o)
            assert decode_exc(FMT, result) == 0b10

        _run_clocked(dut, bench)


# ---------------------------------------------------------------------------
# FPComparator tests (pipelined, latency=2)
# ---------------------------------------------------------------------------

class TestFPComparator:
    def test_one_lt_two(self):
        dut = FPComparator(FMT)

        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.lt) == 1
            assert ctx.get(dut.eq) == 0
            assert ctx.get(dut.gt) == 0
            assert ctx.get(dut.unordered) == 0

        _run_clocked(dut, bench)

    def test_one_eq_one(self):
        dut = FPComparator(FMT)

        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.lt) == 0
            assert ctx.get(dut.eq) == 1
            assert ctx.get(dut.gt) == 0

        _run_clocked(dut, bench)

    def test_nan_unordered(self):
        dut = FPComparator(FMT)

        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.unordered) == 1
            assert ctx.get(dut.lt) == 0
            assert ctx.get(dut.eq) == 0
            assert ctx.get(dut.gt) == 0

        _run_clocked(dut, bench)

    def test_two_gt_one(self):
        dut = FPComparator(FMT)

        async def bench(ctx):
            ctx.set(dut.a, TWO)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.gt) == 1
            assert ctx.get(dut.lt) == 0

        _run_clocked(dut, bench)

    def test_zero_eq_zero(self):
        dut = FPComparator(FMT)

        async def bench(ctx):
            ctx.set(dut.a, ZERO_POS)
            ctx.set(dut.b, ZERO_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.eq) == 1

        _run_clocked(dut, bench)
