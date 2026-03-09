"""Comprehensive edge-case tests for core FP operators.

Covers: FPAdd, FPSub, FPMul, FPDiv, FPSqrt, FPFMA,
        FPComparator, FPAbs, FPNeg, FPMin, FPMax, FPSquare.

Uses FPFormat.half() (we=5, wf=10) for fast simulation.
"""

import pytest
from amaranth.sim import Simulator

from amaranth_fp.format import FPFormat
from amaranth_fp.operators import (
    FPAdd, FPSub, FPMul, FPDiv, FPSqrt, FPFMA,
    FPComparator, FPAbs, FPNeg, FPMin, FPMax, FPSquare,
)
from conftest import (
    fp_zero, fp_inf, fp_nan, fp_one, fp_normal,
    decode_exc, decode_sign, decode_exp, decode_mant,
)

# ---------------------------------------------------------------------------
# Format & constants
# ---------------------------------------------------------------------------

FMT = FPFormat.half()       # we=5, wf=10
BIAS = FMT.bias             # 15
MAX_EXP = (1 << FMT.we) - 2  # 30 (largest biased exponent for normals)
MAX_MANT = (1 << FMT.wf) - 1  # 0x3FF = 1023

# Commonly used encoded values
ZERO_POS = fp_zero(FMT, 0)
ZERO_NEG = fp_zero(FMT, 1)
INF_POS  = fp_inf(FMT, 0)
INF_NEG  = fp_inf(FMT, 1)
NAN_VAL  = fp_nan(FMT)
ONE_POS  = fp_one(FMT, 0)   # 1.0  (exp=15, mant=0)
ONE_NEG  = fp_one(FMT, 1)   # -1.0
TWO      = fp_normal(FMT, 0, 16, 0)                # 2.0
NEG_TWO  = fp_normal(FMT, 1, 16, 0)                # -2.0
THREE    = fp_normal(FMT, 0, 16, 0b1000000000)      # 3.0
FOUR     = fp_normal(FMT, 0, 17, 0)                # 4.0
SIX      = fp_normal(FMT, 0, 17, 0b1000000000)      # 6.0
SEVEN    = fp_normal(FMT, 0, 17, 0b1100000000)      # 7.0
NINE     = fp_normal(FMT, 0, 18, 0b0010000000)      # 9.0
HALF     = fp_normal(FMT, 0, 14, 0)                # 0.5

# Near-max normal: exponent=30, mantissa=all-ones  ≈ 65504
NEAR_MAX = fp_normal(FMT, 0, MAX_EXP, MAX_MANT)
NEG_NEAR_MAX = fp_normal(FMT, 1, MAX_EXP, MAX_MANT)

# Small normal: exponent=1, mantissa=0  ≈ 2^(-14)
SMALL    = fp_normal(FMT, 0, 1, 0)
NEG_SMALL = fp_normal(FMT, 1, 1, 0)

# Very small normal: exponent=1, mantissa=1
TINY     = fp_normal(FMT, 0, 1, 1)

# Exception codes
EXC_ZERO = 0b00
EXC_NORM = 0b01
EXC_INF  = 0b10
EXC_NAN  = 0b11


def _run(dut, testbench):
    """Run a clocked simulation writing VCD to test_edge.vcd."""
    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_edge.vcd"):
        sim.run()


# ===================================================================
# FPAdd edge cases
# ===================================================================

class TestFPAddEdge:
    """Edge-case tests for FPAdd (latency=7)."""

    # --- normal arithmetic ---

    def test_normal_plus_normal_same_exp(self):
        """1.0 + 1.0 = 2.0 (same exponent)."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 16   # 2.0
            assert decode_mant(FMT, r) == 0
        _run(dut, bench)

    def test_normal_plus_normal_exp_diff_1(self):
        """1.0 + 2.0 = 3.0 (exponent difference = 1)."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 16
            assert decode_mant(FMT, r) == 0b1000000000  # 3.0
        _run(dut, bench)

    def test_normal_plus_normal_exp_diff_2(self):
        """1.0 + 4.0 = 5.0 (exponent difference = 2)."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, FOUR)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 17
            assert decode_mant(FMT, r) == 0b0100000000  # 5.0 = 1.01 * 2^2
        _run(dut, bench)

    # --- zero operands ---

    def test_normal_plus_zero(self):
        """2.0 + 0 = 2.0."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, TWO)
            ctx.set(dut.b, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 16
            assert decode_mant(FMT, r) == 0
        _run(dut, bench)

    def test_pos_zero_plus_pos_zero(self):
        """+0 + +0 = +0."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ZERO_POS)
            ctx.set(dut.b, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
        _run(dut, bench)

    def test_neg_zero_plus_neg_zero(self):
        """-0 + -0 = -0."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ZERO_NEG)
            ctx.set(dut.b, ZERO_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
            assert decode_sign(FMT, r) == 1
        _run(dut, bench)

    def test_pos_zero_plus_neg_zero(self):
        """+0 + -0 = +0."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ZERO_POS)
            ctx.set(dut.b, ZERO_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
            assert decode_sign(FMT, r) == 0
        _run(dut, bench)

    # --- exact cancellation ---

    def test_exact_cancellation(self):
        """1.0 + (-1.0) = 0."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, ONE_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
        _run(dut, bench)

    def test_exact_cancellation_large(self):
        """near_max + (-near_max) = 0."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NEAR_MAX)
            ctx.set(dut.b, NEG_NEAR_MAX)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
        _run(dut, bench)

    # --- infinity operands ---

    def test_inf_plus_normal(self):
        """+Inf + 1.0 = +Inf."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
            assert decode_sign(FMT, r) == 0
        _run(dut, bench)

    def test_neg_inf_plus_normal(self):
        """-Inf + 1.0 = -Inf."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_NEG)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
            assert decode_sign(FMT, r) == 1
        _run(dut, bench)

    def test_inf_plus_inf(self):
        """+Inf + +Inf = +Inf."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, INF_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
            assert decode_sign(FMT, r) == 0
        _run(dut, bench)

    def test_inf_plus_neg_inf(self):
        """+Inf + -Inf = NaN."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, INF_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    # --- NaN propagation ---

    def test_nan_plus_normal(self):
        """NaN + 1.0 = NaN."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_normal_plus_nan(self):
        """1.0 + NaN = NaN."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, NAN_VAL)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_nan_plus_nan(self):
        """NaN + NaN = NaN."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            ctx.set(dut.b, NAN_VAL)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    # --- overflow ---

    def test_overflow_large_plus_large(self):
        """near_max + near_max → Inf."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NEAR_MAX)
            ctx.set(dut.b, NEAR_MAX)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
        _run(dut, bench)

    def test_near_max_plus_small_no_overflow(self):
        """near_max + small should stay normal (no overflow)."""
        dut = FPAdd(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NEAR_MAX)
            ctx.set(dut.b, SMALL)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
        _run(dut, bench)


# ===================================================================
# FPSub edge cases
# ===================================================================

class TestFPSubEdge:
    """Edge-case tests for FPSub (latency=7)."""

    def test_three_minus_one(self):
        """3.0 - 1.0 = 2.0."""
        dut = FPSub(FMT)
        async def bench(ctx):
            ctx.set(dut.a, THREE)
            ctx.set(dut.b, ONE_POS)
            for _ in range(7):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 16
            assert decode_mant(FMT, r) == 0
        _run(dut, bench)

    def test_one_minus_one(self):
        """1.0 - 1.0 = 0."""
        dut = FPSub(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, ONE_POS)
            for _ in range(7):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
        _run(dut, bench)

    def test_one_minus_two(self):
        """1.0 - 2.0 = -1.0."""
        dut = FPSub(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, TWO)
            for _ in range(7):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 1
            assert decode_exp(FMT, r) == BIAS  # 1.0
            assert decode_mant(FMT, r) == 0
        _run(dut, bench)

    def test_sub_nan(self):
        """NaN - 1.0 = NaN."""
        dut = FPSub(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            ctx.set(dut.b, ONE_POS)
            for _ in range(7):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_inf_minus_inf(self):
        """+Inf - +Inf = NaN."""
        dut = FPSub(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, INF_POS)
            for _ in range(7):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_zero_minus_zero(self):
        """+0 - +0 = +0."""
        dut = FPSub(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ZERO_POS)
            ctx.set(dut.b, ZERO_POS)
            for _ in range(7):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
        _run(dut, bench)


# ===================================================================
# FPMul edge cases
# ===================================================================

class TestFPMulEdge:
    """Edge-case tests for FPMul (latency=5)."""

    # --- sign rules ---

    def test_pos_times_pos(self):
        """2.0 × 3.0 = 6.0."""
        dut = FPMul(FMT)
        async def bench(ctx):
            ctx.set(dut.a, TWO)
            ctx.set(dut.b, THREE)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 17
            assert decode_mant(FMT, r) == 0b1000000000
        _run(dut, bench)

    def test_neg_times_neg(self):
        """(-2.0) × (-1.0) = 2.0."""
        dut = FPMul(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NEG_TWO)
            ctx.set(dut.b, ONE_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 16
            assert decode_mant(FMT, r) == 0
        _run(dut, bench)

    def test_pos_times_neg(self):
        """2.0 × (-1.0) = -2.0."""
        dut = FPMul(FMT)
        async def bench(ctx):
            ctx.set(dut.a, TWO)
            ctx.set(dut.b, ONE_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 1
            assert decode_exp(FMT, r) == 16
            assert decode_mant(FMT, r) == 0
        _run(dut, bench)

    # --- zero operands ---

    def test_normal_times_zero(self):
        """2.0 × 0 = 0."""
        dut = FPMul(FMT)
        async def bench(ctx):
            ctx.set(dut.a, TWO)
            ctx.set(dut.b, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
        _run(dut, bench)

    def test_neg_times_zero_sign(self):
        """(-2.0) × (+0) = -0 (sign = 1)."""
        dut = FPMul(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NEG_TWO)
            ctx.set(dut.b, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
            assert decode_sign(FMT, r) == 1
        _run(dut, bench)

    # --- identity ---

    def test_normal_times_one(self):
        """3.0 × 1.0 = 3.0."""
        dut = FPMul(FMT)
        async def bench(ctx):
            ctx.set(dut.a, THREE)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 16
            assert decode_mant(FMT, r) == 0b1000000000
        _run(dut, bench)

    # --- infinity ---

    def test_inf_times_normal(self):
        """+Inf × 2.0 = +Inf."""
        dut = FPMul(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
            assert decode_sign(FMT, r) == 0
        _run(dut, bench)

    def test_inf_times_neg_normal(self):
        """+Inf × (-2.0) = -Inf."""
        dut = FPMul(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, NEG_TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
            assert decode_sign(FMT, r) == 1
        _run(dut, bench)

    def test_inf_times_zero(self):
        """+Inf × 0 = NaN."""
        dut = FPMul(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_inf_times_inf(self):
        """+Inf × +Inf = +Inf."""
        dut = FPMul(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, INF_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
        _run(dut, bench)

    def test_neg_inf_times_neg_inf(self):
        """-Inf × -Inf = +Inf."""
        dut = FPMul(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_NEG)
            ctx.set(dut.b, INF_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
            assert decode_sign(FMT, r) == 0
        _run(dut, bench)

    # --- NaN ---

    def test_nan_times_normal(self):
        """NaN × 2.0 = NaN."""
        dut = FPMul(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            ctx.set(dut.b, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_normal_times_nan(self):
        """2.0 × NaN = NaN."""
        dut = FPMul(FMT)
        async def bench(ctx):
            ctx.set(dut.a, TWO)
            ctx.set(dut.b, NAN_VAL)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    # --- overflow / underflow ---

    def test_overflow_large_times_large(self):
        """near_max × near_max → Inf (product overflows)."""
        dut = FPMul(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NEAR_MAX)
            ctx.set(dut.b, NEAR_MAX)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
        _run(dut, bench)

    def test_underflow_small_times_small(self):
        """small × small → 0 (underflow)."""
        dut = FPMul(FMT)
        async def bench(ctx):
            ctx.set(dut.a, SMALL)
            ctx.set(dut.b, SMALL)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
        _run(dut, bench)


# ===================================================================
# FPDiv edge cases
# ===================================================================

class TestFPDivEdge:
    """Edge-case tests for FPDiv (latency=6)."""

    # --- normal division ---

    def test_exact_division(self):
        """6.0 / 2.0 = 3.0."""
        dut = FPDiv(FMT)
        async def bench(ctx):
            ctx.set(dut.a, SIX)
            ctx.set(dut.b, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 16
            assert decode_mant(FMT, r) == 0b1000000000
        _run(dut, bench)

    def test_div_by_one(self):
        """3.0 / 1.0 = 3.0."""
        dut = FPDiv(FMT)
        async def bench(ctx):
            ctx.set(dut.a, THREE)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 16
            assert decode_mant(FMT, r) == 0b1000000000
        _run(dut, bench)

    # --- division by zero ---

    def test_normal_div_zero(self):
        """1.0 / 0 = +Inf."""
        dut = FPDiv(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
        _run(dut, bench)

    def test_neg_div_zero(self):
        """(-1.0) / (+0) = -Inf."""
        dut = FPDiv(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_NEG)
            ctx.set(dut.b, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
            assert decode_sign(FMT, r) == 1
        _run(dut, bench)

    def test_zero_div_zero(self):
        """0 / 0 = NaN."""
        dut = FPDiv(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ZERO_POS)
            ctx.set(dut.b, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    # --- zero numerator ---

    def test_zero_div_normal(self):
        """0 / 2.0 = 0."""
        dut = FPDiv(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ZERO_POS)
            ctx.set(dut.b, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
        _run(dut, bench)

    # --- infinity ---

    def test_inf_div_normal(self):
        """+Inf / 2.0 = +Inf."""
        dut = FPDiv(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
            assert decode_sign(FMT, r) == 0
        _run(dut, bench)

    def test_inf_div_inf(self):
        """+Inf / +Inf = NaN."""
        dut = FPDiv(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, INF_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_normal_div_inf(self):
        """1.0 / +Inf = 0."""
        dut = FPDiv(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, INF_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
        _run(dut, bench)

    # --- NaN ---

    def test_nan_div_normal(self):
        """NaN / 2.0 = NaN."""
        dut = FPDiv(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            ctx.set(dut.b, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_normal_div_nan(self):
        """2.0 / NaN = NaN."""
        dut = FPDiv(FMT)
        async def bench(ctx):
            ctx.set(dut.a, TWO)
            ctx.set(dut.b, NAN_VAL)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    # --- overflow / underflow ---

    def test_overflow_large_div_small(self):
        """near_max / small → Inf."""
        dut = FPDiv(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NEAR_MAX)
            ctx.set(dut.b, SMALL)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
        _run(dut, bench)

    def test_underflow_small_div_large(self):
        """small / near_max → 0."""
        dut = FPDiv(FMT)
        async def bench(ctx):
            ctx.set(dut.a, SMALL)
            ctx.set(dut.b, NEAR_MAX)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
        _run(dut, bench)


# ===================================================================
# FPSqrt edge cases
# ===================================================================

class TestFPSqrtEdge:
    """Edge-case tests for FPSqrt (latency=5)."""

    def test_sqrt_zero(self):
        """sqrt(0) = 0."""
        dut = FPSqrt(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
        _run(dut, bench)

    def test_sqrt_one(self):
        """sqrt(1.0) = 1.0."""
        dut = FPSqrt(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == BIAS
            assert decode_mant(FMT, r) == 0
        _run(dut, bench)

    def test_sqrt_four(self):
        """sqrt(4.0) = 2.0."""
        dut = FPSqrt(FMT)
        async def bench(ctx):
            ctx.set(dut.a, FOUR)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 16
            assert decode_mant(FMT, r) == 0
        _run(dut, bench)

    def test_sqrt_nine(self):
        """sqrt(9.0) = 3.0."""
        dut = FPSqrt(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NINE)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 16
            assert decode_mant(FMT, r) == 0b1000000000
        _run(dut, bench)

    def test_sqrt_negative(self):
        """sqrt(-1.0) = NaN."""
        dut = FPSqrt(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_sqrt_neg_two(self):
        """sqrt(-2.0) = NaN."""
        dut = FPSqrt(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NEG_TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_sqrt_pos_inf(self):
        """sqrt(+Inf) = +Inf."""
        dut = FPSqrt(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
        _run(dut, bench)

    def test_sqrt_neg_inf(self):
        """sqrt(-Inf) = NaN."""
        dut = FPSqrt(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_sqrt_nan(self):
        """sqrt(NaN) = NaN."""
        dut = FPSqrt(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_sqrt_very_small(self):
        """sqrt(very small) → small normal or zero."""
        dut = FPSqrt(FMT)
        async def bench(ctx):
            ctx.set(dut.a, SMALL)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            exc = decode_exc(FMT, r)
            assert exc in (EXC_NORM, EXC_ZERO)
            assert decode_sign(FMT, r) == 0
        _run(dut, bench)


# ===================================================================
# FPFMA edge cases
# ===================================================================

class TestFPFMAEdge:
    """Edge-case tests for FPFMA (a*b+c, latency=9)."""

    def test_normal_fma(self):
        """2.0 × 3.0 + 1.0 = 7.0."""
        dut = FPFMA(FMT)
        async def bench(ctx):
            ctx.set(dut.a, TWO)
            ctx.set(dut.b, THREE)
            ctx.set(dut.c, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 17
            assert decode_mant(FMT, r) == 0b1100000000  # 7.0
        _run(dut, bench)

    def test_zero_times_zero_plus_zero(self):
        """0 × 0 + 0 = 0."""
        dut = FPFMA(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ZERO_POS)
            ctx.set(dut.b, ZERO_POS)
            ctx.set(dut.c, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
        _run(dut, bench)

    def test_inf_times_zero_plus_c(self):
        """Inf × 0 + 1.0 = NaN."""
        dut = FPFMA(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, ZERO_POS)
            ctx.set(dut.c, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_nan_in_a(self):
        """NaN × 2.0 + 1.0 = NaN."""
        dut = FPFMA(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            ctx.set(dut.b, TWO)
            ctx.set(dut.c, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_nan_in_b(self):
        """2.0 × NaN + 1.0 = NaN."""
        dut = FPFMA(FMT)
        async def bench(ctx):
            ctx.set(dut.a, TWO)
            ctx.set(dut.b, NAN_VAL)
            ctx.set(dut.c, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_nan_in_c(self):
        """2.0 × 3.0 + NaN = NaN."""
        dut = FPFMA(FMT)
        async def bench(ctx):
            ctx.set(dut.a, TWO)
            ctx.set(dut.b, THREE)
            ctx.set(dut.c, NAN_VAL)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_exact_cancellation(self):
        """1.0 × 1.0 + (-1.0) = 0."""
        dut = FPFMA(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, ONE_POS)
            ctx.set(dut.c, ONE_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
        _run(dut, bench)

    def test_product_overflow(self):
        """near_max × near_max + 0 → Inf (product overflows)."""
        dut = FPFMA(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NEAR_MAX)
            ctx.set(dut.b, NEAR_MAX)
            ctx.set(dut.c, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
        _run(dut, bench)

    def test_inf_addend(self):
        """1.0 × 1.0 + Inf = Inf."""
        dut = FPFMA(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, ONE_POS)
            ctx.set(dut.c, INF_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
        _run(dut, bench)

    def test_fma_identity(self):
        """1.0 × 3.0 + 0 = 3.0."""
        dut = FPFMA(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, THREE)
            ctx.set(dut.c, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 16
            assert decode_mant(FMT, r) == 0b1000000000
        _run(dut, bench)


# ===================================================================
# FPComparator edge cases
# ===================================================================

class TestFPComparatorEdge:
    """Edge-case tests for FPComparator (latency=2)."""

    def test_normal_lt_normal(self):
        """1.0 < 2.0."""
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
        _run(dut, bench)

    def test_normal_gt_normal(self):
        """3.0 > 2.0."""
        dut = FPComparator(FMT)
        async def bench(ctx):
            ctx.set(dut.a, THREE)
            ctx.set(dut.b, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.gt) == 1
            assert ctx.get(dut.lt) == 0
            assert ctx.get(dut.eq) == 0
            assert ctx.get(dut.unordered) == 0
        _run(dut, bench)

    def test_normal_eq_normal(self):
        """2.0 == 2.0."""
        dut = FPComparator(FMT)
        async def bench(ctx):
            ctx.set(dut.a, TWO)
            ctx.set(dut.b, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.eq) == 1
            assert ctx.get(dut.lt) == 0
            assert ctx.get(dut.gt) == 0
            assert ctx.get(dut.unordered) == 0
        _run(dut, bench)

    def test_pos_zero_eq_neg_zero(self):
        """+0 == -0."""
        dut = FPComparator(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ZERO_POS)
            ctx.set(dut.b, ZERO_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.eq) == 1
            assert ctx.get(dut.unordered) == 0
        _run(dut, bench)

    def test_nan_unordered_with_normal(self):
        """NaN vs 1.0 → unordered."""
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
        _run(dut, bench)

    def test_normal_unordered_with_nan(self):
        """1.0 vs NaN → unordered."""
        dut = FPComparator(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, NAN_VAL)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.unordered) == 1
        _run(dut, bench)

    def test_nan_unordered_with_nan(self):
        """NaN vs NaN → unordered."""
        dut = FPComparator(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            ctx.set(dut.b, NAN_VAL)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.unordered) == 1
            assert ctx.get(dut.eq) == 0
        _run(dut, bench)

    def test_inf_eq_inf(self):
        """+Inf == +Inf (both have exc=10, same sign, same exp/mant=0)."""
        dut = FPComparator(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, INF_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            # Both positive, same magnitude (exp=0, mant=0) → eq
            assert ctx.get(dut.eq) == 1
            assert ctx.get(dut.unordered) == 0
        _run(dut, bench)

    def test_neg_inf_eq_neg_inf(self):
        """-Inf == -Inf."""
        dut = FPComparator(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_NEG)
            ctx.set(dut.b, INF_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.eq) == 1
        _run(dut, bench)

    def test_inf_neq_neg_inf(self):
        """+Inf != -Inf: different signs → gt (positive > negative)."""
        dut = FPComparator(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, INF_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.gt) == 1
            assert ctx.get(dut.eq) == 0
        _run(dut, bench)

    def test_negative_lt_positive(self):
        """-2.0 < 1.0."""
        dut = FPComparator(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NEG_TWO)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.lt) == 1
        _run(dut, bench)


# ===================================================================
# FPAbs edge cases
# ===================================================================

class TestFPAbsEdge:
    """Edge-case tests for FPAbs (latency=1)."""

    def test_abs_positive(self):
        """abs(2.0) = 2.0."""
        dut = FPAbs(FMT)
        async def bench(ctx):
            ctx.set(dut.a, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 16
            assert decode_mant(FMT, r) == 0
        _run(dut, bench)

    def test_abs_negative(self):
        """abs(-2.0) = 2.0."""
        dut = FPAbs(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NEG_TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 16
            assert decode_mant(FMT, r) == 0
        _run(dut, bench)

    def test_abs_pos_zero(self):
        """abs(+0) = +0."""
        dut = FPAbs(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
            assert decode_sign(FMT, r) == 0
        _run(dut, bench)

    def test_abs_neg_zero(self):
        """abs(-0) = +0."""
        dut = FPAbs(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ZERO_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
            assert decode_sign(FMT, r) == 0
        _run(dut, bench)

    def test_abs_pos_inf(self):
        """abs(+Inf) = +Inf."""
        dut = FPAbs(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
            assert decode_sign(FMT, r) == 0
        _run(dut, bench)

    def test_abs_neg_inf(self):
        """abs(-Inf) = +Inf."""
        dut = FPAbs(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
            assert decode_sign(FMT, r) == 0
        _run(dut, bench)

    def test_abs_nan(self):
        """abs(NaN) = NaN."""
        dut = FPAbs(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)


# ===================================================================
# FPNeg edge cases
# ===================================================================

class TestFPNegEdge:
    """Edge-case tests for FPNeg (latency=1)."""

    def test_neg_positive(self):
        """neg(2.0) = -2.0."""
        dut = FPNeg(FMT)
        async def bench(ctx):
            ctx.set(dut.a, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 1
            assert decode_exp(FMT, r) == 16
            assert decode_mant(FMT, r) == 0
        _run(dut, bench)

    def test_neg_negative(self):
        """neg(-2.0) = 2.0."""
        dut = FPNeg(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NEG_TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 16
            assert decode_mant(FMT, r) == 0
        _run(dut, bench)

    def test_neg_pos_zero(self):
        """neg(+0) = -0."""
        dut = FPNeg(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
            assert decode_sign(FMT, r) == 1
        _run(dut, bench)

    def test_neg_neg_zero(self):
        """neg(-0) = +0."""
        dut = FPNeg(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ZERO_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
            assert decode_sign(FMT, r) == 0
        _run(dut, bench)

    def test_neg_pos_inf(self):
        """neg(+Inf) = -Inf."""
        dut = FPNeg(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
            assert decode_sign(FMT, r) == 1
        _run(dut, bench)

    def test_neg_neg_inf(self):
        """neg(-Inf) = +Inf."""
        dut = FPNeg(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
            assert decode_sign(FMT, r) == 0
        _run(dut, bench)

    def test_neg_nan(self):
        """neg(NaN) = NaN."""
        dut = FPNeg(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)


# ===================================================================
# FPMin edge cases
# ===================================================================

class TestFPMinEdge:
    """Edge-case tests for FPMin (latency=3)."""

    def test_min_two_normals(self):
        """min(3.0, 1.0) = 1.0."""
        dut = FPMin(FMT)
        async def bench(ctx):
            ctx.set(dut.a, THREE)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert r == ONE_POS
        _run(dut, bench)

    def test_min_with_zero(self):
        """min(1.0, +0) = +0."""
        dut = FPMin(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
        _run(dut, bench)

    def test_min_with_neg(self):
        """min(-2.0, 1.0) = -2.0."""
        dut = FPMin(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NEG_TWO)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert r == NEG_TWO
        _run(dut, bench)

    def test_min_inf_inf(self):
        """min(+Inf, +Inf) = +Inf (same value)."""
        dut = FPMin(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, INF_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
        _run(dut, bench)

    def test_min_neg_inf_neg_inf(self):
        """min(-Inf, -Inf) = -Inf."""
        dut = FPMin(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_NEG)
            ctx.set(dut.b, INF_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
            assert decode_sign(FMT, r) == 1
        _run(dut, bench)

    def test_min_nan_a(self):
        """min(NaN, 1.0) = NaN."""
        dut = FPMin(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_min_nan_b(self):
        """min(1.0, NaN) = NaN."""
        dut = FPMin(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, NAN_VAL)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_min_equal_values(self):
        """min(2.0, 2.0) = 2.0."""
        dut = FPMin(FMT)
        async def bench(ctx):
            ctx.set(dut.a, TWO)
            ctx.set(dut.b, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert r == TWO
        _run(dut, bench)


# ===================================================================
# FPMax edge cases
# ===================================================================

class TestFPMaxEdge:
    """Edge-case tests for FPMax (latency=3)."""

    def test_max_two_normals(self):
        """max(1.0, 3.0) = 3.0."""
        dut = FPMax(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, THREE)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert r == THREE
        _run(dut, bench)

    def test_max_with_zero(self):
        """max(+0, 1.0) = 1.0."""
        dut = FPMax(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ZERO_POS)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert r == ONE_POS
        _run(dut, bench)

    def test_max_with_neg(self):
        """max(-2.0, 1.0) = 1.0."""
        dut = FPMax(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NEG_TWO)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert r == ONE_POS
        _run(dut, bench)

    def test_max_inf_inf(self):
        """max(+Inf, +Inf) = +Inf."""
        dut = FPMax(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            ctx.set(dut.b, INF_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
            assert decode_sign(FMT, r) == 0
        _run(dut, bench)

    def test_max_neg_inf_neg_inf(self):
        """max(-Inf, -Inf) = -Inf."""
        dut = FPMax(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_NEG)
            ctx.set(dut.b, INF_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
            assert decode_sign(FMT, r) == 1
        _run(dut, bench)

    def test_max_nan_a(self):
        """max(NaN, 1.0) = NaN."""
        dut = FPMax(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            ctx.set(dut.b, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_max_nan_b(self):
        """max(1.0, NaN) = NaN."""
        dut = FPMax(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            ctx.set(dut.b, NAN_VAL)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_max_equal_values(self):
        """max(2.0, 2.0) = 2.0."""
        dut = FPMax(FMT)
        async def bench(ctx):
            ctx.set(dut.a, TWO)
            ctx.set(dut.b, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert r == TWO
        _run(dut, bench)


# ===================================================================
# FPSquare edge cases
# ===================================================================

class TestFPSquareEdge:
    """Edge-case tests for FPSquare (latency=5)."""

    def test_square_two(self):
        """2.0² = 4.0."""
        dut = FPSquare(FMT)
        async def bench(ctx):
            ctx.set(dut.a, TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 17  # 4.0
            assert decode_mant(FMT, r) == 0
        _run(dut, bench)

    def test_square_neg_two(self):
        """(-2.0)² = 4.0 (sign always positive)."""
        dut = FPSquare(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NEG_TWO)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 17
            assert decode_mant(FMT, r) == 0
        _run(dut, bench)

    def test_square_three(self):
        """3.0² = 9.0."""
        dut = FPSquare(FMT)
        async def bench(ctx):
            ctx.set(dut.a, THREE)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == 18
            assert decode_mant(FMT, r) == 0b0010000000  # 9.0
        _run(dut, bench)

    def test_square_one(self):
        """1.0² = 1.0."""
        dut = FPSquare(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ONE_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NORM
            assert decode_sign(FMT, r) == 0
            assert decode_exp(FMT, r) == BIAS
            assert decode_mant(FMT, r) == 0
        _run(dut, bench)

    def test_square_zero(self):
        """0² = 0."""
        dut = FPSquare(FMT)
        async def bench(ctx):
            ctx.set(dut.a, ZERO_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_ZERO
        _run(dut, bench)

    def test_square_inf(self):
        """(+Inf)² = +Inf."""
        dut = FPSquare(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_POS)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
        _run(dut, bench)

    def test_square_neg_inf(self):
        """(-Inf)² = +Inf."""
        dut = FPSquare(FMT)
        async def bench(ctx):
            ctx.set(dut.a, INF_NEG)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
            assert decode_sign(FMT, r) == 0
        _run(dut, bench)

    def test_square_nan(self):
        """NaN² = NaN."""
        dut = FPSquare(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NAN_VAL)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_NAN
        _run(dut, bench)

    def test_square_overflow(self):
        """near_max² → Inf."""
        dut = FPSquare(FMT)
        async def bench(ctx):
            ctx.set(dut.a, NEAR_MAX)
            for _ in range(dut.latency):
                await ctx.tick()
            r = ctx.get(dut.o)
            assert decode_exc(FMT, r) == EXC_INF
        _run(dut, bench)
