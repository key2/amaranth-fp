"""Value-level tests for integer operators.

Tests: IntConstMult, IntConstDiv, IntAddSub, IntSquarer, IntConstantComparator,
       IntIntKCM, IntConstMultShiftAdd, BaseMultiplier, DSPBlock, FixMultAdd,
       IntMultiplierLUT, BaseSquarerLUT, IntMultiAdder, FixMultiAdder.
"""
import pytest
from amaranth.sim import Simulator

from amaranth_fp.operators import IntConstMult, IntConstDiv, IntIntKCM, IntConstMultShiftAdd
from amaranth_fp.integer import (
    IntAddSub,
    IntSquarer,
    IntConstantComparator,
    BaseMultiplier,
    DSPBlock,
    FixMultAdd,
    IntMultiplierLUT,
    BaseSquarerLUT,
    IntMultiAdder,
    FixMultiAdder,
)


def _run(dut, testbench, vcd_name="test_int_op.vcd"):
    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd(vcd_name):
        sim.run()


# ===================================================================
# 1. IntConstMult  (x → result = x * constant, latency=1)
# ===================================================================
class TestIntConstMult:
    """Integer constant multiplication via CSD shift-add."""

    def test_times_3(self):
        """5 * 3 = 15."""
        dut = IntConstMult(width=8, constant=3)

        async def bench(ctx):
            ctx.set(dut.x, 5)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            result = ctx.get(dut.result)
            assert result == 15, f"Expected 15, got {result}"

        _run(dut, bench, "test_int_const_mult_5x3.vcd")

    def test_times_7(self):
        """10 * 7 = 70."""
        dut = IntConstMult(width=8, constant=7)

        async def bench(ctx):
            ctx.set(dut.x, 10)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            result = ctx.get(dut.result)
            assert result == 70, f"Expected 70, got {result}"

        _run(dut, bench, "test_int_const_mult_10x7.vcd")

    def test_times_1(self):
        """42 * 1 = 42."""
        dut = IntConstMult(width=8, constant=1)

        async def bench(ctx):
            ctx.set(dut.x, 42)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            result = ctx.get(dut.result)
            assert result == 42, f"Expected 42, got {result}"

        _run(dut, bench, "test_int_const_mult_42x1.vcd")

    def test_zero_input(self):
        """0 * 5 = 0."""
        dut = IntConstMult(width=8, constant=5)

        async def bench(ctx):
            ctx.set(dut.x, 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            result = ctx.get(dut.result)
            assert result == 0, f"Expected 0, got {result}"

        _run(dut, bench, "test_int_const_mult_0x5.vcd")

    def test_power_of_two(self):
        """3 * 8 = 24."""
        dut = IntConstMult(width=8, constant=8)

        async def bench(ctx):
            ctx.set(dut.x, 3)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            result = ctx.get(dut.result)
            assert result == 24, f"Expected 24, got {result}"

        _run(dut, bench, "test_int_const_mult_3x8.vcd")


# ===================================================================
# 2. IntConstDiv  (a → q, r where a = q*divisor + r, latency=2)
# ===================================================================
class TestIntConstDiv:
    """Integer constant division."""

    def test_10_div_3(self):
        """10 / 3 = 3 remainder 1."""
        dut = IntConstDiv(width=8, divisor=3)

        async def bench(ctx):
            ctx.set(dut.a, 10)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            q = ctx.get(dut.q)
            r = ctx.get(dut.r)
            assert q == 3, f"Expected q=3, got {q}"
            assert r == 1, f"Expected r=1, got {r}"

        _run(dut, bench, "test_int_const_div_10d3.vcd")

    def test_12_div_4(self):
        """12 / 4 = 3 remainder 0."""
        dut = IntConstDiv(width=8, divisor=4)

        async def bench(ctx):
            ctx.set(dut.a, 12)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            q = ctx.get(dut.q)
            r = ctx.get(dut.r)
            assert q == 3, f"Expected q=3, got {q}"
            assert r == 0, f"Expected r=0, got {r}"

        _run(dut, bench, "test_int_const_div_12d4.vcd")

    def test_0_div_5(self):
        """0 / 5 = 0 remainder 0."""
        dut = IntConstDiv(width=8, divisor=5)

        async def bench(ctx):
            ctx.set(dut.a, 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            q = ctx.get(dut.q)
            r = ctx.get(dut.r)
            assert q == 0, f"Expected q=0, got {q}"
            assert r == 0, f"Expected r=0, got {r}"

        _run(dut, bench, "test_int_const_div_0d5.vcd")

    def test_255_div_7(self):
        """255 / 7 = 36 remainder 3."""
        dut = IntConstDiv(width=8, divisor=7)

        async def bench(ctx):
            ctx.set(dut.a, 255)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            q = ctx.get(dut.q)
            r = ctx.get(dut.r)
            assert q == 36, f"Expected q=36, got {q}"
            assert r == 3, f"Expected r=3, got {r}"

        _run(dut, bench, "test_int_const_div_255d7.vcd")

    def test_div_raises_on_zero(self):
        """Divisor=0 should raise ValueError."""
        with pytest.raises(ValueError):
            IntConstDiv(width=8, divisor=0)


# ===================================================================
# 3. IntAddSub  (a, b, op → s, cout, latency=1)
# ===================================================================
class TestIntAddSub:
    """Integer add/subtract."""

    def test_add_3_plus_5(self):
        """3 + 5 = 8."""
        dut = IntAddSub(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 3)
            ctx.set(dut.b, 5)
            ctx.set(dut.op, 0)  # add
            for _ in range(dut.latency + 1):
                await ctx.tick()
            s = ctx.get(dut.s)
            cout = ctx.get(dut.cout)
            assert s == 8, f"Expected 8, got {s}"
            assert cout == 0

        _run(dut, bench, "test_int_add_sub_3p5.vcd")

    def test_sub_10_minus_3(self):
        """10 - 3 = 7."""
        dut = IntAddSub(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 10)
            ctx.set(dut.b, 3)
            ctx.set(dut.op, 1)  # sub
            for _ in range(dut.latency + 1):
                await ctx.tick()
            s = ctx.get(dut.s)
            cout = ctx.get(dut.cout)
            assert s == 7, f"Expected 7, got {s}"
            assert cout == 1  # no borrow → cout=1

        _run(dut, bench, "test_int_add_sub_10m3.vcd")

    def test_add_overflow(self):
        """200 + 100 = 300 → 44 with carry."""
        dut = IntAddSub(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 200)
            ctx.set(dut.b, 100)
            ctx.set(dut.op, 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            s = ctx.get(dut.s)
            cout = ctx.get(dut.cout)
            assert s == (300 & 0xFF), f"Expected {300 & 0xFF}, got {s}"
            assert cout == 1

        _run(dut, bench, "test_int_add_sub_overflow.vcd")

    def test_add_zero(self):
        """0 + 0 = 0."""
        dut = IntAddSub(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 0)
            ctx.set(dut.b, 0)
            ctx.set(dut.op, 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            s = ctx.get(dut.s)
            assert s == 0

        _run(dut, bench, "test_int_add_sub_zero.vcd")

    def test_sub_equal(self):
        """5 - 5 = 0."""
        dut = IntAddSub(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 5)
            ctx.set(dut.b, 5)
            ctx.set(dut.op, 1)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            s = ctx.get(dut.s)
            cout = ctx.get(dut.cout)
            assert s == 0, f"Expected 0, got {s}"
            assert cout == 1  # no borrow

        _run(dut, bench, "test_int_add_sub_5m5.vcd")


# ===================================================================
# 4. IntSquarer  (a → p = a^2, latency=2)
# ===================================================================
class TestIntSquarer:
    """Integer squarer."""

    def test_square_5(self):
        """5^2 = 25."""
        dut = IntSquarer(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 5)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            p = ctx.get(dut.p)
            assert p == 25, f"Expected 25, got {p}"

        _run(dut, bench, "test_int_squarer_5.vcd")

    def test_square_0(self):
        """0^2 = 0."""
        dut = IntSquarer(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            p = ctx.get(dut.p)
            assert p == 0, f"Expected 0, got {p}"

        _run(dut, bench, "test_int_squarer_0.vcd")

    def test_square_1(self):
        """1^2 = 1."""
        dut = IntSquarer(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 1)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            p = ctx.get(dut.p)
            assert p == 1, f"Expected 1, got {p}"

        _run(dut, bench, "test_int_squarer_1.vcd")

    def test_square_15(self):
        """15^2 = 225."""
        dut = IntSquarer(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 15)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            p = ctx.get(dut.p)
            assert p == 225, f"Expected 225, got {p}"

        _run(dut, bench, "test_int_squarer_15.vcd")


# ===================================================================
# 5. IntConstantComparator  (a → lt, eq, gt vs constant, latency=1)
# ===================================================================
class TestIntConstantComparator:
    """Compare integer against compile-time constant."""

    def test_less_than(self):
        """3 < 10."""
        dut = IntConstantComparator(width=8, constant=10)

        async def bench(ctx):
            ctx.set(dut.a, 3)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            assert ctx.get(dut.lt) == 1
            assert ctx.get(dut.eq) == 0
            assert ctx.get(dut.gt) == 0

        _run(dut, bench, "test_int_const_cmp_lt.vcd")

    def test_equal(self):
        """10 == 10."""
        dut = IntConstantComparator(width=8, constant=10)

        async def bench(ctx):
            ctx.set(dut.a, 10)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            assert ctx.get(dut.lt) == 0
            assert ctx.get(dut.eq) == 1
            assert ctx.get(dut.gt) == 0

        _run(dut, bench, "test_int_const_cmp_eq.vcd")

    def test_greater_than(self):
        """20 > 10."""
        dut = IntConstantComparator(width=8, constant=10)

        async def bench(ctx):
            ctx.set(dut.a, 20)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            assert ctx.get(dut.lt) == 0
            assert ctx.get(dut.eq) == 0
            assert ctx.get(dut.gt) == 1

        _run(dut, bench, "test_int_const_cmp_gt.vcd")

    def test_zero_vs_zero(self):
        """0 == 0."""
        dut = IntConstantComparator(width=8, constant=0)

        async def bench(ctx):
            ctx.set(dut.a, 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            assert ctx.get(dut.eq) == 1

        _run(dut, bench, "test_int_const_cmp_0v0.vcd")

    def test_max_vs_const(self):
        """255 > 100."""
        dut = IntConstantComparator(width=8, constant=100)

        async def bench(ctx):
            ctx.set(dut.a, 255)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            assert ctx.get(dut.gt) == 1

        _run(dut, bench, "test_int_const_cmp_max.vcd")


# ===================================================================
# 6. IntIntKCM  (x → o = x * constant, latency=1)
# ===================================================================
class TestIntIntKCM:
    """Integer × integer constant multiplier using KCM."""

    def test_times_5(self):
        """7 * 5 = 35."""
        dut = IntIntKCM(width=8, constant=5)

        async def bench(ctx):
            ctx.set(dut.x, 7)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 35, f"Expected 35, got {o}"

        _run(dut, bench, "test_int_int_kcm_7x5.vcd")

    def test_times_1(self):
        """100 * 1 = 100."""
        dut = IntIntKCM(width=8, constant=1)

        async def bench(ctx):
            ctx.set(dut.x, 100)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 100, f"Expected 100, got {o}"

        _run(dut, bench, "test_int_int_kcm_100x1.vcd")

    def test_zero_input(self):
        """0 * 10 = 0."""
        dut = IntIntKCM(width=8, constant=10)

        async def bench(ctx):
            ctx.set(dut.x, 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0, f"Expected 0, got {o}"

        _run(dut, bench, "test_int_int_kcm_0x10.vcd")

    def test_times_255(self):
        """2 * 255 = 510."""
        dut = IntIntKCM(width=8, constant=255)

        async def bench(ctx):
            ctx.set(dut.x, 2)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 510, f"Expected 510, got {o}"

        _run(dut, bench, "test_int_int_kcm_2x255.vcd")


# ===================================================================
# 7. IntConstMultShiftAdd  (x → o = x * constant, latency=2)
# ===================================================================
class TestIntConstMultShiftAdd:
    """Integer constant multiplier using shift-and-add."""

    def test_times_3(self):
        """4 * 3 = 12."""
        dut = IntConstMultShiftAdd(width=8, constant=3)

        async def bench(ctx):
            ctx.set(dut.x, 4)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 12, f"Expected 12, got {o}"

        _run(dut, bench, "test_int_const_mult_sa_4x3.vcd")

    def test_times_10(self):
        """25 * 10 = 250."""
        dut = IntConstMultShiftAdd(width=8, constant=10)

        async def bench(ctx):
            ctx.set(dut.x, 25)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 250, f"Expected 250, got {o}"

        _run(dut, bench, "test_int_const_mult_sa_25x10.vcd")

    def test_zero_input(self):
        """0 * 5 = 0."""
        dut = IntConstMultShiftAdd(width=8, constant=5)

        async def bench(ctx):
            ctx.set(dut.x, 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0, f"Expected 0, got {o}"

        _run(dut, bench, "test_int_const_mult_sa_0x5.vcd")

    def test_times_1(self):
        """200 * 1 = 200."""
        dut = IntConstMultShiftAdd(width=8, constant=1)

        async def bench(ctx):
            ctx.set(dut.x, 200)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 200, f"Expected 200, got {o}"

        _run(dut, bench, "test_int_const_mult_sa_200x1.vcd")


# ===================================================================
# 8. BaseMultiplier  (x, y → o = x * y, latency=2)
# ===================================================================
class TestBaseMultiplier:
    """Base multiplier abstraction."""

    def test_3_times_4(self):
        """3 * 4 = 12."""
        dut = BaseMultiplier(x_width=8, y_width=8)

        async def bench(ctx):
            ctx.set(dut.x, 3)
            ctx.set(dut.y, 4)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 12, f"Expected 12, got {o}"

        _run(dut, bench, "test_base_mult_3x4.vcd")

    def test_0_times_100(self):
        """0 * 100 = 0."""
        dut = BaseMultiplier(x_width=8, y_width=8)

        async def bench(ctx):
            ctx.set(dut.x, 0)
            ctx.set(dut.y, 100)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0, f"Expected 0, got {o}"

        _run(dut, bench, "test_base_mult_0x100.vcd")

    def test_15_times_15(self):
        """15 * 15 = 225."""
        dut = BaseMultiplier(x_width=8, y_width=8)

        async def bench(ctx):
            ctx.set(dut.x, 15)
            ctx.set(dut.y, 15)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 225, f"Expected 225, got {o}"

        _run(dut, bench, "test_base_mult_15x15.vcd")

    def test_asymmetric_widths(self):
        """4-bit × 8-bit: 15 * 200 = 3000."""
        dut = BaseMultiplier(x_width=4, y_width=8)

        async def bench(ctx):
            ctx.set(dut.x, 15)
            ctx.set(dut.y, 200)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 3000, f"Expected 3000, got {o}"

        _run(dut, bench, "test_base_mult_asym.vcd")


# ===================================================================
# 9. DSPBlock  (x, y → o = x * y, latency=3)
# ===================================================================
class TestDSPBlock:
    """DSP block multiplier."""

    def test_3_times_4(self):
        """3 * 4 = 12."""
        dut = DSPBlock(x_width=8, y_width=8)

        async def bench(ctx):
            ctx.set(dut.x, 3)
            ctx.set(dut.y, 4)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 12, f"Expected 12, got {o}"

        _run(dut, bench, "test_dsp_block_3x4.vcd")

    def test_0_times_0(self):
        """0 * 0 = 0."""
        dut = DSPBlock(x_width=8, y_width=8)

        async def bench(ctx):
            ctx.set(dut.x, 0)
            ctx.set(dut.y, 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0, f"Expected 0, got {o}"

        _run(dut, bench, "test_dsp_block_0x0.vcd")

    def test_max_values(self):
        """255 * 255 = 65025."""
        dut = DSPBlock(x_width=8, y_width=8)

        async def bench(ctx):
            ctx.set(dut.x, 255)
            ctx.set(dut.y, 255)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 65025, f"Expected 65025, got {o}"

        _run(dut, bench, "test_dsp_block_max.vcd")

    def test_default_widths(self):
        """Default 18×25: 100 * 200 = 20000."""
        dut = DSPBlock()  # default x_width=18, y_width=25

        async def bench(ctx):
            ctx.set(dut.x, 100)
            ctx.set(dut.y, 200)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 20000, f"Expected 20000, got {o}"

        _run(dut, bench, "test_dsp_block_default.vcd")


# ===================================================================
# 10. FixMultAdd  (a, b, c → o = a*b + c, latency=3)
# ===================================================================
class TestFixMultAdd:
    """Fixed-point multiply-accumulate."""

    def test_3x4_plus_5(self):
        """3*4 + 5 = 17."""
        dut = FixMultAdd(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 3)
            ctx.set(dut.b, 4)
            ctx.set(dut.c, 5)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 17, f"Expected 17, got {o}"

        _run(dut, bench, "test_fix_mult_add_3x4p5.vcd")

    def test_0x0_plus_0(self):
        """0*0 + 0 = 0."""
        dut = FixMultAdd(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 0)
            ctx.set(dut.b, 0)
            ctx.set(dut.c, 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0, f"Expected 0, got {o}"

        _run(dut, bench, "test_fix_mult_add_0x0p0.vcd")

    def test_1x1_plus_1(self):
        """1*1 + 1 = 2."""
        dut = FixMultAdd(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 1)
            ctx.set(dut.b, 1)
            ctx.set(dut.c, 1)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 2, f"Expected 2, got {o}"

        _run(dut, bench, "test_fix_mult_add_1x1p1.vcd")

    def test_10x10_plus_0(self):
        """10*10 + 0 = 100."""
        dut = FixMultAdd(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 10)
            ctx.set(dut.b, 10)
            ctx.set(dut.c, 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 100, f"Expected 100, got {o}"

        _run(dut, bench, "test_fix_mult_add_10x10p0.vcd")


# ===================================================================
# 11. IntMultiplierLUT  (x, y → o = x * y, latency=1)
# ===================================================================
class TestIntMultiplierLUT:
    """Integer multiplier using LUTs."""

    def test_3_times_4(self):
        """3 * 4 = 12."""
        dut = IntMultiplierLUT(x_width=8, y_width=8)

        async def bench(ctx):
            ctx.set(dut.x, 3)
            ctx.set(dut.y, 4)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 12, f"Expected 12, got {o}"

        _run(dut, bench, "test_int_mult_lut_3x4.vcd")

    def test_0_times_any(self):
        """0 * 42 = 0."""
        dut = IntMultiplierLUT(x_width=8, y_width=8)

        async def bench(ctx):
            ctx.set(dut.x, 0)
            ctx.set(dut.y, 42)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0, f"Expected 0, got {o}"

        _run(dut, bench, "test_int_mult_lut_0x42.vcd")

    def test_7_times_8(self):
        """7 * 8 = 56."""
        dut = IntMultiplierLUT(x_width=8, y_width=8)

        async def bench(ctx):
            ctx.set(dut.x, 7)
            ctx.set(dut.y, 8)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 56, f"Expected 56, got {o}"

        _run(dut, bench, "test_int_mult_lut_7x8.vcd")


# ===================================================================
# 12. BaseSquarerLUT  (x → o = x^2, latency=1)
# ===================================================================
class TestBaseSquarerLUT:
    """Base squarer using LUT."""

    def test_square_7(self):
        """7^2 = 49."""
        dut = BaseSquarerLUT(width=8)

        async def bench(ctx):
            ctx.set(dut.x, 7)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 49, f"Expected 49, got {o}"

        _run(dut, bench, "test_base_squarer_lut_7.vcd")

    def test_square_0(self):
        """0^2 = 0."""
        dut = BaseSquarerLUT(width=8)

        async def bench(ctx):
            ctx.set(dut.x, 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0, f"Expected 0, got {o}"

        _run(dut, bench, "test_base_squarer_lut_0.vcd")

    def test_square_1(self):
        """1^2 = 1."""
        dut = BaseSquarerLUT(width=8)

        async def bench(ctx):
            ctx.set(dut.x, 1)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 1, f"Expected 1, got {o}"

        _run(dut, bench, "test_base_squarer_lut_1.vcd")

    def test_square_10(self):
        """10^2 = 100."""
        dut = BaseSquarerLUT(width=8)

        async def bench(ctx):
            ctx.set(dut.x, 10)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 100, f"Expected 100, got {o}"

        _run(dut, bench, "test_base_squarer_lut_10.vcd")


# ===================================================================
# 13. IntMultiAdder  (inputs[0..n-1] → o, latency=2)
# ===================================================================
class TestIntMultiAdder:
    """Integer multi-operand adder."""

    def test_3_inputs(self):
        """1 + 2 + 3 = 6."""
        dut = IntMultiAdder(width=8, n_inputs=3)

        async def bench(ctx):
            ctx.set(dut.inputs[0], 1)
            ctx.set(dut.inputs[1], 2)
            ctx.set(dut.inputs[2], 3)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 6, f"Expected 6, got {o}"

        _run(dut, bench, "test_int_multi_adder_3.vcd")

    def test_4_inputs(self):
        """10 + 20 + 30 + 40 = 100."""
        dut = IntMultiAdder(width=8, n_inputs=4)

        async def bench(ctx):
            ctx.set(dut.inputs[0], 10)
            ctx.set(dut.inputs[1], 20)
            ctx.set(dut.inputs[2], 30)
            ctx.set(dut.inputs[3], 40)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 100, f"Expected 100, got {o}"

        _run(dut, bench, "test_int_multi_adder_4.vcd")

    def test_all_zeros(self):
        """0 + 0 + 0 = 0."""
        dut = IntMultiAdder(width=8, n_inputs=3)

        async def bench(ctx):
            ctx.set(dut.inputs[0], 0)
            ctx.set(dut.inputs[1], 0)
            ctx.set(dut.inputs[2], 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0, f"Expected 0, got {o}"

        _run(dut, bench, "test_int_multi_adder_zeros.vcd")

    def test_2_inputs(self):
        """100 + 50 = 150."""
        dut = IntMultiAdder(width=8, n_inputs=2)

        async def bench(ctx):
            ctx.set(dut.inputs[0], 100)
            ctx.set(dut.inputs[1], 50)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 150, f"Expected 150, got {o}"

        _run(dut, bench, "test_int_multi_adder_2.vcd")


# ===================================================================
# 14. FixMultiAdder  (inputs[0..n-1] → o, latency=2)
# ===================================================================
class TestFixMultiAdder:
    """Fixed-point multi-operand adder."""

    def test_3_inputs(self):
        """5 + 10 + 15 = 30."""
        dut = FixMultiAdder(width=8, n_inputs=3)

        async def bench(ctx):
            ctx.set(dut.inputs[0], 5)
            ctx.set(dut.inputs[1], 10)
            ctx.set(dut.inputs[2], 15)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 30, f"Expected 30, got {o}"

        _run(dut, bench, "test_fix_multi_adder_3.vcd")

    def test_all_zeros(self):
        """0 + 0 + 0 = 0."""
        dut = FixMultiAdder(width=8, n_inputs=3)

        async def bench(ctx):
            ctx.set(dut.inputs[0], 0)
            ctx.set(dut.inputs[1], 0)
            ctx.set(dut.inputs[2], 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0, f"Expected 0, got {o}"

        _run(dut, bench, "test_fix_multi_adder_zeros.vcd")

    def test_4_inputs(self):
        """1 + 2 + 3 + 4 = 10."""
        dut = FixMultiAdder(width=8, n_inputs=4)

        async def bench(ctx):
            ctx.set(dut.inputs[0], 1)
            ctx.set(dut.inputs[1], 2)
            ctx.set(dut.inputs[2], 3)
            ctx.set(dut.inputs[3], 4)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 10, f"Expected 10, got {o}"

        _run(dut, bench, "test_fix_multi_adder_4.vcd")

    def test_max_values(self):
        """255 + 255 + 255 = 765."""
        dut = FixMultiAdder(width=8, n_inputs=3)

        async def bench(ctx):
            ctx.set(dut.inputs[0], 255)
            ctx.set(dut.inputs[1], 255)
            ctx.set(dut.inputs[2], 255)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 765, f"Expected 765, got {o}"

        _run(dut, bench, "test_fix_multi_adder_max.vcd")
