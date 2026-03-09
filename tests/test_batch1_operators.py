"""Tests for Batch 1 operators: integer, FP variants, constant mult, filters."""
from amaranth import *
from amaranth.sim import Simulator

from amaranth_fp.format import FPFormat
from amaranth_fp.integer import IntAdder, IntMultiplier, IntComparator
from amaranth_fp.operators import FPAddSub, IEEEFPAdd, FixRealKCM
from amaranth_fp.filters import FixFIR

from conftest import encode_fp, fp_normal, decode_exc, decode_sign, decode_exp, decode_mant, encode_ieee


def sim_run(dut, process, *, clocks=20):
    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(process)
    with sim.write_vcd("test_batch1.vcd"):
        sim.run()


def test_int_adder_3_plus_5():
    dut = IntAdder(8)

    async def proc(ctx):
        ctx.set(dut.a, 3)
        ctx.set(dut.b, 5)
        ctx.set(dut.cin, 0)
        await ctx.tick()
        await ctx.tick()
        s = ctx.get(dut.s)
        cout = ctx.get(dut.cout)
        assert s == 8, f"Expected 8, got {s}"
        assert cout == 0

    sim_run(dut, proc)


def test_int_multiplier_3_times_4():
    dut = IntMultiplier(8, 8)

    async def proc(ctx):
        ctx.set(dut.a, 3)
        ctx.set(dut.b, 4)
        await ctx.tick()
        await ctx.tick()
        await ctx.tick()
        p = ctx.get(dut.p)
        assert p == 12, f"Expected 12, got {p}"

    sim_run(dut, proc)


def test_int_comparator_3_lt_5():
    dut = IntComparator(8)

    async def proc(ctx):
        ctx.set(dut.a, 3)
        ctx.set(dut.b, 5)
        await ctx.tick()
        await ctx.tick()
        lt = ctx.get(dut.lt)
        eq = ctx.get(dut.eq)
        gt = ctx.get(dut.gt)
        assert lt == 1, f"Expected lt=1, got {lt}"
        assert eq == 0
        assert gt == 0

    sim_run(dut, proc)


def test_fp_add_sub_add_mode():
    """FPAddSub in add mode should behave like FPAdd."""
    fmt = FPFormat(8, 23)
    dut = FPAddSub(fmt)

    # 1.0 + 1.0 = 2.0
    a = fp_normal(fmt, 0, fmt.bias, 0)  # 1.0
    b = fp_normal(fmt, 0, fmt.bias, 0)  # 1.0

    async def proc(ctx):
        ctx.set(dut.a, a)
        ctx.set(dut.b, b)
        ctx.set(dut.op, 0)  # add
        for _ in range(9):
            await ctx.tick()
        o = ctx.get(dut.o)
        exc = decode_exc(fmt, o)
        sign = decode_sign(fmt, o)
        exp = decode_exp(fmt, o)
        # 2.0 = 1.mantissa=0, exponent=bias+1
        assert exc == 0b01, f"exc={exc:#04b}"
        assert sign == 0
        assert exp == fmt.bias + 1, f"exp={exp}, expected {fmt.bias + 1}"

    sim_run(dut, proc)


def test_fp_add_sub_sub_mode():
    """FPAddSub in sub mode: 1.0 - 1.0 = 0."""
    fmt = FPFormat(8, 23)
    dut = FPAddSub(fmt)

    a = fp_normal(fmt, 0, fmt.bias, 0)
    b = fp_normal(fmt, 0, fmt.bias, 0)

    async def proc(ctx):
        ctx.set(dut.a, a)
        ctx.set(dut.b, b)
        ctx.set(dut.op, 1)  # sub
        for _ in range(9):
            await ctx.tick()
        o = ctx.get(dut.o)
        exc = decode_exc(fmt, o)
        assert exc == 0b00, f"exc={exc:#04b}, expected zero"

    sim_run(dut, proc)


def test_ieee_fp_add_basic():
    """IEEEFPAdd: 1.0 + 1.0 in IEEE format."""
    fmt = FPFormat(8, 23)
    dut = IEEEFPAdd(fmt)

    a_ieee = encode_ieee(fmt, 0, fmt.bias, 0)  # 1.0
    b_ieee = encode_ieee(fmt, 0, fmt.bias, 0)  # 1.0

    async def proc(ctx):
        ctx.set(dut.a, a_ieee)
        ctx.set(dut.b, b_ieee)
        for _ in range(12):
            await ctx.tick()
        o = ctx.get(dut.o)
        # 2.0 in IEEE: sign=0, exp=bias+1, mant=0
        expected_exp = fmt.bias + 1
        got_exp = (o >> fmt.wf) & ((1 << fmt.we) - 1)
        got_sign = (o >> (fmt.we + fmt.wf)) & 1
        got_mant = o & ((1 << fmt.wf) - 1)
        assert got_sign == 0, f"sign={got_sign}"
        assert got_exp == expected_exp, f"exp={got_exp}, expected {expected_exp}"
        assert got_mant == 0, f"mant={got_mant}"

    sim_run(dut, proc)


def test_fix_fir_2tap():
    """2-tap FIR filter with coefficients [1, 1]: output = x[n] + x[n-1]."""
    dut = FixFIR(input_width=8, output_width=16, coefficients=[1, 1], coeff_width=8)

    async def proc(ctx):
        # Transposed FIR [1,1]: acc[1]=sync(1*x), acc[0]=sync(1*x+acc[1])
        # y = acc[0] (comb)
        # Set x=5 → tick1: acc[1]=5, acc[0]=5+0=5
        # Set x=3 → tick2: acc[1]=3, acc[0]=3+5=8
        ctx.set(dut.x, 5)
        await ctx.tick()
        ctx.set(dut.x, 3)
        await ctx.tick()
        y = ctx.get(dut.y)
        assert y == 8, f"Expected 8, got {y}"

    sim_run(dut, proc)


def test_fix_real_kcm_half():
    """FixRealKCM: multiply by 0.5 — for 8-bit input, output should be input/2."""
    dut = FixRealKCM(input_width=8, constant=0.5, output_width=8)

    async def proc(ctx):
        ctx.set(dut.x, 100)
        await ctx.tick()
        await ctx.tick()
        await ctx.tick()
        p = ctx.get(dut.p)
        # 100 * 0.5 = 50 (approximately, depending on fixed-point scaling)
        # The KCM computes (x * c_fixed) >> ow where c_fixed = round(0.5 * 256) = 128
        # So (100 * 128) >> 8 = 12800 >> 8 = 50
        assert p == 50, f"Expected 50, got {p}"

    sim_run(dut, proc)
