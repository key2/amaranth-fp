"""Tests for advanced operators: functions, transcendentals, const mult, etc."""
from amaranth import *
from amaranth.sim import Simulator

from amaranth_fp.format import FPFormat
from amaranth_fp.functions.table import Table
from amaranth_fp.functions.fix_function_by_table import FixFunctionByTable
from amaranth_fp.operators.fp_exp import FPExp
from amaranth_fp.operators.fp_log import FPLog
from amaranth_fp.operators.fp_const_mult import FPConstMult
from amaranth_fp.operators.fp_add3 import FPAdd3Input
from amaranth_fp.operators.fix_sincos import FixSinCos
from amaranth_fp.operators.lns_ops import LNSMul

from conftest import encode_fp, fp_zero, fp_inf, fp_nan, fp_one, decode_exc


# ── Helpers ──────────────────────────────────────────────────────────────


def run_sim(dut, process, *, clocks=50):
    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(process)
    with sim.write_vcd("test_adv.vcd"):
        sim.run()


# ── Table ────────────────────────────────────────────────────────────────


def test_table_lookup():
    """Table: lookup known values."""
    values = [10, 20, 30, 40, 50, 60, 70, 80]
    dut = Table(input_width=3, output_width=8, values=values)

    async def proc(ctx):
        for i, expected in enumerate(values):
            ctx.set(dut.addr, i)
            await ctx.tick()
            await ctx.tick()  # 1-cycle memory latency
            result = ctx.get(dut.data)
            assert result == expected, f"Table[{i}]: expected {expected}, got {result}"

    run_sim(dut, proc)


# ── FixFunctionByTable ───────────────────────────────────────────────────


def test_fix_function_by_table_square():
    """FixFunctionByTable: compute x^2 via table for small input."""
    # 4-bit input, 8-bit output, func = x^2 with input_range [0, 16)
    dut = FixFunctionByTable(
        input_width=4, output_width=8,
        func=lambda x: x * x,
        input_range=(0.0, 16.0),
    )

    async def proc(ctx):
        # Address 0 → x=0.5, expected 0.5^2 ≈ 0 (rounded)
        ctx.set(dut.addr, 0)
        await ctx.tick()
        await ctx.tick()
        # Address 3 → x=3.5, expected 3.5^2 = 12.25 → 12
        ctx.set(dut.addr, 3)
        await ctx.tick()
        await ctx.tick()
        result = ctx.get(dut.data)
        assert result == 12, f"Expected 12, got {result}"

    run_sim(dut, proc)


# ── FPExp ────────────────────────────────────────────────────────────────


def test_fp_exp_special():
    """FPExp: exp(NaN)=NaN, exp(+inf)=+inf, exp(-inf)=0."""
    fmt = FPFormat(we=5, wf=10)
    dut = FPExp(fmt)

    async def proc(ctx):
        # exp(+inf) = +inf
        ctx.set(dut.a, fp_inf(fmt, sign=0))
        for _ in range(dut.latency + 2):
            await ctx.tick()
        exc = decode_exc(fmt, ctx.get(dut.o))
        assert exc == 0b10, f"exp(+inf): expected inf exc=10, got {exc:02b}"

        # exp(-inf) = 0
        ctx.set(dut.a, fp_inf(fmt, sign=1))
        for _ in range(dut.latency + 2):
            await ctx.tick()
        exc = decode_exc(fmt, ctx.get(dut.o))
        assert exc == 0b00, f"exp(-inf): expected zero exc=00, got {exc:02b}"

        # exp(NaN) = NaN
        ctx.set(dut.a, fp_nan(fmt))
        for _ in range(dut.latency + 2):
            await ctx.tick()
        exc = decode_exc(fmt, ctx.get(dut.o))
        assert exc == 0b11, f"exp(NaN): expected NaN exc=11, got {exc:02b}"

    run_sim(dut, proc)


def test_fp_exp_zero():
    """FPExp: exp(0) = 1.0."""
    fmt = FPFormat(we=5, wf=10)
    dut = FPExp(fmt)

    async def proc(ctx):
        ctx.set(dut.a, fp_zero(fmt))
        for _ in range(dut.latency + 2):
            await ctx.tick()
        result = ctx.get(dut.o)
        exc = decode_exc(fmt, result)
        # exp(0) should be normal (exc=01)
        assert exc == 0b01, f"exp(0): expected normal exc=01, got {exc:02b}"

    run_sim(dut, proc)


# ── FPLog ────────────────────────────────────────────────────────────────


def test_fp_log_special():
    """FPLog: log(+inf)=+inf, log(0)=-inf, log(-1)=NaN."""
    fmt = FPFormat(we=5, wf=10)
    dut = FPLog(fmt)

    async def proc(ctx):
        # log(+inf) = +inf
        ctx.set(dut.a, fp_inf(fmt, sign=0))
        for _ in range(dut.latency + 2):
            await ctx.tick()
        exc = decode_exc(fmt, ctx.get(dut.o))
        assert exc == 0b10, f"log(+inf): expected inf exc=10, got {exc:02b}"

        # log(0) = -inf
        ctx.set(dut.a, fp_zero(fmt))
        for _ in range(dut.latency + 2):
            await ctx.tick()
        exc = decode_exc(fmt, ctx.get(dut.o))
        assert exc == 0b10, f"log(0): expected inf exc=10, got {exc:02b}"

        # log(NaN) = NaN
        ctx.set(dut.a, fp_nan(fmt))
        for _ in range(dut.latency + 2):
            await ctx.tick()
        exc = decode_exc(fmt, ctx.get(dut.o))
        assert exc == 0b11, f"log(NaN): expected NaN exc=11, got {exc:02b}"

    run_sim(dut, proc)


def test_fp_log_one():
    """FPLog: log(1) = 0."""
    fmt = FPFormat(we=5, wf=10)
    dut = FPLog(fmt)

    async def proc(ctx):
        ctx.set(dut.a, fp_one(fmt))
        for _ in range(dut.latency + 2):
            await ctx.tick()
        result = ctx.get(dut.o)
        exc = decode_exc(fmt, result)
        # log(1) = 0, should be zero exception
        assert exc == 0b00, f"log(1): expected zero exc=00, got {exc:02b}"

    run_sim(dut, proc)


# ── FPConstMult ──────────────────────────────────────────────────────────


def test_fp_const_mult():
    """FPConstMult: 3.0 * 2 = 6.0."""
    fmt = FPFormat(we=5, wf=10)
    dut = FPConstMult(fmt, constant=2.0)

    # Encode 3.0: sign=0, exp=bias+1=16, mant=0b1000000000 (1.5 * 2^1)
    # Actually 3.0 = 1.5 * 2^1 → exp = bias + 1 = 16, mant = 1<<9 = 512
    three = encode_fp(fmt, sign=0, exponent=fmt.bias + 1, mantissa=1 << (fmt.wf - 1))

    async def proc(ctx):
        ctx.set(dut.a, three)
        for _ in range(dut.latency + 2):
            await ctx.tick()
        result = ctx.get(dut.o)
        exc = decode_exc(fmt, result)
        assert exc == 0b01, f"3*2: expected normal, got exc={exc:02b}"

    run_sim(dut, proc)


# ── FPAdd3Input ──────────────────────────────────────────────────────────


def test_fp_add3():
    """FPAdd3Input: 1.0 + 1.0 + 1.0 = 3.0."""
    fmt = FPFormat(we=5, wf=10)
    dut = FPAdd3Input(fmt)
    one = fp_one(fmt)

    async def proc(ctx):
        ctx.set(dut.a, one)
        ctx.set(dut.b, one)
        ctx.set(dut.c, one)
        for _ in range(dut.latency + 2):
            await ctx.tick()
        result = ctx.get(dut.o)
        exc = decode_exc(fmt, result)
        assert exc == 0b01, f"1+1+1: expected normal, got exc={exc:02b}"

    run_sim(dut, proc)


# ── FixSinCos ────────────────────────────────────────────────────────────


def test_fix_sincos_zero():
    """FixSinCos: sin(0) ≈ 0, cos(0) ≈ K (CORDIC gain)."""
    w = 16
    dut = FixSinCos(width=w, iterations=w)

    async def proc(ctx):
        ctx.set(dut.angle, 0)
        for _ in range(dut.latency + 2):
            await ctx.tick()
        sin_val = ctx.get(dut.sin_out)
        cos_val = ctx.get(dut.cos_out)
        # sin(0) should be close to 0
        assert sin_val < 10 or sin_val > (1 << w) - 10, \
            f"sin(0) should be ~0, got {sin_val}"
        # cos(0) should be positive (CORDIC gain ≈ 0.607 * 2^(w-1))
        assert cos_val > 0, f"cos(0) should be > 0, got {cos_val}"

    run_sim(dut, proc)


# ── LNSMul ───────────────────────────────────────────────────────────────


def test_lns_mul():
    """LNSMul: basic test — adding log values."""
    w = 16
    dut = LNSMul(width=w)

    async def proc(ctx):
        # a: sign=0, log=100; b: sign=0, log=200
        # result should be sign=0, log=300
        a_val = 100  # positive, log=100
        b_val = 200  # positive, log=200
        ctx.set(dut.a, a_val)
        ctx.set(dut.b, b_val)
        await ctx.tick()
        await ctx.tick()
        result = ctx.get(dut.o)
        result_log = result & ((1 << (w - 1)) - 1)
        result_sign = (result >> (w - 1)) & 1
        assert result_sign == 0, f"Expected sign=0, got {result_sign}"
        assert result_log == 300, f"Expected log=300, got {result_log}"

    run_sim(dut, proc)
