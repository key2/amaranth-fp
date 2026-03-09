"""Tests for pipelined InputIEEE and OutputIEEE conversions."""

from amaranth import Module
from amaranth.sim import Simulator

from amaranth_fp.format import FPFormat
from amaranth_fp.conversions import InputIEEE, OutputIEEE
from conftest import encode_ieee, decode_exc, decode_sign, decode_exp, decode_mant

FMT = FPFormat.half()


def _run_clocked(dut, testbench):
    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test.vcd"):
        sim.run()


class TestInputIEEE:
    def test_positive_zero(self):
        dut = InputIEEE(FMT)

        async def bench(ctx):
            ctx.set(dut.ieee_in, encode_ieee(FMT, 0, 0, 0))
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.fp_out)
            assert decode_exc(FMT, result) == 0b00
            assert decode_sign(FMT, result) == 0

        _run_clocked(dut, bench)

    def test_negative_zero(self):
        dut = InputIEEE(FMT)

        async def bench(ctx):
            ctx.set(dut.ieee_in, encode_ieee(FMT, 1, 0, 0))
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.fp_out)
            assert decode_exc(FMT, result) == 0b00
            assert decode_sign(FMT, result) == 1

        _run_clocked(dut, bench)

    def test_positive_inf(self):
        dut = InputIEEE(FMT)

        async def bench(ctx):
            ctx.set(dut.ieee_in, encode_ieee(FMT, 0, 0b11111, 0))
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.fp_out)
            assert decode_exc(FMT, result) == 0b10
            assert decode_sign(FMT, result) == 0

        _run_clocked(dut, bench)

    def test_negative_inf(self):
        dut = InputIEEE(FMT)

        async def bench(ctx):
            ctx.set(dut.ieee_in, encode_ieee(FMT, 1, 0b11111, 0))
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.fp_out)
            assert decode_exc(FMT, result) == 0b10
            assert decode_sign(FMT, result) == 1

        _run_clocked(dut, bench)

    def test_nan(self):
        dut = InputIEEE(FMT)

        async def bench(ctx):
            ctx.set(dut.ieee_in, encode_ieee(FMT, 0, 0b11111, 1))
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.fp_out)
            assert decode_exc(FMT, result) == 0b11

        _run_clocked(dut, bench)

    def test_normal_positive_one(self):
        dut = InputIEEE(FMT)

        async def bench(ctx):
            ctx.set(dut.ieee_in, encode_ieee(FMT, 0, 15, 0))
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.fp_out)
            assert decode_exc(FMT, result) == 0b01
            assert decode_sign(FMT, result) == 0
            assert decode_exp(FMT, result) == 15
            assert decode_mant(FMT, result) == 0

        _run_clocked(dut, bench)

    def test_normal_negative(self):
        dut = InputIEEE(FMT)

        async def bench(ctx):
            ctx.set(dut.ieee_in, encode_ieee(FMT, 1, 15, 0b1000000000))
            for _ in range(dut.latency):
                await ctx.tick()
            result = ctx.get(dut.fp_out)
            assert decode_exc(FMT, result) == 0b01
            assert decode_sign(FMT, result) == 1
            assert decode_exp(FMT, result) == 15
            assert decode_mant(FMT, result) == 0b1000000000

        _run_clocked(dut, bench)


class TestOutputIEEERoundtrip:
    def _roundtrip(self, ieee_val):
        inp = InputIEEE(FMT)
        out = OutputIEEE(FMT)
        m = Module()
        m.submodules.inp = inp
        m.submodules.out = out
        # Connect: inp output -> out input (both pipelined)
        from amaranth import Signal
        m.d.comb += out.fp_in.eq(inp.fp_out)

        async def bench(ctx):
            ctx.set(inp.ieee_in, ieee_val)
            # InputIEEE latency=1, OutputIEEE latency=1, total=2
            for _ in range(2):
                await ctx.tick()
            assert ctx.get(out.ieee_out) == ieee_val

        sim = Simulator(m)
        sim.add_clock(1e-6)
        sim.add_testbench(bench)
        with sim.write_vcd("test_roundtrip.vcd"):
            sim.run()

    def test_roundtrip_zero(self):
        self._roundtrip(encode_ieee(FMT, 0, 0, 0))

    def test_roundtrip_one(self):
        self._roundtrip(encode_ieee(FMT, 0, 15, 0))

    def test_roundtrip_inf(self):
        self._roundtrip(encode_ieee(FMT, 0, 0b11111, 0))
