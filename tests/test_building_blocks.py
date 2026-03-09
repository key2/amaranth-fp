"""Tests for building blocks using Amaranth simulator."""

from amaranth.sim import Simulator

from amaranth_fp.building_blocks import Shifter, LeadingZeroCounter, Normalizer, RoundingUnit


def _run_comb(dut, testbench):
    """Run a combinational testbench (no clock needed)."""
    sim = Simulator(dut)
    sim.add_testbench(testbench)
    with sim.write_vcd("test.vcd"):
        sim.run()


# ---------------------------------------------------------------------------
# Shifter tests
# ---------------------------------------------------------------------------

class TestShifter:
    def test_left_shift_by_0(self):
        dut = Shifter(width=8, shift_width=3, direction="left")

        async def bench(ctx):
            ctx.set(dut.i, 0b00001111)
            ctx.set(dut.shift, 0)
            assert ctx.get(dut.o) == 0b00001111

        _run_comb(dut, bench)

    def test_left_shift_by_2(self):
        dut = Shifter(width=8, shift_width=3, direction="left")

        async def bench(ctx):
            ctx.set(dut.i, 0b00000011)
            ctx.set(dut.shift, 2)
            assert ctx.get(dut.o) == 0b00001100

        _run_comb(dut, bench)

    def test_right_shift_logical(self):
        dut = Shifter(width=8, shift_width=3, direction="right", arithmetic=False)

        async def bench(ctx):
            ctx.set(dut.i, 0b11110000)
            ctx.set(dut.shift, 4)
            assert ctx.get(dut.o) == 0b00001111

        _run_comb(dut, bench)

    def test_right_shift_by_0(self):
        dut = Shifter(width=8, shift_width=3, direction="right")

        async def bench(ctx):
            ctx.set(dut.i, 0b10101010)
            ctx.set(dut.shift, 0)
            assert ctx.get(dut.o) == 0b10101010

        _run_comb(dut, bench)


# ---------------------------------------------------------------------------
# LeadingZeroCounter tests
# ---------------------------------------------------------------------------

class TestLeadingZeroCounter:
    def test_all_zeros_8bit(self):
        dut = LeadingZeroCounter(8)

        async def bench(ctx):
            ctx.set(dut.i, 0)
            assert ctx.get(dut.all_zeros) == 1

        _run_comb(dut, bench)

    def test_msb_set(self):
        dut = LeadingZeroCounter(8)

        async def bench(ctx):
            ctx.set(dut.i, 0b10000000)
            assert ctx.get(dut.count) == 0
            assert ctx.get(dut.all_zeros) == 0

        _run_comb(dut, bench)

    def test_one_leading_zero(self):
        dut = LeadingZeroCounter(8)

        async def bench(ctx):
            ctx.set(dut.i, 0b01000000)
            assert ctx.get(dut.count) == 1

        _run_comb(dut, bench)

    def test_five_leading_zeros(self):
        dut = LeadingZeroCounter(8)

        async def bench(ctx):
            ctx.set(dut.i, 0b00000100)
            assert ctx.get(dut.count) == 5

        _run_comb(dut, bench)

    def test_lsb_only(self):
        dut = LeadingZeroCounter(8)

        async def bench(ctx):
            ctx.set(dut.i, 0b00000001)
            assert ctx.get(dut.count) == 7

        _run_comb(dut, bench)


# ---------------------------------------------------------------------------
# Normalizer tests
# ---------------------------------------------------------------------------

class TestNormalizer:
    def test_already_normalized(self):
        dut = Normalizer(input_width=8, output_width=8)

        async def bench(ctx):
            ctx.set(dut.i, 0b10000000)
            assert ctx.get(dut.count) == 0
            assert ctx.get(dut.o) == 0b10000000

        _run_comb(dut, bench)

    def test_shift_by_3(self):
        dut = Normalizer(input_width=8, output_width=8)

        async def bench(ctx):
            ctx.set(dut.i, 0b00010000)
            assert ctx.get(dut.count) == 3
            assert ctx.get(dut.o) == 0b10000000

        _run_comb(dut, bench)


# ---------------------------------------------------------------------------
# RoundingUnit tests
# ---------------------------------------------------------------------------

class TestRoundingUnit:
    def test_no_rounding_needed(self):
        dut = RoundingUnit(width=4)

        async def bench(ctx):
            # [mantissa(4) | G | R | S] = 0b1010_000
            ctx.set(dut.mantissa_in, 0b1010_000)
            assert ctx.get(dut.mantissa_out) == 0b1010
            assert ctx.get(dut.overflow) == 0

        _run_comb(dut, bench)

    def test_round_up(self):
        dut = RoundingUnit(width=4)

        async def bench(ctx):
            # [mantissa(4) | G | R | S] = 0b1010_110
            ctx.set(dut.mantissa_in, 0b1010_110)
            assert ctx.get(dut.mantissa_out) == 0b1011
            assert ctx.get(dut.overflow) == 0

        _run_comb(dut, bench)

    def test_tie_to_even_round_down(self):
        dut = RoundingUnit(width=4)

        async def bench(ctx):
            # mantissa=0b1010, G=1, R=0, S=0 -> LSB=0, no round up
            ctx.set(dut.mantissa_in, 0b1010_100)
            assert ctx.get(dut.mantissa_out) == 0b1010
            assert ctx.get(dut.overflow) == 0

        _run_comb(dut, bench)

    def test_tie_to_even_round_up(self):
        dut = RoundingUnit(width=4)

        async def bench(ctx):
            # mantissa=0b1011, G=1, R=0, S=0 -> LSB=1, round up
            ctx.set(dut.mantissa_in, 0b1011_100)
            assert ctx.get(dut.mantissa_out) == 0b1100
            assert ctx.get(dut.overflow) == 0

        _run_comb(dut, bench)

    def test_overflow(self):
        dut = RoundingUnit(width=4)

        async def bench(ctx):
            # mantissa=0b1111, G=1, R=1, S=0 -> rounds to overflow
            ctx.set(dut.mantissa_in, 0b1111_110)
            assert ctx.get(dut.mantissa_out) == 0b0000
            assert ctx.get(dut.overflow) == 1

        _run_comb(dut, bench)
