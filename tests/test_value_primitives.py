"""Value-level tests for primitive components.

Tests: GenericLut, GenericMult, GenericMux, RowAdder, BooleanEquation,
       XilinxMUXF7, XilinxMUXF8, XilinxFDCE, XilinxFourToTwoCompressor,
       XilinxTernaryAddSub, XilinxN2MDecoder, XilinxGenericMux,
       XilinxLUT5, XilinxLUT6, IntelTernaryAdder.
"""
import pytest
from amaranth.sim import Simulator

from amaranth_fp.primitives import (
    GenericLut,
    GenericMult,
    GenericMux,
    RowAdder,
    BooleanEquation,
)
from amaranth_fp.primitives.xilinx import (
    XilinxMUXF7,
    XilinxMUXF8,
    XilinxFDCE,
    XilinxFourToTwoCompressor,
    XilinxTernaryAddSub,
    XilinxN2MDecoder,
    XilinxGenericMux,
    XilinxLUT5,
    XilinxLUT6,
)
from amaranth_fp.primitives.intel import IntelTernaryAdder


def _run(dut, testbench, vcd_name="test_prim.vcd"):
    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd(vcd_name):
        sim.run()


def _run_comb(dut, testbench, vcd_name="test_prim_comb.vcd"):
    """Run a combinational (latency=0) DUT — no clock needed.

    Wraps the testbench to replace ctx.tick() with ctx.delay() automatically.
    """
    import types

    async def _comb_bench(ctx):
        # Monkey-patch ctx.tick to use delay for combinational circuits
        original_tick = ctx.tick
        async def _delay_tick(*args, **kwargs):
            await ctx.delay(1e-6)
        ctx.tick = _delay_tick
        await testbench(ctx)

    sim = Simulator(dut)
    sim.add_testbench(_comb_bench)
    with sim.write_vcd(vcd_name):
        sim.run()


# ===================================================================
# 1. GenericLut  (addr → data, latency=1, uses Memory read port)
# ===================================================================
class TestGenericLut:
    """Configurable lookup table."""

    def test_identity_table(self):
        """Table[i] = i for 4-bit input, 8-bit output."""
        contents = list(range(16))
        dut = GenericLut(input_width=4, output_width=8, contents=contents)

        async def bench(ctx):
            for addr in [0, 5, 10, 15]:
                ctx.set(dut.addr, addr)
                for _ in range(dut.latency + 1):
                    await ctx.tick()
                data = ctx.get(dut.data)
                assert data == addr, f"addr={addr}: expected {addr}, got {data}"

        _run(dut, bench, "test_generic_lut_identity.vcd")

    def test_square_table(self):
        """Table[i] = i^2 for 3-bit input."""
        contents = [i * i for i in range(8)]
        dut = GenericLut(input_width=3, output_width=8, contents=contents)

        async def bench(ctx):
            for addr in range(8):
                ctx.set(dut.addr, addr)
                for _ in range(dut.latency + 1):
                    await ctx.tick()
                data = ctx.get(dut.data)
                assert data == addr * addr, f"addr={addr}: expected {addr*addr}, got {data}"

        _run(dut, bench, "test_generic_lut_square.vcd")

    def test_constant_table(self):
        """Table[i] = 42 for all entries."""
        contents = [42] * 4
        dut = GenericLut(input_width=2, output_width=8, contents=contents)

        async def bench(ctx):
            for addr in range(4):
                ctx.set(dut.addr, addr)
                for _ in range(dut.latency + 1):
                    await ctx.tick()
                data = ctx.get(dut.data)
                assert data == 42, f"addr={addr}: expected 42, got {data}"

        _run(dut, bench, "test_generic_lut_constant.vcd")


# ===================================================================
# 2. GenericMult  (a, b → o = a*b, latency=1)
# ===================================================================
class TestGenericMult:
    """Parameterised unsigned multiplier."""

    def test_3_times_4(self):
        """3 × 4 = 12."""
        dut = GenericMult(a_width=4, b_width=4)

        async def bench(ctx):
            ctx.set(dut.a, 3)
            ctx.set(dut.b, 4)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 12, f"Expected 12, got {o}"

        _run(dut, bench, "test_generic_mult_3x4.vcd")

    def test_0_times_x(self):
        """0 × 15 = 0."""
        dut = GenericMult(a_width=4, b_width=4)

        async def bench(ctx):
            ctx.set(dut.a, 0)
            ctx.set(dut.b, 15)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0, f"Expected 0, got {o}"

        _run(dut, bench, "test_generic_mult_0x15.vcd")

    def test_max_values(self):
        """15 × 15 = 225."""
        dut = GenericMult(a_width=4, b_width=4)

        async def bench(ctx):
            ctx.set(dut.a, 15)
            ctx.set(dut.b, 15)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 225, f"Expected 225, got {o}"

        _run(dut, bench, "test_generic_mult_15x15.vcd")

    def test_asymmetric_widths(self):
        """3-bit × 5-bit: 7 × 31 = 217."""
        dut = GenericMult(a_width=3, b_width=5)

        async def bench(ctx):
            ctx.set(dut.a, 7)
            ctx.set(dut.b, 31)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 217, f"Expected 217, got {o}"

        _run(dut, bench, "test_generic_mult_asym.vcd")

    def test_1_times_1(self):
        """1 × 1 = 1."""
        dut = GenericMult(a_width=4, b_width=4)

        async def bench(ctx):
            ctx.set(dut.a, 1)
            ctx.set(dut.b, 1)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 1, f"Expected 1, got {o}"

        _run(dut, bench, "test_generic_mult_1x1.vcd")


# ===================================================================
# 3. GenericMux  (inputs[sel] → o, latency=1)
# ===================================================================
class TestGenericMux:
    """Parameterised N-to-1 multiplexer."""

    def test_sel_0(self):
        """sel=0 → input[0]."""
        dut = GenericMux(width=8, n_inputs=4)

        async def bench(ctx):
            ctx.set(dut.inputs[0], 10)
            ctx.set(dut.inputs[1], 20)
            ctx.set(dut.inputs[2], 30)
            ctx.set(dut.inputs[3], 40)
            ctx.set(dut.sel, 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 10, f"Expected 10, got {o}"

        _run(dut, bench, "test_generic_mux_sel0.vcd")

    def test_sel_1(self):
        """sel=1 → input[1]."""
        dut = GenericMux(width=8, n_inputs=4)

        async def bench(ctx):
            ctx.set(dut.inputs[0], 10)
            ctx.set(dut.inputs[1], 20)
            ctx.set(dut.inputs[2], 30)
            ctx.set(dut.inputs[3], 40)
            ctx.set(dut.sel, 1)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 20, f"Expected 20, got {o}"

        _run(dut, bench, "test_generic_mux_sel1.vcd")

    def test_sel_3(self):
        """sel=3 → input[3]."""
        dut = GenericMux(width=8, n_inputs=4)

        async def bench(ctx):
            ctx.set(dut.inputs[0], 10)
            ctx.set(dut.inputs[1], 20)
            ctx.set(dut.inputs[2], 30)
            ctx.set(dut.inputs[3], 40)
            ctx.set(dut.sel, 3)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 40, f"Expected 40, got {o}"

        _run(dut, bench, "test_generic_mux_sel3.vcd")

    def test_2_input_mux(self):
        """2-input mux: sel=0→A, sel=1→B."""
        dut = GenericMux(width=8, n_inputs=2)

        async def bench(ctx):
            ctx.set(dut.inputs[0], 0xAA)
            ctx.set(dut.inputs[1], 0x55)
            ctx.set(dut.sel, 1)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0x55, f"Expected 0x55, got {o:#x}"

        _run(dut, bench, "test_generic_mux_2input.vcd")


# ===================================================================
# 4. RowAdder  (inputs[0..n-1] → o, latency=1)
# ===================================================================
class TestRowAdder:
    """Multi-operand row adder."""

    def test_3_input_add(self):
        """1 + 2 + 3 = 6."""
        dut = RowAdder(width=8, n_inputs=3)

        async def bench(ctx):
            ctx.set(dut.inputs[0], 1)
            ctx.set(dut.inputs[1], 2)
            ctx.set(dut.inputs[2], 3)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 6, f"Expected 6, got {o}"

        _run(dut, bench, "test_row_adder_3.vcd")

    def test_4_input_add(self):
        """10 + 20 + 30 + 40 = 100."""
        dut = RowAdder(width=8, n_inputs=4)

        async def bench(ctx):
            ctx.set(dut.inputs[0], 10)
            ctx.set(dut.inputs[1], 20)
            ctx.set(dut.inputs[2], 30)
            ctx.set(dut.inputs[3], 40)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 100, f"Expected 100, got {o}"

        _run(dut, bench, "test_row_adder_4.vcd")

    def test_all_zeros(self):
        """0 + 0 + 0 = 0."""
        dut = RowAdder(width=8, n_inputs=3)

        async def bench(ctx):
            ctx.set(dut.inputs[0], 0)
            ctx.set(dut.inputs[1], 0)
            ctx.set(dut.inputs[2], 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0, f"Expected 0, got {o}"

        _run(dut, bench, "test_row_adder_zeros.vcd")

    def test_max_values(self):
        """255 + 255 + 255 = 765."""
        dut = RowAdder(width=8, n_inputs=3)

        async def bench(ctx):
            ctx.set(dut.inputs[0], 255)
            ctx.set(dut.inputs[1], 255)
            ctx.set(dut.inputs[2], 255)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 765, f"Expected 765, got {o}"

        _run(dut, bench, "test_row_adder_max.vcd")


# ===================================================================
# 5. BooleanEquation  (inputs → output, latency=1)
# ===================================================================
class TestBooleanEquation:
    """Truth-table-defined Boolean function."""

    def test_xor3_truth_table(self):
        """3-input XOR: init=0b10010110 (0x96)."""
        # XOR truth table for 3 inputs: output=1 when odd number of 1s
        tt = 0b10010110  # bit i = XOR of bits of i
        dut = BooleanEquation(n_inputs=3, truth_table=tt)

        async def bench(ctx):
            for i in range(8):
                ctx.set(dut.inputs, i)
                for _ in range(dut.latency + 1):
                    await ctx.tick()
                out = ctx.get(dut.output)
                expected = (tt >> i) & 1
                assert out == expected, f"input={i:03b}: expected {expected}, got {out}"

        _run(dut, bench, "test_bool_eq_xor3.vcd")

    def test_and2_truth_table(self):
        """2-input AND: init=0b1000."""
        tt = 0b1000  # AND: only input=11 gives 1
        dut = BooleanEquation(n_inputs=2, truth_table=tt)

        async def bench(ctx):
            for i in range(4):
                ctx.set(dut.inputs, i)
                for _ in range(dut.latency + 1):
                    await ctx.tick()
                out = ctx.get(dut.output)
                expected = (tt >> i) & 1
                assert out == expected, f"input={i:02b}: expected {expected}, got {out}"

        _run(dut, bench, "test_bool_eq_and2.vcd")

    def test_or2_truth_table(self):
        """2-input OR: init=0b1110."""
        tt = 0b1110  # OR: input=00 gives 0, rest give 1
        dut = BooleanEquation(n_inputs=2, truth_table=tt)

        async def bench(ctx):
            for i in range(4):
                ctx.set(dut.inputs, i)
                for _ in range(dut.latency + 1):
                    await ctx.tick()
                out = ctx.get(dut.output)
                expected = (tt >> i) & 1
                assert out == expected, f"input={i:02b}: expected {expected}, got {out}"

        _run(dut, bench, "test_bool_eq_or2.vcd")

    def test_all_zero(self):
        """Constant 0 function."""
        dut = BooleanEquation(n_inputs=2, truth_table=0)

        async def bench(ctx):
            for i in range(4):
                ctx.set(dut.inputs, i)
                for _ in range(dut.latency + 1):
                    await ctx.tick()
                out = ctx.get(dut.output)
                assert out == 0, f"input={i}: expected 0, got {out}"

        _run(dut, bench, "test_bool_eq_zero.vcd")

    def test_all_one(self):
        """Constant 1 function."""
        dut = BooleanEquation(n_inputs=2, truth_table=0b1111)

        async def bench(ctx):
            for i in range(4):
                ctx.set(dut.inputs, i)
                for _ in range(dut.latency + 1):
                    await ctx.tick()
                out = ctx.get(dut.output)
                assert out == 1, f"input={i}: expected 1, got {out}"

        _run(dut, bench, "test_bool_eq_one.vcd")


# ===================================================================
# 6. XilinxMUXF7  (i0, i1, sel → o, latency=0)
# ===================================================================
class TestXilinxMUXF7:
    """Xilinx MUXF7 (2:1 mux in CLB)."""

    def test_sel_0(self):
        """sel=0 → i0."""
        dut = XilinxMUXF7()

        async def bench(ctx):
            ctx.set(dut.i0, 1)
            ctx.set(dut.i1, 0)
            ctx.set(dut.sel, 0)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 1, f"Expected 1, got {o}"

        _run_comb(dut, bench, "test_muxf7_sel0.vcd")

    def test_sel_1(self):
        """sel=1 → i1."""
        dut = XilinxMUXF7()

        async def bench(ctx):
            ctx.set(dut.i0, 0)
            ctx.set(dut.i1, 1)
            ctx.set(dut.sel, 1)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 1, f"Expected 1, got {o}"

        _run_comb(dut, bench, "test_muxf7_sel1.vcd")

    def test_both_zero(self):
        """Both inputs 0 → output 0."""
        dut = XilinxMUXF7()

        async def bench(ctx):
            ctx.set(dut.i0, 0)
            ctx.set(dut.i1, 0)
            ctx.set(dut.sel, 0)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0, f"Expected 0, got {o}"

        _run_comb(dut, bench, "test_muxf7_both0.vcd")


# ===================================================================
# 7. XilinxMUXF8  (i0, i1, sel → o, latency=0)
# ===================================================================
class TestXilinxMUXF8:
    """Xilinx MUXF8 (2:1 mux)."""

    def test_sel_0(self):
        """sel=0 → i0."""
        dut = XilinxMUXF8()

        async def bench(ctx):
            ctx.set(dut.i0, 1)
            ctx.set(dut.i1, 0)
            ctx.set(dut.sel, 0)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 1, f"Expected 1, got {o}"

        _run_comb(dut, bench, "test_muxf8_sel0.vcd")

    def test_sel_1(self):
        """sel=1 → i1."""
        dut = XilinxMUXF8()

        async def bench(ctx):
            ctx.set(dut.i0, 0)
            ctx.set(dut.i1, 1)
            ctx.set(dut.sel, 1)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 1, f"Expected 1, got {o}"

        _run_comb(dut, bench, "test_muxf8_sel1.vcd")


# ===================================================================
# 8. XilinxFDCE  (d, ce, clr → q, latency=1)
# ===================================================================
class TestXilinxFDCE:
    """Xilinx FDCE (D flip-flop with clock enable and clear)."""

    def test_load_with_ce(self):
        """d=1, ce=1 → q=1 after clock."""
        dut = XilinxFDCE()

        async def bench(ctx):
            ctx.set(dut.d, 1)
            ctx.set(dut.ce, 1)
            ctx.set(dut.clr, 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            q = ctx.get(dut.q)
            assert q == 1, f"Expected 1, got {q}"

        _run(dut, bench, "test_fdce_load.vcd")

    def test_hold_without_ce(self):
        """d=1, ce=0 → q stays 0."""
        dut = XilinxFDCE()

        async def bench(ctx):
            ctx.set(dut.d, 1)
            ctx.set(dut.ce, 0)
            ctx.set(dut.clr, 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            q = ctx.get(dut.q)
            assert q == 0, f"Expected 0, got {q}"

        _run(dut, bench, "test_fdce_hold.vcd")

    def test_clear(self):
        """Load 1, then clear → q=0."""
        dut = XilinxFDCE()

        async def bench(ctx):
            # Load 1
            ctx.set(dut.d, 1)
            ctx.set(dut.ce, 1)
            ctx.set(dut.clr, 0)
            await ctx.tick()
            await ctx.tick()
            # Now clear
            ctx.set(dut.clr, 1)
            ctx.set(dut.ce, 0)
            await ctx.tick()
            await ctx.tick()
            q = ctx.get(dut.q)
            assert q == 0, f"Expected 0 after clear, got {q}"

        _run(dut, bench, "test_fdce_clear.vcd")


# ===================================================================
# 9. XilinxFourToTwoCompressor  (x0..x3 → s, c, latency=0)
# ===================================================================
class TestXilinxFourToTwoCompressor:
    """Xilinx 4:2 compressor."""

    def test_all_ones(self):
        """x0=x1=x2=x3=1: s = 1^1^1^1=0, c = (1&1)|(1&1)=1."""
        dut = XilinxFourToTwoCompressor(width=1)

        async def bench(ctx):
            ctx.set(dut.x0, 1)
            ctx.set(dut.x1, 1)
            ctx.set(dut.x2, 1)
            ctx.set(dut.x3, 1)
            await ctx.tick()
            s = ctx.get(dut.s)
            c = ctx.get(dut.c)
            assert s == 0, f"Expected s=0, got {s}"
            assert c == 1, f"Expected c=1, got {c}"

        _run_comb(dut, bench, "test_42comp_all1.vcd")

    def test_one_hot(self):
        """x0=1, rest=0: s=1, c=0."""
        dut = XilinxFourToTwoCompressor(width=1)

        async def bench(ctx):
            ctx.set(dut.x0, 1)
            ctx.set(dut.x1, 0)
            ctx.set(dut.x2, 0)
            ctx.set(dut.x3, 0)
            await ctx.tick()
            s = ctx.get(dut.s)
            c = ctx.get(dut.c)
            assert s == 1, f"Expected s=1, got {s}"
            assert c == 0, f"Expected c=0, got {c}"

        _run_comb(dut, bench, "test_42comp_onehot.vcd")

    def test_two_inputs(self):
        """x0=1, x1=1, x2=0, x3=0: s=0, c=1."""
        dut = XilinxFourToTwoCompressor(width=1)

        async def bench(ctx):
            ctx.set(dut.x0, 1)
            ctx.set(dut.x1, 1)
            ctx.set(dut.x2, 0)
            ctx.set(dut.x3, 0)
            await ctx.tick()
            s = ctx.get(dut.s)
            c = ctx.get(dut.c)
            assert s == 0, f"Expected s=0, got {s}"
            assert c == 1, f"Expected c=1, got {c}"

        _run_comb(dut, bench, "test_42comp_two.vcd")

    def test_three_inputs(self):
        """x0=1, x1=1, x2=1, x3=0: s=1, c=1."""
        dut = XilinxFourToTwoCompressor(width=1)

        async def bench(ctx):
            ctx.set(dut.x0, 1)
            ctx.set(dut.x1, 1)
            ctx.set(dut.x2, 1)
            ctx.set(dut.x3, 0)
            await ctx.tick()
            s = ctx.get(dut.s)
            c = ctx.get(dut.c)
            assert s == 1, f"Expected s=1, got {s}"
            assert c == 1, f"Expected c=1, got {c}"

        _run_comb(dut, bench, "test_42comp_three.vcd")

    def test_multibit(self):
        """4-bit compressor: x0=5, x1=3, x2=6, x3=9."""
        dut = XilinxFourToTwoCompressor(width=4)

        async def bench(ctx):
            ctx.set(dut.x0, 5)   # 0101
            ctx.set(dut.x1, 3)   # 0011
            ctx.set(dut.x2, 6)   # 0110
            ctx.set(dut.x3, 9)   # 1001
            await ctx.tick()
            s = ctx.get(dut.s)
            c = ctx.get(dut.c)
            expected_s = 5 ^ 3 ^ 6 ^ 9
            expected_c = (5 & 3) | (6 & 9)
            assert s == expected_s, f"Expected s={expected_s}, got {s}"
            assert c == expected_c, f"Expected c={expected_c}, got {c}"

        _run_comb(dut, bench, "test_42comp_multibit.vcd")


# ===================================================================
# 10. XilinxTernaryAddSub  (a, b, c → o = a+b+c, latency=0)
# ===================================================================
class TestXilinxTernaryAddSub:
    """Xilinx ternary adder/subtractor."""

    def test_1_2_3(self):
        """1 + 2 + 3 = 6."""
        dut = XilinxTernaryAddSub(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 1)
            ctx.set(dut.b, 2)
            ctx.set(dut.c, 3)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 6, f"Expected 6, got {o}"

        _run_comb(dut, bench, "test_ternary_add_123.vcd")

    def test_all_zeros(self):
        """0 + 0 + 0 = 0."""
        dut = XilinxTernaryAddSub(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 0)
            ctx.set(dut.b, 0)
            ctx.set(dut.c, 0)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0, f"Expected 0, got {o}"

        _run_comb(dut, bench, "test_ternary_add_zeros.vcd")

    def test_max_values(self):
        """255 + 255 + 255 = 765."""
        dut = XilinxTernaryAddSub(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 255)
            ctx.set(dut.b, 255)
            ctx.set(dut.c, 255)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 765, f"Expected 765, got {o}"

        _run_comb(dut, bench, "test_ternary_add_max.vcd")

    def test_10_20_30(self):
        """10 + 20 + 30 = 60."""
        dut = XilinxTernaryAddSub(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 10)
            ctx.set(dut.b, 20)
            ctx.set(dut.c, 30)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 60, f"Expected 60, got {o}"

        _run_comb(dut, bench, "test_ternary_add_102030.vcd")


# ===================================================================
# 11. XilinxN2MDecoder  (sel → o = 1<<sel, latency=0)
# ===================================================================
class TestXilinxN2MDecoder:
    """Xilinx n-to-m decoder."""

    def test_sel_0(self):
        """sel=0 → o=0b00000001."""
        dut = XilinxN2MDecoder(n=3)

        async def bench(ctx):
            ctx.set(dut.sel, 0)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 1, f"Expected 1, got {o}"

        _run_comb(dut, bench, "test_n2m_sel0.vcd")

    def test_sel_2(self):
        """sel=2 → o=0b00000100."""
        dut = XilinxN2MDecoder(n=3)

        async def bench(ctx):
            ctx.set(dut.sel, 2)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 4, f"Expected 4, got {o}"

        _run_comb(dut, bench, "test_n2m_sel2.vcd")

    def test_sel_7(self):
        """sel=7 → o=0b10000000."""
        dut = XilinxN2MDecoder(n=3)

        async def bench(ctx):
            ctx.set(dut.sel, 7)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 128, f"Expected 128, got {o}"

        _run_comb(dut, bench, "test_n2m_sel7.vcd")

    def test_all_values(self):
        """Verify all 8 outputs for 3-bit decoder."""
        dut = XilinxN2MDecoder(n=3)

        async def bench(ctx):
            for i in range(8):
                ctx.set(dut.sel, i)
                await ctx.tick()
                o = ctx.get(dut.o)
                expected = 1 << i
                assert o == expected, f"sel={i}: expected {expected}, got {o}"

        _run_comb(dut, bench, "test_n2m_all.vcd")


# ===================================================================
# 12. XilinxGenericMux  (inputs, sel → o, latency=0)
# ===================================================================
class TestXilinxGenericMux:
    """Xilinx generic multiplexer."""

    def test_sel_0(self):
        """sel=0 → first 8 bits of inputs."""
        dut = XilinxGenericMux(width=8, sel_bits=2)

        async def bench(ctx):
            # Pack 4 values: [0xAA, 0xBB, 0xCC, 0xDD]
            packed = 0xAA | (0xBB << 8) | (0xCC << 16) | (0xDD << 24)
            ctx.set(dut.inputs, packed)
            ctx.set(dut.sel, 0)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0xAA, f"Expected 0xAA, got {o:#x}"

        _run_comb(dut, bench, "test_xmux_sel0.vcd")

    def test_sel_2(self):
        """sel=2 → third 8-bit slice."""
        dut = XilinxGenericMux(width=8, sel_bits=2)

        async def bench(ctx):
            packed = 0xAA | (0xBB << 8) | (0xCC << 16) | (0xDD << 24)
            ctx.set(dut.inputs, packed)
            ctx.set(dut.sel, 2)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0xCC, f"Expected 0xCC, got {o:#x}"

        _run_comb(dut, bench, "test_xmux_sel2.vcd")

    def test_sel_3(self):
        """sel=3 → fourth 8-bit slice."""
        dut = XilinxGenericMux(width=8, sel_bits=2)

        async def bench(ctx):
            packed = 0xAA | (0xBB << 8) | (0xCC << 16) | (0xDD << 24)
            ctx.set(dut.inputs, packed)
            ctx.set(dut.sel, 3)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0xDD, f"Expected 0xDD, got {o:#x}"

        _run_comb(dut, bench, "test_xmux_sel3.vcd")


# ===================================================================
# 13. XilinxLUT5  (i → o = (init >> i) & 1, latency=0)
# ===================================================================
class TestXilinxLUT5:
    """Xilinx 5-input LUT."""

    def test_and5(self):
        """5-input AND: init = 1<<31 = 0x80000000."""
        init = 1 << 31  # only input=11111 gives 1
        dut = XilinxLUT5(init=init)

        async def bench(ctx):
            # All ones → 1
            ctx.set(dut.i, 0b11111)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 1, f"Expected 1 for all-ones, got {o}"
            # All zeros → 0
            ctx.set(dut.i, 0b00000)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0, f"Expected 0 for all-zeros, got {o}"

        _run_comb(dut, bench, "test_lut5_and.vcd")

    def test_identity_bit0(self):
        """LUT that outputs bit 0 of input: init = 0xAAAAAAAA."""
        init = 0xAAAAAAAA  # bit i of init = bit 0 of i
        dut = XilinxLUT5(init=init)

        async def bench(ctx):
            for i in range(32):
                ctx.set(dut.i, i)
                await ctx.tick()
                o = ctx.get(dut.o)
                expected = (init >> i) & 1
                assert o == expected, f"i={i}: expected {expected}, got {o}"

        _run_comb(dut, bench, "test_lut5_id_bit0.vcd")

    def test_all_zero_init(self):
        """init=0 → always output 0."""
        dut = XilinxLUT5(init=0)

        async def bench(ctx):
            for i in [0, 15, 31]:
                ctx.set(dut.i, i)
                await ctx.tick()
                o = ctx.get(dut.o)
                assert o == 0, f"i={i}: expected 0, got {o}"

        _run_comb(dut, bench, "test_lut5_zero.vcd")

    def test_all_one_init(self):
        """init=0xFFFFFFFF → always output 1."""
        dut = XilinxLUT5(init=0xFFFFFFFF)

        async def bench(ctx):
            for i in [0, 15, 31]:
                ctx.set(dut.i, i)
                await ctx.tick()
                o = ctx.get(dut.o)
                assert o == 1, f"i={i}: expected 1, got {o}"

        _run_comb(dut, bench, "test_lut5_ones.vcd")


# ===================================================================
# 14. XilinxLUT6  (i → o = (init >> i) & 1, latency=0)
# ===================================================================
class TestXilinxLUT6:
    """Xilinx 6-input LUT."""

    def test_and6(self):
        """6-input AND: init = 1<<63."""
        init = 1 << 63
        dut = XilinxLUT6(init=init)

        async def bench(ctx):
            ctx.set(dut.i, 0b111111)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 1, f"Expected 1 for all-ones, got {o}"
            ctx.set(dut.i, 0b000000)
            await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0, f"Expected 0 for all-zeros, got {o}"

        _run_comb(dut, bench, "test_lut6_and.vcd")

    def test_xor_pair(self):
        """XOR of bits 0 and 1: init where bit i = (i&1)^((i>>1)&1)."""
        init = 0
        for i in range(64):
            if ((i & 1) ^ ((i >> 1) & 1)):
                init |= (1 << i)
        dut = XilinxLUT6(init=init)

        async def bench(ctx):
            for i in range(64):
                ctx.set(dut.i, i)
                await ctx.tick()
                o = ctx.get(dut.o)
                expected = (init >> i) & 1
                assert o == expected, f"i={i}: expected {expected}, got {o}"

        _run_comb(dut, bench, "test_lut6_xor.vcd")

    def test_all_zero_init(self):
        """init=0 → always output 0."""
        dut = XilinxLUT6(init=0)

        async def bench(ctx):
            for i in [0, 32, 63]:
                ctx.set(dut.i, i)
                await ctx.tick()
                o = ctx.get(dut.o)
                assert o == 0, f"i={i}: expected 0, got {o}"

        _run_comb(dut, bench, "test_lut6_zero.vcd")


# ===================================================================
# 15. IntelTernaryAdder  (a, b, c → o = a+b+c, latency=1)
# ===================================================================
class TestIntelTernaryAdder:
    """Intel ternary adder."""

    def test_1_2_3(self):
        """1 + 2 + 3 = 6."""
        dut = IntelTernaryAdder(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 1)
            ctx.set(dut.b, 2)
            ctx.set(dut.c, 3)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 6, f"Expected 6, got {o}"

        _run(dut, bench, "test_intel_ternary_123.vcd")

    def test_all_zeros(self):
        """0 + 0 + 0 = 0."""
        dut = IntelTernaryAdder(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 0)
            ctx.set(dut.b, 0)
            ctx.set(dut.c, 0)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 0, f"Expected 0, got {o}"

        _run(dut, bench, "test_intel_ternary_zeros.vcd")

    def test_max_values(self):
        """255 + 255 + 255 = 765."""
        dut = IntelTernaryAdder(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 255)
            ctx.set(dut.b, 255)
            ctx.set(dut.c, 255)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 765, f"Expected 765, got {o}"

        _run(dut, bench, "test_intel_ternary_max.vcd")

    def test_10_20_30(self):
        """10 + 20 + 30 = 60."""
        dut = IntelTernaryAdder(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 10)
            ctx.set(dut.b, 20)
            ctx.set(dut.c, 30)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 60, f"Expected 60, got {o}"

        _run(dut, bench, "test_intel_ternary_102030.vcd")

    def test_power_of_two(self):
        """128 + 64 + 32 = 224."""
        dut = IntelTernaryAdder(width=8)

        async def bench(ctx):
            ctx.set(dut.a, 128)
            ctx.set(dut.b, 64)
            ctx.set(dut.c, 32)
            for _ in range(dut.latency + 1):
                await ctx.tick()
            o = ctx.get(dut.o)
            assert o == 224, f"Expected 224, got {o}"

        _run(dut, bench, "test_intel_ternary_pow2.vcd")
