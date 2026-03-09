"""Tests for batch 2 operators."""
from amaranth import *
from amaranth.sim import Simulator

from amaranth_fp.complex.fix_complex_adder import FixComplexAdder
from amaranth_fp.complex.fix_complex_mult import FixComplexMult
from amaranth_fp.bitheap.bit_heap import BitHeap
from amaranth_fp.bitheap.compressor import Compressor
from amaranth_fp.operators.lns_div import LNSDiv
from amaranth_fp.operators.lns_sqrt import LNSSqrt
from amaranth_fp.operators.sorting_network import SortingNetwork
from amaranth_fp.operators.fix_sum_of_squares import FixSumOfSquares
from amaranth_fp.operators.int_dual_add_sub import IntDualAddSub


def _run_sim(dut, process, *, clocks=20):
    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(process)
    with sim.write_vcd("test_batch2.vcd"):
        sim.run()


async def _tick(ctx, n=1):
    for _ in range(n):
        await ctx.tick()


def test_fix_complex_adder():
    """(1+2i) + (3+4i) = (4+6i)."""
    dut = FixComplexAdder(16)

    async def proc(ctx):
        ctx.set(dut.a_re, 1)
        ctx.set(dut.a_im, 2)
        ctx.set(dut.b_re, 3)
        ctx.set(dut.b_im, 4)
        await _tick(ctx, dut.latency + 1)
        assert ctx.get(dut.o_re) == 4, f"re={ctx.get(dut.o_re)}"
        assert ctx.get(dut.o_im) == 6, f"im={ctx.get(dut.o_im)}"

    _run_sim(dut, proc)


def test_fix_complex_mult():
    """(1+2i)*(3+4i) = (1*3-2*4) + (1*4+2*3)i = -5 + 10i."""
    dut = FixComplexMult(16)

    async def proc(ctx):
        ctx.set(dut.a_re, 1)
        ctx.set(dut.a_im, 2)
        ctx.set(dut.b_re, 3)
        ctx.set(dut.b_im, 4)
        await _tick(ctx, dut.latency + 1)
        o_re = ctx.get(dut.o_re)
        o_im = ctx.get(dut.o_im)
        # Signed interpretation
        mask = (1 << 33) - 1
        re_val = o_re if o_re < (1 << 32) else o_re - (1 << 33)
        im_val = o_im if o_im < (1 << 32) else o_im - (1 << 33)
        assert re_val == -5, f"re={re_val}"
        assert im_val == 10, f"im={im_val}"

    _run_sim(dut, proc)


def test_bitheap():
    """Add bits and check output."""
    bh = BitHeap(max_weight=4, width=8)
    # Add bit at weight 0 and weight 2
    s0 = Signal(name="b0")
    s1 = Signal(name="b2")
    bh.add_bit(0, s0)
    bh.add_bit(2, s1)

    async def proc(ctx):
        ctx.set(s0, 1)
        ctx.set(s1, 1)
        await _tick(ctx, bh.latency + 1)
        assert ctx.get(bh.output) == 5, f"got {ctx.get(bh.output)}"  # 1 + 4 = 5

    _run_sim(bh, proc)


def test_compressor_3_2():
    """(3,2) full adder: 3 bits at weight 0 → 2-bit output."""
    comp = Compressor(input_counts=[3], output_width=2)

    async def proc(ctx):
        ctx.set(comp.inputs[0], 1)
        ctx.set(comp.inputs[1], 1)
        ctx.set(comp.inputs[2], 1)
        await _tick(ctx, comp.latency + 1)
        assert ctx.get(comp.output) == 3, f"got {ctx.get(comp.output)}"  # 1+1+1=3

    _run_sim(comp, proc)


def test_lns_div():
    """LNS division: subtract log values."""
    dut = LNSDiv(8)

    async def proc(ctx):
        # log=10, log=3, result should be 7
        ctx.set(dut.a, 10)  # positive, log=10
        ctx.set(dut.b, 3)   # positive, log=3
        await _tick(ctx, dut.latency + 1)
        result = ctx.get(dut.o)
        assert (result & 0x7F) == 7, f"got {result & 0x7F}"

    _run_sim(dut, proc)


def test_lns_sqrt():
    """LNS sqrt: halve log value."""
    dut = LNSSqrt(8)

    async def proc(ctx):
        ctx.set(dut.a, 20)  # positive, log=20
        await _tick(ctx, dut.latency + 1)
        result = ctx.get(dut.o)
        assert (result & 0x7F) == 10, f"got {result & 0x7F}"

    _run_sim(dut, proc)


def test_sorting_network():
    """Sort 4 elements."""
    dut = SortingNetwork(width=8, n_elements=4)

    async def proc(ctx):
        ctx.set(dut.inputs[0], 30)
        ctx.set(dut.inputs[1], 10)
        ctx.set(dut.inputs[2], 40)
        ctx.set(dut.inputs[3], 20)
        await _tick(ctx, dut.latency + 2)
        vals = [ctx.get(dut.outputs[i]) for i in range(4)]
        assert vals == [10, 20, 30, 40], f"got {vals}"

    _run_sim(dut, proc)


def test_fix_sum_of_squares():
    """3^2 + 4^2 = 25."""
    dut = FixSumOfSquares(width=8, n_inputs=2)

    async def proc(ctx):
        ctx.set(dut.inputs[0], 3)
        ctx.set(dut.inputs[1], 4)
        await _tick(ctx, dut.latency + 1)
        assert ctx.get(dut.output) == 25, f"got {ctx.get(dut.output)}"

    _run_sim(dut, proc)


def test_int_dual_add_sub():
    """5+3=8, 5-3=2."""
    dut = IntDualAddSub(8)

    async def proc(ctx):
        ctx.set(dut.a, 5)
        ctx.set(dut.b, 3)
        await _tick(ctx, dut.latency + 1)
        assert ctx.get(dut.sum_out) == 8, f"sum={ctx.get(dut.sum_out)}"
        assert ctx.get(dut.diff_out) == 2, f"diff={ctx.get(dut.diff_out)}"

    _run_sim(dut, proc)
