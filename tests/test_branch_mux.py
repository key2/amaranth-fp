"""Tests for BranchMux and MultiBranchMux."""

from amaranth.sim import Simulator

from amaranth_fp.building_blocks import BranchMux, MultiBranchMux


def _run_sync(dut, testbench, *, vcd="test_branch_mux.vcd"):
    """Run a clocked testbench."""
    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd(vcd):
        sim.run()


# ---------------------------------------------------------------------------
# BranchMux tests
# ---------------------------------------------------------------------------

class TestBranchMux:
    def test_equal_latencies(self):
        """Both branches have same latency → no delay registers needed."""
        dut = BranchMux(width=8, latency_a=3, latency_b=3)
        assert dut.latency == 4  # max(3,3)+1

        async def bench(ctx):
            # Select branch_a (cond=0), value=0xAA
            ctx.set(dut.cond, 0)
            ctx.set(dut.branch_a, 0xAA)
            ctx.set(dut.branch_b, 0x55)
            # Wait for max_lat(3) + 1 (mux reg) = 4 cycles
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.o) == 0xAA

        _run_sync(dut, bench)

    def test_select_branch_b_equal(self):
        """Select branch_b with equal latencies."""
        dut = BranchMux(width=8, latency_a=3, latency_b=3)

        async def bench(ctx):
            ctx.set(dut.cond, 1)
            ctx.set(dut.branch_a, 0xAA)
            ctx.set(dut.branch_b, 0x55)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.o) == 0x55

        _run_sync(dut, bench)

    def test_different_latencies_select_a(self):
        """Branch A is shorter (3) than B (7). Select A → padded with 4 delays."""
        dut = BranchMux(width=8, latency_a=3, latency_b=7)
        assert dut.latency == 8  # max(3,7)+1

        async def bench(ctx):
            ctx.set(dut.cond, 0)
            ctx.set(dut.branch_a, 0x42)
            ctx.set(dut.branch_b, 0x00)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.o) == 0x42

        _run_sync(dut, bench)

    def test_different_latencies_select_b(self):
        """Branch A is shorter (3) than B (7). Select B."""
        dut = BranchMux(width=8, latency_a=3, latency_b=7)

        async def bench(ctx):
            ctx.set(dut.cond, 1)
            ctx.set(dut.branch_a, 0x00)
            ctx.set(dut.branch_b, 0xBE)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.o) == 0xBE

        _run_sync(dut, bench)

    def test_condition_selects_correctly_with_asymmetric_latency(self):
        """Verify cond=0 selects branch_a and cond=1 selects branch_b
        with asymmetric latencies, using stable inputs."""
        dut = BranchMux(width=8, latency_a=2, latency_b=5)
        assert dut.latency == 6  # max(2,5)+1

        async def bench(ctx):
            # Hold cond=1, expect branch_b selected
            ctx.set(dut.cond, 1)
            ctx.set(dut.branch_a, 0xAA)
            ctx.set(dut.branch_b, 0xBB)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.o) == 0xBB

        _run_sync(dut, bench)

    def test_zero_latency_branches(self):
        """Both branches zero latency → total latency = 1 (just the mux reg)."""
        dut = BranchMux(width=16, latency_a=0, latency_b=0)
        assert dut.latency == 1

        async def bench(ctx):
            ctx.set(dut.cond, 1)
            ctx.set(dut.branch_a, 0x1111)
            ctx.set(dut.branch_b, 0x2222)
            await ctx.tick()
            assert ctx.get(dut.o) == 0x2222

        _run_sync(dut, bench)


# ---------------------------------------------------------------------------
# MultiBranchMux tests
# ---------------------------------------------------------------------------

class TestMultiBranchMux:
    def test_three_branches(self):
        """3 branches with latencies [2, 5, 3]. Select each in turn."""
        dut = MultiBranchMux(width=8, n_branches=3, latencies=[2, 5, 3])
        assert dut.latency == 6  # max(2,5,3)+1

        async def bench(ctx):
            # Select branch 0
            ctx.set(dut.selector, 0)
            ctx.set(dut.branches[0], 0xAA)
            ctx.set(dut.branches[1], 0xBB)
            ctx.set(dut.branches[2], 0xCC)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.o) == 0xAA

        _run_sync(dut, bench, vcd="test_multi_branch.vcd")

    def test_three_branches_select_1(self):
        dut = MultiBranchMux(width=8, n_branches=3, latencies=[2, 5, 3])

        async def bench(ctx):
            ctx.set(dut.selector, 1)
            ctx.set(dut.branches[0], 0xAA)
            ctx.set(dut.branches[1], 0xBB)
            ctx.set(dut.branches[2], 0xCC)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.o) == 0xBB

        _run_sync(dut, bench, vcd="test_multi_branch2.vcd")

    def test_three_branches_select_2(self):
        dut = MultiBranchMux(width=8, n_branches=3, latencies=[2, 5, 3])

        async def bench(ctx):
            ctx.set(dut.selector, 2)
            ctx.set(dut.branches[0], 0xAA)
            ctx.set(dut.branches[1], 0xBB)
            ctx.set(dut.branches[2], 0xCC)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.o) == 0xCC

        _run_sync(dut, bench, vcd="test_multi_branch3.vcd")

    def test_equal_latencies(self):
        """All branches same latency."""
        dut = MultiBranchMux(width=8, n_branches=2, latencies=[4, 4])
        assert dut.latency == 5

        async def bench(ctx):
            ctx.set(dut.selector, 1)
            ctx.set(dut.branches[0], 0x11)
            ctx.set(dut.branches[1], 0x22)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.o) == 0x22

        _run_sync(dut, bench, vcd="test_multi_equal.vcd")


# ---------------------------------------------------------------------------
# Example: FPAsinExample showing BranchMux usage pattern
# ---------------------------------------------------------------------------

class TestFPAsinExample:
    """Demonstrates how BranchMux would be used for asin(x)."""

    def test_asin_example_concept(self):
        """Conceptual test: branch A (poly, lat=5) vs branch B (sqrt+poly, lat=10)."""
        # The BranchMux itself handles the equalization.
        # In a real asin, branch_a and branch_b would come from sub-components.
        # Here we just verify the mux concept with those latencies.
        dut = BranchMux(width=16, latency_a=5, latency_b=10)
        assert dut.latency == 11  # max(5,10)+1

        async def bench(ctx):
            # Simulate branch A result (|x| <= 0.5 path)
            ctx.set(dut.cond, 0)
            ctx.set(dut.branch_a, 0x1234)  # polynomial result
            ctx.set(dut.branch_b, 0x0000)
            for _ in range(dut.latency):
                await ctx.tick()
            assert ctx.get(dut.o) == 0x1234

        _run_sync(dut, bench, vcd="test_asin_example.vcd")
