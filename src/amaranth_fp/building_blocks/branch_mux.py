"""Branch mux for equalizing latency between pipelined branches."""
from __future__ import annotations

from amaranth.hdl import *

from ..pipelined import PipelinedComponent

__all__ = ["BranchMux", "MultiBranchMux"]


class BranchMux(PipelinedComponent):
    """Mux between two pipelined branches with different latencies.

    Equalizes latency by padding the shorter branch with delay registers.
    The condition signal is also delayed to arrive at the mux simultaneously.

    Parameters
    ----------
    width : int
        Bit width of data signals.
    latency_a : int
        Latency of branch A (selected when ``cond=0``).
    latency_b : int
        Latency of branch B (selected when ``cond=1``).

    Ports
    -----
    cond : Signal, in
        1-bit condition (0 → select branch_a, 1 → select branch_b).
    branch_a : Signal(width), in
        Data from branch A.
    branch_b : Signal(width), in
        Data from branch B.
    o : Signal(width), out
        Output (registered).

    Latency
    -------
    ``max(latency_a, latency_b) + 1`` (the +1 is for the final mux register).
    """

    def __init__(self, width: int, latency_a: int, latency_b: int):
        super().__init__()
        self.width = width
        self.latency_a = latency_a
        self.latency_b = latency_b
        self.latency = max(latency_a, latency_b) + 1

        self.cond = Signal(name="cond")
        self.branch_a = Signal(width, name="branch_a")
        self.branch_b = Signal(width, name="branch_b")
        self.o = Signal(width, name="o")

    def elaborate(self, platform):
        m = Module()
        max_lat = max(self.latency_a, self.latency_b)

        # Delay branch_a by (max_lat - latency_a) cycles
        a_delayed = self.branch_a
        for i in range(max_lat - self.latency_a):
            d = Signal(self.width, name=f"a_delay_{i}")
            m.d.sync += d.eq(a_delayed)
            a_delayed = d

        # Delay branch_b by (max_lat - latency_b) cycles
        b_delayed = self.branch_b
        for i in range(max_lat - self.latency_b):
            d = Signal(self.width, name=f"b_delay_{i}")
            m.d.sync += d.eq(b_delayed)
            b_delayed = d

        # Delay condition by max_lat cycles (it was computed at cycle 0)
        cond_delayed = self.cond
        for i in range(max_lat):
            d = Signal(name=f"cond_delay_{i}")
            m.d.sync += d.eq(cond_delayed)
            cond_delayed = d

        # Final mux + register
        mux_result = Signal(self.width, name="mux_result")
        m.d.comb += mux_result.eq(Mux(cond_delayed, b_delayed, a_delayed))
        m.d.sync += self.o.eq(mux_result)

        return m


class MultiBranchMux(PipelinedComponent):
    """Mux between N pipelined branches with different latencies.

    Equalizes latency by padding shorter branches with delay registers.
    The selector signal is also delayed to arrive at the mux simultaneously.

    Parameters
    ----------
    width : int
        Bit width of data signals.
    n_branches : int
        Number of branches.
    latencies : list[int]
        Latency for each branch.

    Ports
    -----
    selector : Signal(range(n_branches)), in
        Branch selector index.
    branches : list[Signal(width)], in
        Data from each branch.
    o : Signal(width), out
        Output (registered).

    Latency
    -------
    ``max(latencies) + 1``.
    """

    def __init__(self, width: int, n_branches: int, latencies: list[int]):
        super().__init__()
        if len(latencies) != n_branches:
            raise ValueError("len(latencies) must equal n_branches")
        self.width = width
        self.n_branches = n_branches
        self.latencies = list(latencies)
        self.latency = max(latencies) + 1

        self.selector = Signal(range(n_branches), name="selector")
        self.branches = [
            Signal(width, name=f"branch_{i}") for i in range(n_branches)
        ]
        self.o = Signal(width, name="o")

    def elaborate(self, platform):
        m = Module()
        max_lat = max(self.latencies)

        # Delay each branch to align
        delayed = []
        for idx, (branch, lat) in enumerate(zip(self.branches, self.latencies)):
            sig = branch
            for i in range(max_lat - lat):
                d = Signal(self.width, name=f"br{idx}_delay_{i}")
                m.d.sync += d.eq(sig)
                sig = d
            delayed.append(sig)

        # Delay selector by max_lat cycles
        sel_delayed = self.selector
        for i in range(max_lat):
            d = Signal(self.selector.shape(), name=f"sel_delay_{i}")
            m.d.sync += d.eq(sel_delayed)
            sel_delayed = d

        # Priority mux via Switch
        mux_result = Signal(self.width, name="mux_result")
        with m.Switch(sel_delayed):
            for idx, sig in enumerate(delayed):
                with m.Case(idx):
                    m.d.comb += mux_result.eq(sig)

        m.d.sync += self.o.eq(mux_result)

        return m
