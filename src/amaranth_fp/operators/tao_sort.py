"""Tao sorting network (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["TaoSort"]


class TaoSort(PipelinedComponent):
    """Tao's sorting network — an area-efficient sorting network variant.

    Parameters
    ----------
    width : int
        Element width.
    n_inputs : int
        Number of elements to sort.
    """

    def __init__(self, width: int, n_inputs: int = 4) -> None:
        super().__init__()
        self.width = width
        self.n_inputs = n_inputs
        self.inputs = [Signal(width, name=f"in{i}") for i in range(n_inputs)]
        self.outputs = [Signal(width, name=f"out{i}") for i in range(n_inputs)]
        self.latency = (n_inputs + 1) // 2

    def _compare_swap(self, m, a, b, ra, rb):
        """Generate a compare-and-swap unit."""
        with m.If(a > b):
            m.d.sync += [ra.eq(b), rb.eq(a)]
        with m.Else():
            m.d.sync += [ra.eq(a), rb.eq(b)]

    def elaborate(self, platform) -> Module:
        m = Module()
        n = self.n_inputs
        w = self.width

        if n <= 1:
            m.d.comb += self.outputs[0].eq(self.inputs[0])
            return m

        # Odd-even transposition sort (simple network)
        regs = [Signal(w, name=f"r0_{i}") for i in range(n)]
        for i in range(n):
            m.d.comb += regs[i].eq(self.inputs[i])

        for stage in range(self.latency):
            next_regs = [Signal(w, name=f"r{stage+1}_{i}") for i in range(n)]
            start = stage % 2
            for i in range(start, n - 1, 2):
                self._compare_swap(m, regs[i], regs[i + 1],
                                   next_regs[i], next_regs[i + 1])
            # Pass through unpaired elements
            if start == 1:
                m.d.sync += next_regs[0].eq(regs[0])
            if (n - start) % 2 == 1:
                m.d.sync += next_regs[n - 1].eq(regs[n - 1])
            regs = next_regs

        for i in range(n):
            m.d.comb += self.outputs[i].eq(regs[i])
        return m
