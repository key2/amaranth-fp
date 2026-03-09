"""Bitonic sorting network (pipelined)."""
from __future__ import annotations

import math

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["SortingNetwork"]


class SortingNetwork(PipelinedComponent):
    """Bitonic sorting network.

    Parameters
    ----------
    width : int
        Element bit width.
    n_elements : int
        Number of elements to sort (rounded up to power of 2).
    """

    def __init__(self, width: int, n_elements: int) -> None:
        super().__init__()
        self.width = width
        # Round up to power of 2
        self.n_elements = n_elements
        n = 1
        while n < n_elements:
            n <<= 1
        self._n = n
        log_n = int(math.log2(n)) if n > 1 else 1
        self.inputs = [Signal(width, name=f"sort_in_{i}") for i in range(n_elements)]
        self.outputs = [Signal(width, name=f"sort_out_{i}") for i in range(n_elements)]
        self.latency = log_n * (log_n + 1) // 2

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        n = self._n
        n_elem = self.n_elements

        # Pad inputs to power of 2 with max values
        cur = []
        for i in range(n):
            s = Signal(w, name=f"sn_cur_{i}")
            if i < n_elem:
                m.d.comb += s.eq(self.inputs[i])
            else:
                m.d.comb += s.eq((1 << w) - 1)  # max value
            cur.append(s)

        if n <= 1:
            if n_elem > 0:
                m.d.comb += self.outputs[0].eq(cur[0])
            return m

        log_n = int(math.log2(n))

        # Bitonic sort network
        for stage in range(log_n):
            for sub_stage in range(stage + 1):
                nxt = []
                half = 1 << (stage - sub_stage)
                for i in range(n):
                    nxt.append(Signal(w, name=f"sn_s{stage}_{sub_stage}_{i}"))

                for block_start in range(0, n, 2 * half):
                    ascending = ((block_start >> (stage + 1)) & 1) == 0
                    for i in range(half):
                        idx_lo = block_start + i
                        idx_hi = block_start + i + half
                        if idx_lo < n and idx_hi < n:
                            if ascending:
                                with m.If(cur[idx_lo] > cur[idx_hi]):
                                    m.d.comb += [nxt[idx_lo].eq(cur[idx_hi]), nxt[idx_hi].eq(cur[idx_lo])]
                                with m.Else():
                                    m.d.comb += [nxt[idx_lo].eq(cur[idx_lo]), nxt[idx_hi].eq(cur[idx_hi])]
                            else:
                                with m.If(cur[idx_lo] < cur[idx_hi]):
                                    m.d.comb += [nxt[idx_lo].eq(cur[idx_hi]), nxt[idx_hi].eq(cur[idx_lo])]
                                with m.Else():
                                    m.d.comb += [nxt[idx_lo].eq(cur[idx_lo]), nxt[idx_hi].eq(cur[idx_hi])]

                # Pipeline register after each sub-stage
                reg = []
                for i in range(n):
                    r = Signal(w, name=f"sn_r{stage}_{sub_stage}_{i}")
                    m.d.sync += r.eq(nxt[i])
                    reg.append(r)
                cur = reg

        for i in range(n_elem):
            m.d.comb += self.outputs[i].eq(cur[i])

        return m
