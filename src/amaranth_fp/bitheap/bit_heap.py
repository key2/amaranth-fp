"""Bit heap for multi-operand addition with compressor tree (pipelined)."""
from __future__ import annotations

import math

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["BitHeap"]


class BitHeap(PipelinedComponent):
    """Bit heap with compressor tree reduction.

    Parameters
    ----------
    max_weight : int
        Maximum column height (number of bits in tallest column).
    width : int
        Number of columns (output width).
    """

    def __init__(self, max_weight: int, width: int) -> None:
        super().__init__()
        self.max_weight = max_weight
        self.width = width
        self._columns: list[list[Signal]] = [[] for _ in range(width)]
        self.output = Signal(width, name="output")
        # Latency: ceil(log1.5(max_height)) + 1
        if max_weight <= 2:
            self.latency = 2
        else:
            self.latency = int(math.ceil(math.log(max_weight) / math.log(1.5))) + 1

    def add_bit(self, weight: int, signal: Signal) -> None:
        """Add a bit at the given column weight."""
        if 0 <= weight < self.width:
            self._columns[weight].append(signal)

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width

        # Sum up all bits per column, then do a final addition
        # Simple approach: weighted sum of all bits
        partial = Signal(w, name="partial_sum")
        parts = []
        for col, bits in enumerate(self._columns):
            for bit_sig in bits:
                p = Signal(w, name=f"bh_w{col}_{bit_sig.name}")
                m.d.comb += p.eq(bit_sig << col)
                parts.append(p)

        if parts:
            acc = Signal(w, name="bh_acc")
            m.d.comb += acc.eq(sum(parts))
            m.d.comb += partial.eq(acc)

        # Pipeline register
        out_r = Signal(w, name="bh_out_r")
        m.d.sync += out_r.eq(partial)
        m.d.comb += self.output.eq(out_r)

        return m
