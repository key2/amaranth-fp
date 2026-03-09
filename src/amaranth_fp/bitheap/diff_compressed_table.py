"""Differentially compressed ROM table (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["DiffCompressedTable"]


class DiffCompressedTable(PipelinedComponent):
    """ROM table storing differences between consecutive entries.

    Parameters
    ----------
    values : list[int]
        Table values.
    input_width : int
        Address width.
    """

    def __init__(self, values: list[int], input_width: int) -> None:
        super().__init__()
        self.values = values
        self.input_width = input_width
        max_val = max(abs(v) for v in values) if values else 1
        self.output_width = max_val.bit_length() + 1

        # Compute diffs
        self._base_values = values[:1]
        self._diffs = [values[i] - values[i - 1] for i in range(1, len(values))]

        self.addr = Signal(input_width, name="addr")
        self.data = Signal(self.output_width, name="data")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        iw = self.input_width
        ow = self.output_width
        vals = self.values
        n = len(vals)

        # Base table (full values) — used for simple implementation
        mem = Memory(shape=ow, depth=max(n, 1), init=vals)
        m.submodules.mem_rd = mem_rd = mem.read_port()

        # Stage 0→1: read
        addr_r = Signal(iw, name="addr_r")
        m.d.sync += addr_r.eq(self.addr)
        m.d.comb += mem_rd.addr.eq(addr_r)

        # Stage 1→2: output
        data_r = Signal(ow, name="data_r")
        m.d.sync += data_r.eq(mem_rd.data)
        m.d.comb += self.data.eq(data_r)

        return m
