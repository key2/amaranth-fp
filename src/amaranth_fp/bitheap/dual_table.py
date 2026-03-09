"""Dual-output ROM table (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["DualTable"]


class DualTable(PipelinedComponent):
    """Single-address, dual-output ROM table.

    Parameters
    ----------
    values_a : list[int]
    values_b : list[int]
    input_width : int
    output_width_a : int
    output_width_b : int
    """

    def __init__(
        self,
        values_a: list[int],
        values_b: list[int],
        input_width: int,
        output_width_a: int,
        output_width_b: int,
    ) -> None:
        super().__init__()
        self.values_a = values_a
        self.values_b = values_b
        self.input_width = input_width
        self.output_width_a = output_width_a
        self.output_width_b = output_width_b

        self.addr = Signal(input_width, name="addr")
        self.data_a = Signal(output_width_a, name="data_a")
        self.data_b = Signal(output_width_b, name="data_b")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        n = max(len(self.values_a), len(self.values_b), 1)
        owa = self.output_width_a
        owb = self.output_width_b

        mem_a = Memory(shape=owa, depth=n, init=self.values_a)
        mem_b = Memory(shape=owb, depth=n, init=self.values_b)
        m.submodules.rd_a = rd_a = mem_a.read_port()
        m.submodules.rd_b = rd_b = mem_b.read_port()

        m.d.comb += [rd_a.addr.eq(self.addr), rd_b.addr.eq(self.addr)]

        da_r = Signal(owa, name="da_r")
        db_r = Signal(owb, name="db_r")
        m.d.sync += [da_r.eq(rd_a.data), db_r.eq(rd_b.data)]
        m.d.comb += [self.data_a.eq(da_r), self.data_b.eq(db_r)]

        return m
