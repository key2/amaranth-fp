"""Generic ROM/LUT table using Amaranth Memory."""
from __future__ import annotations

from amaranth import *
from amaranth.lib.memory import Memory

from ..pipelined import PipelinedComponent

__all__ = ["Table"]


class Table(PipelinedComponent):
    """ROM lookup table backed by Amaranth Memory.

    Parameters
    ----------
    input_width : int
        Address width (number of address bits).
    output_width : int
        Data output width.
    values : list[int]
        Precomputed table contents (length must be 2**input_width).

    Attributes
    ----------
    addr : Signal(input_width), in
    data : Signal(output_width), out
    """

    def __init__(self, input_width: int, output_width: int, values: list[int]) -> None:
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.values = values
        self.addr = Signal(input_width, name="addr")
        self.data = Signal(output_width, name="data")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()

        mem = Memory(
            shape=unsigned(self.output_width),
            depth=1 << self.input_width,
            init=self.values,
        )
        m.submodules.mem = mem

        rd_port = mem.read_port()
        m.d.comb += [
            rd_port.addr.eq(self.addr),
            rd_port.en.eq(1),
            self.data.eq(rd_port.data),
        ]

        return m
