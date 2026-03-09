"""Generic LUT primitive (pipelined)."""
from __future__ import annotations

from amaranth import *
from amaranth.lib.memory import Memory

from ..pipelined import PipelinedComponent

__all__ = ["GenericLut"]


class GenericLut(PipelinedComponent):
    """Configurable lookup table.

    Parameters
    ----------
    input_width : int
        Number of address bits.
    output_width : int
        Width of each table entry.
    contents : list[int]
        Table contents (length must be ``2**input_width``).
    """

    def __init__(self, input_width: int, output_width: int, contents: list[int]) -> None:
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.contents = list(contents)
        assert len(self.contents) == (1 << input_width)
        self.addr = Signal(input_width, name="addr")
        self.data = Signal(output_width, name="data")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        mem = Memory(shape=unsigned(self.output_width),
                     depth=1 << self.input_width, init=self.contents)
        m.submodules.mem = mem
        rp = mem.read_port()
        m.d.comb += [
            rp.addr.eq(self.addr),
            rp.en.eq(1),
            self.data.eq(rp.data),
        ]
        return m
