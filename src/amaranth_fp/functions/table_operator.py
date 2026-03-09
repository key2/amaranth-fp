"""Generic table operator (pipelined)."""
from __future__ import annotations

from amaranth import *
from amaranth.lib.memory import Memory

from ..pipelined import PipelinedComponent

__all__ = ["TableOperator"]


class TableOperator(PipelinedComponent):
    """Generic table-based operator with configurable contents.

    Parameters
    ----------
    input_width : int
    output_width : int
    contents : list[int] | None
    """

    def __init__(self, input_width: int, output_width: int,
                 contents: list[int] | None = None) -> None:
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        depth = 1 << input_width
        self.contents = list(contents) if contents else list(range(depth))
        assert len(self.contents) == depth
        self.x = Signal(input_width, name="x")
        self.y = Signal(output_width, name="y")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        mem = Memory(shape=unsigned(self.output_width),
                     depth=1 << self.input_width, init=self.contents)
        m.submodules.mem = mem
        rp = mem.read_port()
        m.d.comb += [
            rp.addr.eq(self.x),
            rp.en.eq(1),
            self.y.eq(rp.data),
        ]
        return m
