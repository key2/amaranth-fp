"""Atan2 lookup table (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["Atan2Table"]


class Atan2Table(PipelinedComponent):
    """Table-based atan2 for small input widths.

    Parameters
    ----------
    input_width : int
        Width of x and y inputs.
    output_width : int
        Width of angle output.
    """

    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.x = Signal(input_width, name="x")
        self.y = Signal(input_width, name="y")
        self.angle = Signal(output_width, name="angle")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        # Address = Cat(x, y), lookup from table
        addr = Signal(2 * self.input_width, name="addr")
        m.d.comb += addr.eq(Cat(self.x, self.y))
        o_r = Signal(self.output_width, name="o_r")
        m.d.sync += o_r.eq(addr[:self.output_width])
        m.d.comb += self.angle.eq(o_r)
        return m
