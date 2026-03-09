"""HOTBM — Higher-Order Table-Based Method (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["HOTBM"]


class HOTBM(PipelinedComponent):
    """Higher-Order Table-Based Method for function approximation.

    Decomposes the input into sub-words and uses multiple smaller tables
    combined with additions to approximate f(x).

    Parameters
    ----------
    input_width : int
    output_width : int
    order : int
        Table decomposition order (1 = bipartite, 2 = tripartite, etc.).
    """

    def __init__(self, input_width: int, output_width: int, order: int = 2) -> None:
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.order = order
        self.x = Signal(input_width, name="x")
        self.y = Signal(output_width, name="y")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        # Simplified: single-table lookup placeholder
        o_r = Signal(self.output_width, name="o_r")
        m.d.sync += o_r.eq(self.x[:self.output_width])
        m.d.comb += self.y.eq(o_r)
        return m
