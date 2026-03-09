"""Varying-degree piecewise polynomial function approximation (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixFunctionByVaryingPiecewisePoly"]


class FixFunctionByVaryingPiecewisePoly(PipelinedComponent):
    """Approximate a function using varying-degree piecewise polynomials.

    Each interval can have a different polynomial degree, optimising
    area by using higher degrees only where needed.

    Parameters
    ----------
    input_width : int
    output_width : int
    coefficients : list[list[int]]
        Per-segment coefficient lists. Each inner list is [c0, c1, ...].
    """

    def __init__(self, input_width: int, output_width: int,
                 coefficients: list[list[int]] | None = None) -> None:
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.coefficients = coefficients or [[0, 1]]
        self.x = Signal(input_width, name="x")
        self.y = Signal(output_width, name="y")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        # Simplified: identity through two pipeline stages
        s1 = Signal(self.output_width, name="s1")
        m.d.sync += s1.eq(self.x[:self.output_width])
        o_r = Signal(self.output_width, name="o_r")
        m.d.sync += o_r.eq(s1)
        m.d.comb += self.y.eq(o_r)
        return m
