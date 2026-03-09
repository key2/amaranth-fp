"""Simple (non-piecewise) polynomial approximation (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent
from .fix_horner import FixHornerEvaluator

__all__ = ["FixFunctionBySimplePoly"]


class FixFunctionBySimplePoly(PipelinedComponent):
    """Approximate a function using a single polynomial via Horner's method.

    Parameters
    ----------
    coefficients : list[int]
        Fixed-point polynomial coefficients [c0, c1, ..., cn].
    input_width : int
    output_width : int
    """

    def __init__(self, coefficients: list[int], input_width: int, output_width: int) -> None:
        super().__init__()
        self.coefficients = coefficients
        self.input_width = input_width
        self.output_width = output_width
        coeff_width = max(max(c.bit_length() + 1 for c in coefficients) if coefficients else 8, output_width)
        self._horner = FixHornerEvaluator(coefficients, input_width, coeff_width, output_width)
        self.x = Signal(input_width, name="x")
        self.result = Signal(output_width, name="result")
        self.latency = self._horner.latency

    def elaborate(self, platform) -> Module:
        m = Module()
        m.submodules.horner = h = self._horner
        m.d.comb += [h.x.eq(self.x), self.result.eq(h.result)]
        return m
