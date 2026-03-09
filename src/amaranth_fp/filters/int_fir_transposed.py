"""Transposed FIR filter (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["IntFIRTransposed"]


class IntFIRTransposed(PipelinedComponent):
    """Integer transposed-form FIR filter.

    In transposed form, each tap is: accum[i] = x*coeff[i] + accum[i+1].
    This naturally pipelines at one sample per clock.

    Parameters
    ----------
    width : int
        Sample width.
    coefficients : list[int]
        Filter tap coefficients.
    """

    def __init__(self, width: int, coefficients: list[int] | None = None) -> None:
        super().__init__()
        self.width = width
        self.coefficients = coefficients or [1]
        self.n_taps = len(self.coefficients)
        out_bits = width + max(c.bit_length() for c in self.coefficients if c) + (self.n_taps - 1).bit_length()
        self.x = Signal(width, name="x")
        self.y = Signal(out_bits, name="y")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.y.shape()[0] if isinstance(self.y.shape(), tuple) else len(self.y)
        # Transposed FIR
        accums = [Signal(w, name=f"acc{i}") for i in range(self.n_taps)]
        for i in range(self.n_taps):
            prod = Signal(w, name=f"prod{i}")
            m.d.comb += prod.eq(self.x * self.coefficients[i])
            if i < self.n_taps - 1:
                m.d.sync += accums[i].eq(prod + accums[i + 1])
            else:
                m.d.sync += accums[i].eq(prod)
        m.d.comb += self.y.eq(accums[0])
        return m
