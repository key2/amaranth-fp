"""FIR filter — transposed form (pipelined, latency = len(coefficients))."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixFIR"]


class FixFIR(PipelinedComponent):
    """Fixed-point FIR filter in transposed form.

    Parameters
    ----------
    input_width : int
    output_width : int
    coefficients : list[int]
    coeff_width : int
    """

    def __init__(self, input_width: int, output_width: int,
                 coefficients: list[int], coeff_width: int) -> None:
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.coefficients = list(coefficients)
        self.coeff_width = coeff_width
        self.x = Signal(input_width, name="x")
        self.y = Signal(output_width, name="y")
        self.latency = len(coefficients)

    def elaborate(self, platform) -> Module:
        m = Module()
        iw = self.input_width
        ow = self.output_width
        cw = self.coeff_width
        n = len(self.coefficients)

        # Transposed FIR: each tap computes c[i]*x and adds to accumulator chain
        # y[n] = c[0]*x[n] + c[1]*x[n-1] + ... + c[N-1]*x[n-N+1]
        # Transposed: acc[i] = c[i]*x + acc[i+1] (registered)
        acc_width = iw + cw + n  # enough bits

        # Build chain from last to first
        acc = [Signal(acc_width, name=f"acc_{i}") for i in range(n)]

        for i in range(n - 1, -1, -1):
            prod = Signal(iw + cw, name=f"prod_{i}")
            m.d.comb += prod.eq(self.x * self.coefficients[i])

            if i == n - 1:
                m.d.sync += acc[i].eq(prod)
            else:
                m.d.sync += acc[i].eq(prod + acc[i + 1])

        m.d.comb += self.y.eq(acc[0][:ow])

        return m
