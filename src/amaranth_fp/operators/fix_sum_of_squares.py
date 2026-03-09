"""Sum of squares without sqrt (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixSumOfSquares"]


class FixSumOfSquares(PipelinedComponent):
    """Compute sum of squares: x0^2 + x1^2 + ... + x_{n-1}^2.

    Parameters
    ----------
    width : int
        Input element width.
    n_inputs : int
        Number of inputs.
    """

    def __init__(self, width: int, n_inputs: int) -> None:
        super().__init__()
        self.width = width
        self.n_inputs = n_inputs
        self.inputs = [Signal(signed(width), name=f"x_{i}") for i in range(n_inputs)]
        self.output = Signal(2 * width + n_inputs.bit_length(), name="sum_sq")
        self.latency = 3

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        n = self.n_inputs
        ow = 2 * w + n.bit_length()

        # Stage 0→1: compute squares
        squares = []
        for i in range(n):
            sq = Signal(2 * w, name=f"sq_{i}")
            m.d.comb += sq.eq(self.inputs[i] * self.inputs[i])
            sq_r = Signal(2 * w, name=f"sq_r_{i}")
            m.d.sync += sq_r.eq(sq)
            squares.append(sq_r)

        # Stage 1→2: sum
        total = Signal(ow, name="total")
        if squares:
            parts = []
            for sq in squares:
                p = Signal(ow, name=f"sp_{sq.name}")
                m.d.comb += p.eq(sq)
                parts.append(p)
            m.d.comb += total.eq(sum(parts))

        total_r = Signal(ow, name="total_r")
        m.d.sync += total_r.eq(total)

        # Stage 2→3: output register
        out_r = Signal(ow, name="sos_out")
        m.d.sync += out_r.eq(total_r)
        m.d.comb += self.output.eq(out_r)

        return m
