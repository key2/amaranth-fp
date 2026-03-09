"""2D/3D fixed-point vector norm (pipelined, 6 stages).

Compute sqrt(x^2 + y^2) or sqrt(x^2 + y^2 + z^2).
"""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixNorm"]


class FixNorm(PipelinedComponent):
    """Fixed-point vector norm.

    Parameters
    ----------
    width : int
        Component bit width.
    dimensions : int
        2 or 3.

    Attributes
    ----------
    inputs : list[Signal(width)]
    result : Signal(width)
    """

    def __init__(self, width: int, dimensions: int = 2) -> None:
        super().__init__()
        assert dimensions in (2, 3)
        self.width = width
        self.dimensions = dimensions
        self.inputs = [Signal(width, name=f"v_{i}") for i in range(dimensions)]
        self.result = Signal(width, name="result")
        self.latency = 6

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        dims = self.dimensions
        sq_w = 2 * w

        # Stage 0: Square each input
        squares = []
        for i in range(dims):
            sq = Signal(sq_w, name=f"sq_{i}")
            m.d.comb += sq.eq(self.inputs[i] * self.inputs[i])
            squares.append(sq)

        sq_r1 = []
        for i in range(dims):
            r = Signal(sq_w, name=f"sq_r1_{i}")
            m.d.sync += r.eq(squares[i])
            sq_r1.append(r)

        # Stage 1: Sum of squares
        sum_sq = Signal(sq_w + 2, name="sum_sq")
        if dims == 2:
            m.d.comb += sum_sq.eq(sq_r1[0] + sq_r1[1])
        else:
            m.d.comb += sum_sq.eq(sq_r1[0] + sq_r1[1] + sq_r1[2])

        sum_r2 = Signal(sq_w + 2, name="sum_r2")
        m.d.sync += sum_r2.eq(sum_sq)

        # Stages 2-4: Integer square root (Newton-Raphson, 3 iterations)
        # Initial guess: shift right by half the bit width
        guess = Signal(w + 1, name="guess")
        m.d.comb += guess.eq(sum_r2 >> w)

        g_r3 = Signal(w + 1, name="g_r3")
        m.d.sync += g_r3.eq(Mux(guess == 0, 1, guess))

        # Iteration 1
        div1 = Signal(sq_w + 2, name="div1")
        m.d.comb += div1.eq(Mux(g_r3 == 0, 0, sum_r2))  # sum_r2 / g_r3
        new_g1 = Signal(w + 1, name="new_g1")
        # Newton: g = (g + S/g) / 2; approximate with shifts
        m.d.comb += new_g1.eq((g_r3 + (sum_r2 >> (w - 1))) >> 1)

        g_r4 = Signal(w + 1, name="g_r4")
        sum_r4 = Signal(sq_w + 2, name="sum_r4")
        m.d.sync += [g_r4.eq(Mux(new_g1 == 0, 1, new_g1)), sum_r4.eq(sum_r2)]

        # Iteration 2
        new_g2 = Signal(w + 1, name="new_g2")
        m.d.comb += new_g2.eq((g_r4 + (sum_r4 >> (w - 1))) >> 1)

        g_r5 = Signal(w + 1, name="g_r5")
        m.d.sync += g_r5.eq(Mux(new_g2 == 0, 1, new_g2))

        # Stage 5: Output
        out_r = Signal(w, name="out_r")
        m.d.sync += out_r.eq(g_r5[:w])
        m.d.comb += self.result.eq(out_r)

        return m
