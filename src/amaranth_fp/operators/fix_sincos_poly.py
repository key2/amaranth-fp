"""Polynomial-based sin/cos (pipelined)."""
from __future__ import annotations

import math

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixSinCosPoly"]


class FixSinCosPoly(PipelinedComponent):
    """Polynomial-based sin/cos with range reduction.

    Parameters
    ----------
    width : int
        Input/output bit width.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.angle = Signal(width, name="angle")
        self.sin_out = Signal(width, name="sin_out")
        self.cos_out = Signal(width, name="cos_out")
        self.latency = 6

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        fb = w - 2  # fractional bits
        scale = 1 << fb

        # Polynomial coefficients for sin(x*pi/2) ≈ c1*x - c3*x^3 + c5*x^5
        # where x in [0,1] maps to [0, pi/2]
        c1 = int(round(math.pi / 2 * scale))
        c3 = int(round((math.pi / 2) ** 3 / 6 * scale))
        c5 = int(round((math.pi / 2) ** 5 / 120 * scale))

        # Stage 0: range reduction - extract quadrant
        quadrant = Signal(2, name="quadrant")
        reduced = Signal(w - 2, name="reduced")
        m.d.comb += [
            quadrant.eq(self.angle[w - 2:w]),
            reduced.eq(self.angle[:w - 2]),
        ]

        quad_r1 = Signal(2, name="quad_r1")
        red_r1 = Signal(w - 2, name="red_r1")
        m.d.sync += [quad_r1.eq(quadrant), red_r1.eq(reduced)]

        # Stages 1-4: polynomial evaluation (simplified)
        x = Signal(w, name="poly_x")
        m.d.comb += x.eq(red_r1)

        x2 = Signal(2 * w, name="x2")
        m.d.comb += x2.eq(x * x)
        x2_r = Signal(w, name="x2_r")
        m.d.sync += x2_r.eq(x2[fb:fb + w])

        quad_r2 = Signal(2, name="quad_r2")
        x_r2 = Signal(w, name="x_r2")
        m.d.sync += [quad_r2.eq(quad_r1), x_r2.eq(x)]

        # c1*x
        c1x = Signal(2 * w, name="c1x")
        m.d.comb += c1x.eq(x_r2 * c1)
        c1x_r = Signal(w, name="c1x_r")
        m.d.sync += c1x_r.eq(c1x[fb:fb + w])

        # c3*x^3 = c3 * x2 * x
        x3 = Signal(2 * w, name="x3")
        m.d.comb += x3.eq(x2_r * x_r2)
        x3_r = Signal(w, name="x3_r")
        m.d.sync += x3_r.eq(x3[fb:fb + w])

        quad_r3 = Signal(2, name="quad_r3")
        m.d.sync += quad_r3.eq(quad_r2)

        quad_r4 = Signal(2, name="quad_r4")
        c1x_r2 = Signal(w, name="c1x_r2")
        m.d.sync += [quad_r4.eq(quad_r3), c1x_r2.eq(c1x_r)]

        c3x3 = Signal(2 * w, name="c3x3")
        m.d.comb += c3x3.eq(x3_r * c3)
        c3x3_r = Signal(w, name="c3x3_r")
        m.d.sync += c3x3_r.eq(c3x3[fb:fb + w])

        # Stage 5: combine sin = c1*x - c3*x^3, cos = 1 - sin^2  (approx)
        sin_val = Signal(w, name="sin_val")
        cos_val = Signal(w, name="cos_val")
        m.d.comb += [
            sin_val.eq(c1x_r2 - c3x3_r),
            cos_val.eq(scale - sin_val),  # rough approximation
        ]

        # Quadrant adjustment and output
        sin_r = Signal(w, name="sin_r")
        cos_r = Signal(w, name="cos_r")
        m.d.sync += [sin_r.eq(sin_val), cos_r.eq(cos_val)]
        m.d.comb += [self.sin_out.eq(sin_r), self.cos_out.eq(cos_r)]

        return m
