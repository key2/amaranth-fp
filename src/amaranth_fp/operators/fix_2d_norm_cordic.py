"""2D norm using CORDIC vectoring (pipelined)."""
from __future__ import annotations

import math

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["Fix2DNormCORDIC"]


class Fix2DNormCORDIC(PipelinedComponent):
    """2D norm sqrt(x^2+y^2) using CORDIC vectoring mode.

    Parameters
    ----------
    width : int
        Input bit width.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.x = Signal(signed(width), name="x")
        self.y = Signal(signed(width), name="y")
        self.norm = Signal(width, name="norm")
        self.latency = width + 2

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        n = w

        # CORDIC gain
        K = 1.0
        for i in range(n):
            K *= 1.0 / math.sqrt(1.0 + 2.0 ** (-2 * i))

        # Stage 0: take absolute values
        abs_x = Signal(signed(w + 2), name="abs_x")
        abs_y = Signal(signed(w + 2), name="abs_y")
        m.d.comb += [
            abs_x.eq(Mux(self.x[w - 1], -self.x, self.x)),
            abs_y.eq(Mux(self.y[w - 1], -self.y, self.y)),
        ]

        x_r = Signal(signed(w + 2), name="cx_0")
        y_r = Signal(signed(w + 2), name="cy_0")
        m.d.sync += [x_r.eq(abs_x), y_r.eq(abs_y)]

        prev_x, prev_y = x_r, y_r

        # CORDIC vectoring: drive y to 0
        for i in range(n):
            next_x = Signal(signed(w + 2), name=f"cx_{i+1}")
            next_y = Signal(signed(w + 2), name=f"cy_{i+1}")

            x_shift = Signal(signed(w + 2), name=f"cxs_{i}")
            y_shift = Signal(signed(w + 2), name=f"cys_{i}")
            m.d.comb += [x_shift.eq(prev_x >> i), y_shift.eq(prev_y >> i)]

            with m.If(prev_y >= 0):
                m.d.sync += [next_x.eq(prev_x + y_shift), next_y.eq(prev_y - x_shift)]
            with m.Else():
                m.d.sync += [next_x.eq(prev_x - y_shift), next_y.eq(prev_y + x_shift)]

            prev_x, prev_y = next_x, next_y

        # Final: multiply by K (approximated as shift)
        K_fp = int(round(K * (1 << (w - 1))))
        norm_wide = Signal(2 * w + 2, name="norm_wide")
        m.d.comb += norm_wide.eq(prev_x * K_fp)

        out_r = Signal(w, name="norm_r")
        m.d.sync += out_r.eq(norm_wide[w - 1:2 * w - 1])
        m.d.comb += self.norm.eq(out_r)

        return m
