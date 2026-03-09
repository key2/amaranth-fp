"""Fixed-point sine and cosine using CORDIC (pipelined, fully unrolled)."""
from __future__ import annotations

import math

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixSinCos"]


class FixSinCos(PipelinedComponent):
    """Fixed-point sine and cosine via CORDIC vectoring mode.

    Parameters
    ----------
    width : int
        Input/output bit width.
    iterations : int or None
        CORDIC iterations (default: width).

    Attributes
    ----------
    angle : Signal(width), in
        Input angle in fixed-point (full range maps to [0, 2*pi)).
    sin_out : Signal(width), out
    cos_out : Signal(width), out
    """

    def __init__(self, width: int, iterations: int | None = None) -> None:
        super().__init__()
        self.width = width
        self.iterations = iterations if iterations is not None else width
        self.angle = Signal(width, name="angle")
        self.sin_out = Signal(width, name="sin_out")
        self.cos_out = Signal(width, name="cos_out")
        self.latency = self.iterations + 2

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        n = self.iterations

        # Precompute CORDIC angles: atan(2^-i) scaled to width bits
        # Full circle = 2^width
        scale = (1 << w) / (2 * math.pi)
        atan_table = []
        for i in range(n):
            angle = math.atan(2.0 ** (-i))
            atan_table.append(int(round(angle * scale)) & ((1 << w) - 1))

        # CORDIC gain K = product of 1/sqrt(1+2^-2i) for i=0..n-1
        K = 1.0
        for i in range(n):
            K *= 1.0 / math.sqrt(1.0 + 2.0 ** (-2 * i))
        K_fp = int(round(K * (1 << (w - 1))))

        # Stage 0: Initialize x=K, y=0, z=angle
        x_init = Signal(signed(w + 2), name="x_init")
        y_init = Signal(signed(w + 2), name="y_init")
        z_init = Signal(signed(w + 2), name="z_init")

        m.d.comb += [
            x_init.eq(K_fp),
            y_init.eq(0),
            z_init.eq(self.angle),
        ]

        x_r = Signal(signed(w + 2), name="x_r0")
        y_r = Signal(signed(w + 2), name="y_r0")
        z_r = Signal(signed(w + 2), name="z_r0")
        m.d.sync += [x_r.eq(x_init), y_r.eq(y_init), z_r.eq(z_init)]

        prev_x, prev_y, prev_z = x_r, y_r, z_r

        # Unrolled CORDIC iterations, each is a pipeline stage
        for i in range(n):
            next_x = Signal(signed(w + 2), name=f"x_{i + 1}")
            next_y = Signal(signed(w + 2), name=f"y_{i + 1}")
            next_z = Signal(signed(w + 2), name=f"z_{i + 1}")

            x_shift = Signal(signed(w + 2), name=f"xs_{i}")
            y_shift = Signal(signed(w + 2), name=f"ys_{i}")
            m.d.comb += [
                x_shift.eq(prev_x >> i),
                y_shift.eq(prev_y >> i),
            ]

            with m.If(prev_z >= 0):
                m.d.sync += [
                    next_x.eq(prev_x - y_shift),
                    next_y.eq(prev_y + x_shift),
                    next_z.eq(prev_z - atan_table[i]),
                ]
            with m.Else():
                m.d.sync += [
                    next_x.eq(prev_x + y_shift),
                    next_y.eq(prev_y - x_shift),
                    next_z.eq(prev_z + atan_table[i]),
                ]

            prev_x, prev_y, prev_z = next_x, next_y, next_z

        # Output register
        cos_r = Signal(w, name="cos_r")
        sin_r = Signal(w, name="sin_r")
        m.d.sync += [
            cos_r.eq(prev_x[:w]),
            sin_r.eq(prev_y[:w]),
        ]
        m.d.comb += [
            self.cos_out.eq(cos_r),
            self.sin_out.eq(sin_r),
        ]

        return m
