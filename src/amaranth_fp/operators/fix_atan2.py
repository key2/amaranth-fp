"""Fixed-point atan2(y, x) using CORDIC rotation mode (pipelined)."""
from __future__ import annotations

import math

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixAtan2"]


class FixAtan2(PipelinedComponent):
    """Fixed-point atan2(y, x) via CORDIC rotation mode.

    Parameters
    ----------
    width : int
        Input/output bit width.

    Attributes
    ----------
    x : Signal(width), in
    y : Signal(width), in
    angle : Signal(width), out
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.x = Signal(width, name="x")
        self.y = Signal(width, name="y")
        self.angle = Signal(width, name="angle")
        self.latency = width + 2

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        n = w  # iterations = width

        scale = (1 << w) / (2 * math.pi)
        atan_table = []
        for i in range(n):
            ang = math.atan(2.0 ** (-i))
            atan_table.append(int(round(ang * scale)) & ((1 << w) - 1))

        # Stage 0: Initialize
        x_r = Signal(signed(w + 2), name="xr0")
        y_r = Signal(signed(w + 2), name="yr0")
        z_r = Signal(signed(w + 2), name="zr0")
        m.d.sync += [x_r.eq(self.x), y_r.eq(self.y), z_r.eq(0)]

        prev_x, prev_y, prev_z = x_r, y_r, z_r

        for i in range(n):
            next_x = Signal(signed(w + 2), name=f"ax_{i + 1}")
            next_y = Signal(signed(w + 2), name=f"ay_{i + 1}")
            next_z = Signal(signed(w + 2), name=f"az_{i + 1}")

            x_shift = Signal(signed(w + 2), name=f"axs_{i}")
            y_shift = Signal(signed(w + 2), name=f"ays_{i}")
            m.d.comb += [
                x_shift.eq(prev_x >> i),
                y_shift.eq(prev_y >> i),
            ]

            # CORDIC vectoring: drive y toward 0
            with m.If(prev_y >= 0):
                m.d.sync += [
                    next_x.eq(prev_x + y_shift),
                    next_y.eq(prev_y - x_shift),
                    next_z.eq(prev_z + atan_table[i]),
                ]
            with m.Else():
                m.d.sync += [
                    next_x.eq(prev_x - y_shift),
                    next_y.eq(prev_y + x_shift),
                    next_z.eq(prev_z - atan_table[i]),
                ]

            prev_x, prev_y, prev_z = next_x, next_y, next_z

        # Output register
        angle_r = Signal(w, name="angle_r")
        m.d.sync += angle_r.eq(prev_z[:w])
        m.d.comb += self.angle.eq(angle_r)

        return m
