"""3D norm using two CORDIC stages (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent
from .fix_2d_norm_cordic import Fix2DNormCORDIC

__all__ = ["Fix3DNormCORDIC"]


class Fix3DNormCORDIC(PipelinedComponent):
    """3D norm sqrt(x^2+y^2+z^2) using two CORDIC 2D norms.

    First computes r_xy = norm(x,y), then norm(r_xy, z).

    Parameters
    ----------
    width : int
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.x = Signal(signed(width), name="x")
        self.y = Signal(signed(width), name="y")
        self.z = Signal(signed(width), name="z")
        self.norm = Signal(width, name="norm")
        self.latency = 2 * (width + 2)

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width

        norm_xy = Fix2DNormCORDIC(w)
        norm_xyz = Fix2DNormCORDIC(w)
        m.submodules.norm_xy = norm_xy
        m.submodules.norm_xyz = norm_xyz

        m.d.comb += [norm_xy.x.eq(self.x), norm_xy.y.eq(self.y)]

        # Delay z to align with norm_xy output
        z_d = self.z
        for i in range(norm_xy.latency):
            z_next = Signal(signed(w), name=f"z_d{i+1}")
            m.d.sync += z_next.eq(z_d)
            z_d = z_next

        m.d.comb += [
            norm_xyz.x.eq(norm_xy.norm),
            norm_xyz.y.eq(z_d),
            self.norm.eq(norm_xyz.norm),
        ]

        return m
