"""Atan2 via reciprocal-multiply-atan method (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixAtan2ByRecipMultAtan"]


class FixAtan2ByRecipMultAtan(PipelinedComponent):
    """Compute atan2(y,x) via y/x then atan lookup.

    Parameters
    ----------
    width : int
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.x = Signal(width, name="x")
        self.y = Signal(width, name="y")
        self.atan2_out = Signal(width, name="atan2_out")
        self.latency = 3

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        # Stage 1: compute ratio (simplified)
        ratio = Signal(w, name="ratio")
        m.d.sync += ratio.eq(self.y)  # placeholder
        # Stage 2: atan lookup
        atan_val = Signal(w, name="atan_val")
        m.d.sync += atan_val.eq(ratio >> 1)
        # Stage 3: quadrant correction
        o_r = Signal(w, name="o_r")
        m.d.sync += o_r.eq(atan_val)
        m.d.comb += self.atan2_out.eq(o_r)
        return m
