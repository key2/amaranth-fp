"""Intel LCELL primitive."""
from __future__ import annotations
from amaranth import *
from ...pipelined import PipelinedComponent

__all__ = ["IntelLCELL"]


class IntelLCELL(PipelinedComponent):
    """Intel LCELL (logic cell) primitive.

    Parameters
    ----------
    lut_mask : int
        LUT configuration mask.
    """

    def __init__(self, lut_mask: int = 0):
        super().__init__()
        self.lut_mask = lut_mask
        self.a = Signal(4, name="a")
        self.o = Signal(1, name="o")
        self.latency = 0

    def elaborate(self, platform) -> Module:
        m = Module()
        m.d.comb += self.o.eq(self.a[0])
        return m
