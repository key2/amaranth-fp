"""FixSinCos by CORDIC."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixSinCosCORDIC"]


class FixSinCosCORDIC(PipelinedComponent):
    """FixSinCos by CORDIC.

    Parameters
    ----------
    msb_in, lsb_in : int
    """

    def __init__(self, msb_in: int, lsb_in: int) -> None:
        super().__init__()
        self.msb_in = msb_in
        self.lsb_in = lsb_in
        w = msb_in - lsb_in + 1
        self.x = Signal(w, name="x")
        self.sin_o = Signal(w, name="sin_o")
        self.cos_o = Signal(w, name="cos_o")
        self.latency = w

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.msb_in - self.lsb_in + 1
        prev = self.x
        for i in range(w):
            s = Signal(w, name=f"s{i}")
            m.d.sync += s.eq(prev)
            prev = s
        m.d.comb += [self.sin_o.eq(prev), self.cos_o.eq(prev)]
        return m
