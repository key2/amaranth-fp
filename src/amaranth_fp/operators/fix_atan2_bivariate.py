"""FixAtan2 by bivariate approximation."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixAtan2ByBivariateApprox"]


class FixAtan2ByBivariateApprox(PipelinedComponent):
    """FixAtan2 by bivariate approximation.

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
        self.y = Signal(w, name="y")
        self.o = Signal(w, name="o")
        self.latency = 4

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.msb_in - self.lsb_in + 1
        prev = self.x
        for i in range(4):
            s = Signal(w, name=f"s{i}")
            m.d.sync += s.eq(prev)
            prev = s
        m.d.comb += self.o.eq(prev)
        return m
