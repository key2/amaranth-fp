"""Leading zero/one counter (3-level tree)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["LZOC3"]


class LZOC3(PipelinedComponent):
    """Leading zero/one counter (3-level tree).

    Parameters
    ----------
    width : int
    count_zeros : bool
    """

    def __init__(self, width: int, count_zeros: bool = True) -> None:
        super().__init__()
        self.width = width
        self.count_zeros = count_zeros
        self.x = Signal(width, name="x")
        self.count = Signal(range(width + 1), name="count")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        cnt = Signal(range(w + 1), name="cnt")
        # Count leading zeros/ones
        m.d.comb += cnt.eq(0)
        for i in range(w):
            bit_val = 0 if self.count_zeros else 1
            with m.If(self.x[w - 1 - i] == bit_val):
                m.d.comb += cnt.eq(i + 1)
        out = Signal(range(w + 1), name="out")
        m.d.sync += out.eq(cnt)
        m.d.comb += self.count.eq(out)
        return m
