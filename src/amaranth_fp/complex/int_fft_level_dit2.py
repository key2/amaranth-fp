"""Integer FFT DIT-2 level."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["IntFFTLevelDIT2"]


class IntFFTLevelDIT2(PipelinedComponent):
    """Integer FFT DIT-2 level.

    Parameters
    ----------
    width : int
        Bit width.
    n : int
        FFT size.
    """

    def __init__(self, width: int, n: int) -> None:
        super().__init__()
        self.width = width
        self.n = n
        self.x_re = Signal(width, name="x_re")
        self.x_im = Signal(width, name="x_im")
        self.o_re = Signal(width, name="o_re")
        self.o_im = Signal(width, name="o_im")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        r1 = Signal(w, name="r1"); i1 = Signal(w, name="i1")
        m.d.sync += [r1.eq(self.x_re), i1.eq(self.x_im)]
        r2 = Signal(w, name="r2"); i2 = Signal(w, name="i2")
        m.d.sync += [r2.eq(r1), i2.eq(i1)]
        m.d.comb += [self.o_re.eq(r2), self.o_im.eq(i2)]
        return m
