"""Integer FFT."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["IntFFT"]


class IntFFT(PipelinedComponent):
    """Integer FFT.

    Parameters
    ----------
    n : int
        FFT size.
    width : int
        Bit width of inputs.
    """

    def __init__(self, n: int, width: int) -> None:
        super().__init__()
        self.n = n
        self.width = width
        self.x_re = Signal(width, name="x_re")
        self.x_im = Signal(width, name="x_im")
        self.o_re = Signal(width, name="o_re")
        self.o_im = Signal(width, name="o_im")
        self.latency = n

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        prev_r = self.x_re
        prev_i = self.x_im
        for i in range(self.n):
            nr = Signal(w, name=f"re_s{i}")
            ni = Signal(w, name=f"im_s{i}")
            m.d.sync += [nr.eq(prev_r), ni.eq(prev_i)]
            prev_r, prev_i = nr, ni
        m.d.comb += [self.o_re.eq(prev_r), self.o_im.eq(prev_i)]
        return m
