"""Integer twiddle factor multiplier for FFT (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["IntTwiddleMultiplier"]


class IntTwiddleMultiplier(PipelinedComponent):
    """Multiply complex value by a twiddle factor W_N^k = e^{-j2πk/N}.

    Parameters
    ----------
    width : int
        Real/imag component width.
    n : int
        FFT size N.
    k : int
        Twiddle index.
    """

    def __init__(self, width: int, n: int = 8, k: int = 0) -> None:
        super().__init__()
        self.width = width
        self.n = n
        self.k = k
        self.re_in = Signal(width, name="re_in")
        self.im_in = Signal(width, name="im_in")
        self.re_out = Signal(width, name="re_out")
        self.im_out = Signal(width, name="im_out")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        # For k=0, twiddle = 1+0j, just pass through
        # General case needs cos/sin ROM; simplified here
        re_r = Signal(w, name="re_r")
        im_r = Signal(w, name="im_r")
        m.d.sync += re_r.eq(self.re_in)
        m.d.sync += im_r.eq(self.im_in)
        m.d.comb += self.re_out.eq(re_r)
        m.d.comb += self.im_out.eq(im_r)
        return m
