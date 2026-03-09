"""Integer FFT butterfly (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["IntFFTButterfly"]


class IntFFTButterfly(PipelinedComponent):
    """Radix-2 DIT FFT butterfly.

    Computes: A' = A + W*B, B' = A - W*B

    Parameters
    ----------
    width : int
        Real/imag component width.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.a_re = Signal(width, name="a_re")
        self.a_im = Signal(width, name="a_im")
        self.b_re = Signal(width, name="b_re")
        self.b_im = Signal(width, name="b_im")
        self.o_a_re = Signal(width, name="o_a_re")
        self.o_a_im = Signal(width, name="o_a_im")
        self.o_b_re = Signal(width, name="o_b_re")
        self.o_b_im = Signal(width, name="o_b_im")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        # Simplified butterfly without twiddle (W=1):
        # A' = A + B, B' = A - B
        m.d.sync += [
            self.o_a_re.eq(self.a_re + self.b_re),
            self.o_a_im.eq(self.a_im + self.b_im),
            self.o_b_re.eq(self.a_re - self.b_re),
            self.o_b_im.eq(self.a_im - self.b_im),
        ]
        return m
