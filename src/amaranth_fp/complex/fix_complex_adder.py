"""Fixed-point complex addition (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixComplexAdder"]


class FixComplexAdder(PipelinedComponent):
    """Complex addition: (a_re+a_im*i) + (b_re+b_im*i).

    Parameters
    ----------
    width : int
        Bit width of each real/imaginary component.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.a_re = Signal(signed(width), name="a_re")
        self.a_im = Signal(signed(width), name="a_im")
        self.b_re = Signal(signed(width), name="b_re")
        self.b_im = Signal(signed(width), name="b_im")
        self.o_re = Signal(signed(width + 1), name="o_re")
        self.o_im = Signal(signed(width + 1), name="o_im")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width

        re_sum = Signal(signed(w + 1), name="re_sum")
        im_sum = Signal(signed(w + 1), name="im_sum")
        m.d.comb += [
            re_sum.eq(self.a_re + self.b_re),
            im_sum.eq(self.a_im + self.b_im),
        ]

        o_re_r = Signal(signed(w + 1), name="o_re_r")
        o_im_r = Signal(signed(w + 1), name="o_im_r")
        m.d.sync += [o_re_r.eq(re_sum), o_im_r.eq(im_sum)]
        m.d.comb += [self.o_re.eq(o_re_r), self.o_im.eq(o_im_r)]

        return m
