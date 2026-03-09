"""Radix-2 FFT butterfly (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent
from .fix_complex_mult import FixComplexMult
from .fix_complex_adder import FixComplexAdder

__all__ = ["FixComplexR2Butterfly"]


class FixComplexR2Butterfly(PipelinedComponent):
    """Radix-2 butterfly: X = A + W*B, Y = A - W*B.

    Parameters
    ----------
    width : int
        Bit width of each component.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.a_re = Signal(signed(width), name="a_re")
        self.a_im = Signal(signed(width), name="a_im")
        self.b_re = Signal(signed(width), name="b_re")
        self.b_im = Signal(signed(width), name="b_im")
        self.w_re = Signal(signed(width), name="w_re")
        self.w_im = Signal(signed(width), name="w_im")
        pw = 2 * width + 1
        self.x_re = Signal(signed(pw + 1), name="x_re")
        self.x_im = Signal(signed(pw + 1), name="x_im")
        self.y_re = Signal(signed(pw + 1), name="y_re")
        self.y_im = Signal(signed(pw + 1), name="y_im")
        self.latency = 4  # mult(3) + add(1)

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        pw = 2 * w + 1

        # W*B multiplication (latency 3)
        mult = FixComplexMult(w)
        m.submodules.mult = mult
        m.d.comb += [
            mult.a_re.eq(self.w_re),
            mult.a_im.eq(self.w_im),
            mult.b_re.eq(self.b_re),
            mult.b_im.eq(self.b_im),
        ]

        # Delay A by 3 cycles to align with mult output
        a_re_d = self.a_re
        a_im_d = self.a_im
        for i in range(3):
            a_re_next = Signal(signed(w), name=f"a_re_d{i+1}")
            a_im_next = Signal(signed(w), name=f"a_im_d{i+1}")
            m.d.sync += [a_re_next.eq(a_re_d), a_im_next.eq(a_im_d)]
            a_re_d = a_re_next
            a_im_d = a_im_next

        # Stage 3→4: add/sub
        wb_re = mult.o_re
        wb_im = mult.o_im

        x_re_c = Signal(signed(pw + 1), name="x_re_c")
        x_im_c = Signal(signed(pw + 1), name="x_im_c")
        y_re_c = Signal(signed(pw + 1), name="y_re_c")
        y_im_c = Signal(signed(pw + 1), name="y_im_c")
        m.d.comb += [
            x_re_c.eq(a_re_d + wb_re),
            x_im_c.eq(a_im_d + wb_im),
            y_re_c.eq(a_re_d - wb_re),
            y_im_c.eq(a_im_d - wb_im),
        ]

        x_re_r = Signal(signed(pw + 1), name="x_re_r")
        x_im_r = Signal(signed(pw + 1), name="x_im_r")
        y_re_r = Signal(signed(pw + 1), name="y_re_r")
        y_im_r = Signal(signed(pw + 1), name="y_im_r")
        m.d.sync += [
            x_re_r.eq(x_re_c), x_im_r.eq(x_im_c),
            y_re_r.eq(y_re_c), y_im_r.eq(y_im_c),
        ]
        m.d.comb += [
            self.x_re.eq(x_re_r), self.x_im.eq(x_im_r),
            self.y_re.eq(y_re_r), self.y_im.eq(y_im_r),
        ]

        return m
