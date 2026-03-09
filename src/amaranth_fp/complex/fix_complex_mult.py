"""Fixed-point complex multiplication (pipelined, Karatsuba trick)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixComplexMult"]


class FixComplexMult(PipelinedComponent):
    """Complex multiplication using 3 multiplies (Karatsuba).

    (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    Karatsuba: k1=a*c, k2=b*d, k3=(a+b)*(c+d) => re=k1-k2, im=k3-k1-k2.

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
        self.o_re = Signal(signed(2 * width + 1), name="o_re")
        self.o_im = Signal(signed(2 * width + 1), name="o_im")
        self.latency = 3

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        pw = 2 * w  # product width

        # Stage 0: compute sums for Karatsuba
        a_plus_b = Signal(signed(w + 1), name="a_plus_b")
        c_plus_d = Signal(signed(w + 1), name="c_plus_d")
        m.d.comb += [
            a_plus_b.eq(self.a_re + self.a_im),
            c_plus_d.eq(self.b_re + self.b_im),
        ]

        a_re_r1 = Signal(signed(w), name="a_re_r1")
        a_im_r1 = Signal(signed(w), name="a_im_r1")
        b_re_r1 = Signal(signed(w), name="b_re_r1")
        b_im_r1 = Signal(signed(w), name="b_im_r1")
        apb_r1 = Signal(signed(w + 1), name="apb_r1")
        cpd_r1 = Signal(signed(w + 1), name="cpd_r1")
        m.d.sync += [
            a_re_r1.eq(self.a_re), a_im_r1.eq(self.a_im),
            b_re_r1.eq(self.b_re), b_im_r1.eq(self.b_im),
            apb_r1.eq(a_plus_b), cpd_r1.eq(c_plus_d),
        ]

        # Stage 1: 3 multiplies
        k1 = Signal(signed(pw), name="k1")
        k2 = Signal(signed(pw), name="k2")
        k3 = Signal(signed(pw + 2), name="k3")
        m.d.comb += [
            k1.eq(a_re_r1 * b_re_r1),
            k2.eq(a_im_r1 * b_im_r1),
            k3.eq(apb_r1 * cpd_r1),
        ]

        k1_r2 = Signal(signed(pw), name="k1_r2")
        k2_r2 = Signal(signed(pw), name="k2_r2")
        k3_r2 = Signal(signed(pw + 2), name="k3_r2")
        m.d.sync += [k1_r2.eq(k1), k2_r2.eq(k2), k3_r2.eq(k3)]

        # Stage 2: combine
        o_re_comb = Signal(signed(pw + 1), name="o_re_comb")
        o_im_comb = Signal(signed(pw + 3), name="o_im_comb")
        m.d.comb += [
            o_re_comb.eq(k1_r2 - k2_r2),
            o_im_comb.eq(k3_r2 - k1_r2 - k2_r2),
        ]

        o_re_r = Signal(signed(pw + 1), name="o_re_r")
        o_im_r = Signal(signed(pw + 1), name="o_im_r")
        m.d.sync += [
            o_re_r.eq(o_re_comb),
            o_im_r.eq(o_im_comb[:pw + 1]),
        ]
        m.d.comb += [self.o_re.eq(o_re_r), self.o_im.eq(o_im_r)]

        return m
