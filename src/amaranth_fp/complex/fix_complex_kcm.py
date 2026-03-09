"""Fixed-point complex KCM (constant multiplier)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixComplexKCM"]


class FixComplexKCM(PipelinedComponent):
    """Fixed-point complex KCM (constant multiplier).

    Parameters
    ----------
    msb_in, lsb_in : int
        MSB/LSB of input.
    constant_re, constant_im : float
        Real/imag parts of constant.
    """

    def __init__(self, msb_in: int, lsb_in: int, constant_re: float, constant_im: float) -> None:
        super().__init__()
        self.msb_in = msb_in
        self.lsb_in = lsb_in
        self.constant_re = constant_re
        self.constant_im = constant_im
        w = msb_in - lsb_in + 1
        self.x_re = Signal(w, name="x_re")
        self.x_im = Signal(w, name="x_im")
        self.o_re = Signal(w, name="o_re")
        self.o_im = Signal(w, name="o_im")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.msb_in - self.lsb_in + 1
        # Stage 0: multiply by constant (simplified integer approx)
        prod_re = Signal(2 * w, name="prod_re")
        prod_im = Signal(2 * w, name="prod_im")
        # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        cre = int(self.constant_re * (1 << (-self.lsb_in)))
        cim = int(self.constant_im * (1 << (-self.lsb_in)))
        m.d.comb += [
            prod_re.eq(self.x_re * cre - self.x_im * cim),
            prod_im.eq(self.x_re * cim + self.x_im * cre),
        ]
        # Pipeline stage 1
        pr1 = Signal(2 * w, name="pr1")
        pi1 = Signal(2 * w, name="pi1")
        m.d.sync += [pr1.eq(prod_re), pi1.eq(prod_im)]
        # Stage 2: truncate
        or2 = Signal(w, name="or2")
        oi2 = Signal(w, name="oi2")
        m.d.sync += [or2.eq(pr1[:w]), oi2.eq(pi1[:w])]
        m.d.comb += [self.o_re.eq(or2), self.o_im.eq(oi2)]
        return m
