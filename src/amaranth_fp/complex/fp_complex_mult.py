"""Floating-point complex multiplication (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent
from ..format import FPFormat
from ..operators.fp_mul import FPMul
from ..operators.fp_add import FPAdd
from ..operators.fp_sub import FPSub

__all__ = ["FPComplexMultiplier"]


class FPComplexMultiplier(PipelinedComponent):
    """FP complex multiplication: (a+bi)(c+di) = (ac-bd)+(ad+bc)i.

    Uses 4 FPMul + 2 FPAdd/Sub.

    Parameters
    ----------
    fmt : FPFormat
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        w = fmt.width
        self.a_re = Signal(w, name="a_re")
        self.a_im = Signal(w, name="a_im")
        self.b_re = Signal(w, name="b_re")
        self.b_im = Signal(w, name="b_im")
        self.o_re = Signal(w, name="o_re")
        self.o_im = Signal(w, name="o_im")

        self._mul_ac = FPMul(fmt)
        self._mul_bd = FPMul(fmt)
        self._mul_ad = FPMul(fmt)
        self._mul_bc = FPMul(fmt)
        self._sub_re = FPSub(fmt)
        self._add_im = FPAdd(fmt)
        self.latency = self._mul_ac.latency + self._add_im.latency

    def elaborate(self, platform) -> Module:
        m = Module()
        m.submodules.mul_ac = mul_ac = self._mul_ac
        m.submodules.mul_bd = mul_bd = self._mul_bd
        m.submodules.mul_ad = mul_ad = self._mul_ad
        m.submodules.mul_bc = mul_bc = self._mul_bc
        m.submodules.sub_re = sub_re = self._sub_re
        m.submodules.add_im = add_im = self._add_im

        m.d.comb += [
            mul_ac.a.eq(self.a_re), mul_ac.b.eq(self.b_re),
            mul_bd.a.eq(self.a_im), mul_bd.b.eq(self.b_im),
            mul_ad.a.eq(self.a_re), mul_ad.b.eq(self.b_im),
            mul_bc.a.eq(self.a_im), mul_bc.b.eq(self.b_re),
            sub_re.a.eq(mul_ac.o), sub_re.b.eq(mul_bd.o),
            add_im.a.eq(mul_ad.o), add_im.b.eq(mul_bc.o),
            self.o_re.eq(sub_re.o), self.o_im.eq(add_im.o),
        ]

        return m
