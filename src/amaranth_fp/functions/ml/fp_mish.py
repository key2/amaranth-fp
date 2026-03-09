"""Floating-point mish(x) = x * tanh(softplus(x)).

Composed from FPSoftplus + FPTanh + FPMul.
"""
from __future__ import annotations

from amaranth import *

from ...format import FPFormat
from ...pipelined import PipelinedComponent
from ...operators.fp_mul import FPMul
from ..math.fp_tanh import FPTanh
from .fp_softplus import FPSoftplus

__all__ = ["FPMish"]


class FPMish(PipelinedComponent):
    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")

        self._softplus = FPSoftplus(fmt)
        self._tanh = FPTanh(fmt)
        self._mul = FPMul(fmt)
        self.latency = self._softplus.latency + self._tanh.latency + self._mul.latency

    def elaborate(self, platform) -> Module:
        m = Module()
        m.submodules.softplus = softplus = self._softplus
        m.submodules.tanh = tanh = self._tanh
        m.submodules.mul = mul = self._mul

        m.d.comb += softplus.a.eq(self.a)
        m.d.comb += tanh.a.eq(softplus.o)

        # Delay x to match softplus + tanh latency
        x_delayed = self.a
        for i in range(softplus.latency + tanh.latency):
            d = Signal(self.fmt.width, name=f"x_mish_d{i}")
            m.d.sync += d.eq(x_delayed)
            x_delayed = d

        m.d.comb += [mul.a.eq(x_delayed), mul.b.eq(tanh.o), self.o.eq(mul.o)]
        return m
