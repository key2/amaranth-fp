"""Floating-point swish(x) = x * sigmoid(x).

Composed from FPSigmoid + FPMul. Latency: FPSigmoid.latency + 5.
"""
from __future__ import annotations

from amaranth import *

from ...format import FPFormat
from ...pipelined import PipelinedComponent
from ...operators.fp_mul import FPMul
from .fp_sigmoid import FPSigmoid

__all__ = ["FPSwish"]


class FPSwish(PipelinedComponent):
    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")

        self._sigmoid = FPSigmoid(fmt)
        self._mul = FPMul(fmt)
        self.latency = self._sigmoid.latency + self._mul.latency

    def elaborate(self, platform) -> Module:
        m = Module()
        m.submodules.sigmoid = sigmoid = self._sigmoid
        m.submodules.mul = mul = self._mul

        m.d.comb += sigmoid.a.eq(self.a)

        # Delay x to match sigmoid latency
        x_delayed = self.a
        for i in range(sigmoid.latency):
            d = Signal(self.fmt.width, name=f"x_swish_d{i}")
            m.d.sync += d.eq(x_delayed)
            x_delayed = d

        m.d.comb += [mul.a.eq(x_delayed), mul.b.eq(sigmoid.o), self.o.eq(mul.o)]
        return m
