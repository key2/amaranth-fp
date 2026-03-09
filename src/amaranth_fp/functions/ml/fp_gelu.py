"""Floating-point GELU(x) = x * 0.5 * (1 + erf(x/sqrt(2))).

Composed from FPConstMult + FPErf + FPAdd + FPMul.
"""
from __future__ import annotations

import math

from amaranth import *

from ...format import FPFormat
from ...pipelined import PipelinedComponent
from ...operators.fp_const_mult import FPConstMult
from ...operators.fp_add import FPAdd
from ...operators.fp_mul import FPMul
from ..math.fp_erf import FPErf

__all__ = ["FPGELU"]


class FPGELU(PipelinedComponent):
    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")

        self._inv_sqrt2 = FPConstMult(fmt, 1.0 / math.sqrt(2))
        self._erf = FPErf(fmt)
        self._add = FPAdd(fmt)
        self._half = FPConstMult(fmt, 0.5)
        self._mul = FPMul(fmt)

        # x/sqrt(2) → erf → +1 → *0.5 → *x
        self.latency = (self._inv_sqrt2.latency + self._erf.latency +
                        self._add.latency + self._half.latency + self._mul.latency)

    def _encode_one(self) -> int:
        fmt = self.fmt
        return (0b01 << (1 + fmt.we + fmt.wf)) | (fmt.bias << fmt.wf)

    def elaborate(self, platform) -> Module:
        m = Module()
        m.submodules.inv_sqrt2 = inv_sqrt2 = self._inv_sqrt2
        m.submodules.erf = erf = self._erf
        m.submodules.add = add = self._add
        m.submodules.half = half = self._half
        m.submodules.mul = mul = self._mul

        one_enc = self._encode_one()

        # x / sqrt(2)
        m.d.comb += inv_sqrt2.a.eq(self.a)
        # erf(x/sqrt(2))
        m.d.comb += erf.a.eq(inv_sqrt2.o)
        # 1 + erf(x/sqrt(2))
        m.d.comb += [add.a.eq(one_enc), add.b.eq(erf.o)]
        # 0.5 * (1 + erf(...))
        m.d.comb += half.a.eq(add.o)
        # x * 0.5 * (1 + erf(...))
        total_delay = (inv_sqrt2.latency + erf.latency + add.latency + half.latency)
        x_delayed = self.a
        for i in range(total_delay):
            d = Signal(self.fmt.width, name=f"x_gelu_d{i}")
            m.d.sync += d.eq(x_delayed)
            x_delayed = d
        m.d.comb += [mul.a.eq(x_delayed), mul.b.eq(half.o)]
        m.d.comb += self.o.eq(mul.o)

        return m
