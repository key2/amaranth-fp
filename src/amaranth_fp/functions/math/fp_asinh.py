"""Floating-point asinh(x) = log(x + sqrt(x^2 + 1)).

Composed from FPSquare + FPAdd + FPSqrt + FPLog.
Latency: sum of submodule latencies.
"""
from __future__ import annotations

import math

from amaranth import *

from ...format import FPFormat
from ...pipelined import PipelinedComponent
from ...operators.fp_mul import FPMul
from ...operators.fp_add import FPAdd
from ...operators.fp_sqrt import FPSqrt
from ...operators.fp_log import FPLog

__all__ = ["FPAsinh"]


class FPAsinh(PipelinedComponent):
    """Pipelined floating-point asinh(x) = log(x + sqrt(x^2 + 1)).

    Parameters
    ----------
    fmt : FPFormat
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")

        self._square = FPMul(fmt)   # x^2
        self._add1 = FPAdd(fmt)     # x^2 + 1
        self._sqrt = FPSqrt(fmt)    # sqrt(x^2 + 1)
        self._add2 = FPAdd(fmt)     # x + sqrt(x^2 + 1)
        self._log = FPLog(fmt)      # log(x + sqrt(x^2 + 1))

        self.latency = (self._square.latency + self._add1.latency +
                        self._sqrt.latency + self._add2.latency + self._log.latency)

    def _encode_one(self) -> int:
        """Encode 1.0 in FloPoCo internal format."""
        fmt = self.fmt
        return (0b01 << (1 + fmt.we + fmt.wf)) | (fmt.bias << fmt.wf)

    def elaborate(self, platform) -> Module:
        m = Module()
        m.submodules.square = sq = self._square
        m.submodules.add1 = add1 = self._add1
        m.submodules.sqrt = sqrt = self._sqrt
        m.submodules.add2 = add2 = self._add2
        m.submodules.log = log = self._log

        one_enc = self._encode_one()

        # x^2
        m.d.comb += [sq.a.eq(self.a), sq.b.eq(self.a)]
        # x^2 + 1
        m.d.comb += [add1.a.eq(sq.o), add1.b.eq(one_enc)]
        # sqrt(x^2 + 1)
        m.d.comb += sqrt.a.eq(add1.o)
        # x + sqrt(x^2 + 1)
        # Need to delay x to match the pipeline depth of sq+add1+sqrt
        x_delayed = self.a
        for i in range(sq.latency + add1.latency + sqrt.latency):
            d = Signal(self.fmt.width, name=f"x_delay_{i}")
            m.d.sync += d.eq(x_delayed)
            x_delayed = d
        m.d.comb += [add2.a.eq(x_delayed), add2.b.eq(sqrt.o)]
        # log(x + sqrt(x^2 + 1))
        m.d.comb += [log.a.eq(add2.o), self.o.eq(log.o)]

        return m
