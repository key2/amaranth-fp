"""Floating-point 10^x = 2^(x * log2(10)).

Wraps FPConstMult + FPExp2. Latency: 3 + FPExp2.latency.
"""
from __future__ import annotations

import math

from amaranth import *

from ...format import FPFormat
from ...pipelined import PipelinedComponent
from ...operators.fp_const_mult import FPConstMult
from .fp_exp2 import FPExp2

__all__ = ["FPExp10"]

_LOG2_10 = math.log2(10)


class FPExp10(PipelinedComponent):
    """Pipelined floating-point 10^x.

    Parameters
    ----------
    fmt : FPFormat
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")

        self._mult = FPConstMult(fmt, _LOG2_10)
        self._exp2 = FPExp2(fmt)
        self.latency = self._mult.latency + self._exp2.latency

    def elaborate(self, platform) -> Module:
        m = Module()
        m.submodules.mult = mult = self._mult
        m.submodules.exp2 = exp2 = self._exp2

        m.d.comb += [
            mult.a.eq(self.a),
            exp2.a.eq(mult.o),
            self.o.eq(exp2.o),
        ]
        return m
