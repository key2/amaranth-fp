"""Floating-point log10(x) = log2(x) * log10(2).

Wraps FPLog2 + FPConstMult. Latency: FPLog2.latency + 3.
"""
from __future__ import annotations

import math

from amaranth import *

from ...format import FPFormat
from ...pipelined import PipelinedComponent
from ...operators.fp_const_mult import FPConstMult
from .fp_log2 import FPLog2

__all__ = ["FPLog10"]

_LOG10_2 = math.log10(2)


class FPLog10(PipelinedComponent):
    """Pipelined floating-point log10(x).

    Parameters
    ----------
    fmt : FPFormat
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")

        self._log2 = FPLog2(fmt)
        self._mult = FPConstMult(fmt, _LOG10_2)
        self.latency = self._log2.latency + self._mult.latency

    def elaborate(self, platform) -> Module:
        m = Module()
        m.submodules.log2 = log2 = self._log2
        m.submodules.mult = mult = self._mult

        m.d.comb += [
            log2.a.eq(self.a),
            mult.a.eq(log2.o),
            self.o.eq(mult.o),
        ]
        return m
