"""Floating-point power x^y = exp(y * log(x)) (pipelined).

Uses FPLog + FPMul + FPExp as submodules.
"""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from .fp_log import FPLog
from .fp_mul import FPMul
from .fp_exp import FPExp

__all__ = ["FPPow"]


class FPPow(PipelinedComponent):
    """Pipelined floating-point power x^y.

    Computes x^y = exp(y * log(x)).

    Parameters
    ----------
    fmt : FPFormat
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.x = Signal(fmt.width, name="x")
        self.y = Signal(fmt.width, name="y")
        self.o = Signal(fmt.width, name="o")

        self._log = FPLog(fmt)
        self._mul = FPMul(fmt)
        self._exp = FPExp(fmt)
        self.latency = self._log.latency + self._mul.latency + self._exp.latency

    def elaborate(self, platform) -> Module:
        m = Module()

        log = self._log
        mul = self._mul
        exp = self._exp

        m.submodules.log = log
        m.submodules.mul = mul
        m.submodules.exp = exp

        # log(x)
        m.d.comb += log.a.eq(self.x)

        # Delay y to align with log output
        y_delayed = self.y
        for i in range(log.latency):
            y_next = Signal(self.fmt.width, name=f"y_d{i}")
            m.d.sync += y_next.eq(y_delayed)
            y_delayed = y_next

        # y * log(x)
        m.d.comb += [
            mul.a.eq(y_delayed),
            mul.b.eq(log.o),
        ]

        # exp(y * log(x))
        m.d.comb += exp.a.eq(mul.o)
        m.d.comb += self.o.eq(exp.o)

        return m
