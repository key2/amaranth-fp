"""Floating-point constant division (pipelined, 3 stages).

Divides FP by compile-time integer constant using reciprocal multiplication.
"""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from .fp_const_mult import FPConstMult

__all__ = ["FPConstDiv"]


class FPConstDiv(PipelinedComponent):
    """Divide FP value by compile-time integer constant.

    Uses reciprocal multiplication: x/d = x * (1/d).

    Parameters
    ----------
    fmt : FPFormat
    divisor : int
    """

    def __init__(self, fmt: FPFormat, divisor: int) -> None:
        super().__init__()
        self.fmt = fmt
        self.divisor = divisor
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")
        self._mult = FPConstMult(fmt, 1.0 / divisor)
        self.latency = self._mult.latency

    def elaborate(self, platform) -> Module:
        m = Module()
        m.submodules.mult = self._mult
        m.d.comb += [
            self._mult.a.eq(self.a),
            self.o.eq(self._mult.o),
        ]
        return m
