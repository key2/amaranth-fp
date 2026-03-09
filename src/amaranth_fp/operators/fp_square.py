"""Floating-point squarer (pipelined, delegates to FPMul).

Optimized squaring: connects both inputs of FPMul to the same operand.
Operates on the internal FloPoCo format:
    [exception(2) | sign(1) | exponent(we) | mantissa(wf)]
"""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from .fp_mul import FPMul

__all__ = ["FPSquare"]


class FPSquare(PipelinedComponent):
    """Pipelined floating-point squarer: o = a * a (inherits FPMul latency).

    Parameters
    ----------
    fmt : FPFormat
        Floating-point format (defines we, wf).

    Attributes
    ----------
    a : Signal(fmt.width), in
    o : Signal(fmt.width), out
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self._mul = FPMul(fmt)
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")
        self.latency = self._mul.latency

    def elaborate(self, platform) -> Module:
        m = Module()

        mul = self._mul
        m.submodules.mul = mul

        m.d.comb += [
            mul.a.eq(self.a),
            mul.b.eq(self.a),
            self.o.eq(mul.o),
        ]

        return m
