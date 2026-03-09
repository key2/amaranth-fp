"""Floating-point subtraction wrapper.

Negates the sign of b, then delegates to FPAdd.
Operates on the internal FloPoCo format:
    [exception(2) | sign(1) | exponent(we) | mantissa(wf)]
"""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from .fp_add import FPAdd

__all__ = ["FPSub"]


class FPSub(Elaboratable):
    """Combinational floating-point subtractor: o = a - b.

    Parameters
    ----------
    fmt : FPFormat
        Floating-point format (defines we, wf).

    Attributes
    ----------
    a : Signal(fmt.width), in
    b : Signal(fmt.width), in
    o : Signal(fmt.width), out
    """

    def __init__(self, fmt: FPFormat) -> None:
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.b = Signal(fmt.width, name="b")
        self.o = Signal(fmt.width, name="o")
        # Compute latency without creating an unused elaboratable
        self._latency_add = FPAdd(fmt)
        self.latency = self._latency_add.latency

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we = fmt.we
        wf = fmt.wf

        adder = FPAdd(fmt)
        m.submodules.adder = adder

        # Negate b: flip sign bit (bit wf+we)
        b_negated = Signal(fmt.width, name="b_negated")
        sign_pos = wf + we
        m.d.comb += [
            b_negated[:sign_pos].eq(self.b[:sign_pos]),
            b_negated[sign_pos].eq(~self.b[sign_pos]),
            b_negated[sign_pos + 1:].eq(self.b[sign_pos + 1:]),
        ]

        m.d.comb += [
            adder.a.eq(self.a),
            adder.b.eq(b_negated),
            self.o.eq(adder.o),
        ]

        self.latency = adder.latency

        return m
