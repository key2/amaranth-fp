"""Floating-point negation (pipelined, 1 stage).

Flips the sign bit. Operates on the internal FloPoCo format:
    [exception(2) | sign(1) | exponent(we) | mantissa(wf)]
"""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent

__all__ = ["FPNeg"]


class FPNeg(PipelinedComponent):
    """Pipelined floating-point negation: o = -a (1-cycle latency).

    NaN stays NaN (sign is irrelevant). Zero sign is flipped.

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
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we = fmt.we
        wf = fmt.wf
        sign_pos = wf + we

        # ==================================================================
        # Stage 0: flip sign bit
        # ==================================================================
        neg_val = Signal(fmt.width, name="neg_val")
        m.d.comb += [
            neg_val[:sign_pos].eq(self.a[:sign_pos]),
            neg_val[sign_pos].eq(~self.a[sign_pos]),
            neg_val[sign_pos + 1:].eq(self.a[sign_pos + 1:]),
        ]

        # ── Stage 0 → 1 pipeline register (output) ──
        o_r1 = Signal(fmt.width, name="o_r1")
        m.d.sync += o_r1.eq(neg_val)
        m.d.comb += self.o.eq(o_r1)

        return m
