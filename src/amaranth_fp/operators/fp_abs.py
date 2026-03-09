"""Floating-point absolute value (pipelined, 1 stage).

Clears the sign bit. Operates on the internal FloPoCo format:
    [exception(2) | sign(1) | exponent(we) | mantissa(wf)]
"""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent

__all__ = ["FPAbs"]


class FPAbs(PipelinedComponent):
    """Pipelined floating-point absolute value: o = |a| (1-cycle latency).

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
        # Stage 0: clear sign bit
        # ==================================================================
        abs_val = Signal(fmt.width, name="abs_val")
        m.d.comb += [
            abs_val[:sign_pos].eq(self.a[:sign_pos]),
            abs_val[sign_pos].eq(0),
            abs_val[sign_pos + 1:].eq(self.a[sign_pos + 1:]),
        ]

        # ── Stage 0 → 1 pipeline register (output) ──
        o_r1 = Signal(fmt.width, name="o_r1")
        m.d.sync += o_r1.eq(abs_val)
        m.d.comb += self.o.eq(o_r1)

        return m
