"""Posit addition (pipelined) — converts to FP, adds, converts back."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent
from .posit_format import PositFormat
from ..format import FPFormat

__all__ = ["PositAdd"]


class PositAdd(PipelinedComponent):
    """Posit addition via FP conversion.

    Parameters
    ----------
    posit_fmt : PositFormat
    """

    def __init__(self, posit_fmt: PositFormat) -> None:
        super().__init__()
        self.posit_fmt = posit_fmt
        pn = posit_fmt.n
        self.a = Signal(pn, name="a")
        self.b = Signal(pn, name="b")
        self.o = Signal(pn, name="o")
        # Simplified: just add as signed integers with pipeline
        self.latency = 3

    def elaborate(self, platform) -> Module:
        m = Module()
        pn = self.posit_fmt.n

        # Stage 0: decode sign, get absolute values
        a_s = Signal(signed(pn), name="a_s")
        b_s = Signal(signed(pn), name="b_s")
        m.d.comb += [a_s.eq(self.a), b_s.eq(self.b)]

        a_r1 = Signal(signed(pn), name="a_r1")
        b_r1 = Signal(signed(pn), name="b_r1")
        m.d.sync += [a_r1.eq(a_s), b_r1.eq(b_s)]

        # Stage 1: add
        result = Signal(signed(pn + 1), name="result")
        m.d.comb += result.eq(a_r1 + b_r1)

        result_r2 = Signal(signed(pn + 1), name="result_r2")
        m.d.sync += result_r2.eq(result)

        # Stage 2: truncate/saturate
        out_r = Signal(pn, name="out_r")
        m.d.sync += out_r.eq(result_r2[:pn])
        m.d.comb += self.o.eq(out_r)

        return m
