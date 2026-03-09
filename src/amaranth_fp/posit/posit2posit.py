"""Posit format conversion (Posit to Posit)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["Posit2Posit"]


class Posit2Posit(PipelinedComponent):
    """Posit format conversion (Posit to Posit).

    Parameters
    ----------
    width_in, es_in, width_out, es_out : int
    """

    def __init__(self, width_in: int, es_in: int, width_out: int, es_out: int) -> None:
        super().__init__()
        self.width_in = width_in
        self.es_in = es_in
        self.width_out = width_out
        self.es_out = es_out
        self.x = Signal(width_in, name="x")
        self.o = Signal(width_out, name="o")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        s1 = Signal(self.width_in, name="s1")
        m.d.sync += s1.eq(self.x)
        out = Signal(self.width_out, name="out")
        m.d.sync += out.eq(s1[:self.width_out])
        m.d.comb += self.o.eq(out)
        return m
