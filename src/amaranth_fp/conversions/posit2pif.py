"""Posit to PIF (Posit Internal Format) conversion (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["Posit2PIF"]


class Posit2PIF(PipelinedComponent):
    """Convert packed posit to Posit Internal Format.

    Parameters
    ----------
    nbits : int
    es : int
    """

    def __init__(self, nbits: int = 16, es: int = 1) -> None:
        super().__init__()
        self.nbits = nbits
        self.es = es
        pif_width = 1 + nbits + es + max(nbits - es - 2, 1)
        self.posit_in = Signal(nbits, name="posit_in")
        self.pif_out = Signal(pif_width, name="pif_out")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        # Simplified: zero-extend posit into PIF
        o_r = Signal(self.pif_out.shape(), name="o_r")
        m.d.sync += o_r.eq(self.posit_in)
        m.d.comb += self.pif_out.eq(o_r)
        return m
