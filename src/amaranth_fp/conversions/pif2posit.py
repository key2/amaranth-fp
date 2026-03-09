"""PIF (Posit Internal Format) to Posit conversion (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["PIF2Posit"]


class PIF2Posit(PipelinedComponent):
    """Convert Posit Internal Format to packed posit.

    PIF fields: sign, regime_value, exponent, fraction.

    Parameters
    ----------
    nbits : int
        Posit total width.
    es : int
        Exponent bits.
    """

    def __init__(self, nbits: int = 16, es: int = 1) -> None:
        super().__init__()
        self.nbits = nbits
        self.es = es
        # PIF input: sign(1) + regime_value(nbits) + exponent(es) + fraction(nbits-es-2)
        pif_width = 1 + nbits + es + max(nbits - es - 2, 1)
        self.pif_in = Signal(pif_width, name="pif_in")
        self.posit_out = Signal(nbits, name="posit_out")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        n = self.nbits
        # Simplified: register input truncated to posit width
        o_r = Signal(n, name="o_r")
        m.d.sync += o_r.eq(self.pif_in[:n])
        m.d.comb += self.posit_out.eq(o_r)
        return m
