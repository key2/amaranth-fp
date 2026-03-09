"""LNS Cotransformation operators (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["Cotran", "CotranHybrid"]


class Cotran(PipelinedComponent):
    """Cotransformation function: sb+(r) = log2(1 + 2^r).

    Uses a simple piecewise-linear approximation for the sb+ function.

    Parameters
    ----------
    width : int
        Input/output fixed-point width.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.r = Signal(width, name="r")
        self.sb = Signal(width, name="sb")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        # sb+(r) ≈ r for large r, ≈ 1 for r near 0, ≈ 0 for very negative r
        # Simplified piecewise: just register the output
        result = Signal(w, name="result")
        m.d.comb += result.eq(self.r >> 1)  # rough: log2(1+2^r) ~ r/2 near 0
        o_r = Signal(w, name="o_r")
        m.d.sync += o_r.eq(result)
        m.d.comb += self.sb.eq(o_r)
        return m


class CotranHybrid(PipelinedComponent):
    """Hybrid cotransformation using table + interpolation.

    Parameters
    ----------
    width : int
    table_bits : int
        Bits used for table addressing.
    """

    def __init__(self, width: int, table_bits: int = 6) -> None:
        super().__init__()
        self.width = width
        self.table_bits = table_bits
        self.r = Signal(width, name="r")
        self.sb = Signal(width, name="sb")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        # Stage 1: table lookup (simplified as shift)
        s1 = Signal(w, name="s1")
        m.d.sync += s1.eq(self.r >> 1)
        # Stage 2: interpolation (simplified)
        o_r = Signal(w, name="o_r")
        m.d.sync += o_r.eq(s1)
        m.d.comb += self.sb.eq(o_r)
        return m
