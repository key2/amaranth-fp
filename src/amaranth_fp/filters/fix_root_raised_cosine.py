"""Root Raised Cosine filter (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixRootRaisedCosine"]


class FixRootRaisedCosine(PipelinedComponent):
    """Fixed-point Root Raised Cosine FIR filter.

    Parameters
    ----------
    width : int
        Sample width.
    n_taps : int
        Number of filter taps.
    rolloff : float
        Roll-off factor (0..1).
    """

    def __init__(self, width: int, n_taps: int = 11, rolloff: float = 0.35) -> None:
        super().__init__()
        self.width = width
        self.n_taps = n_taps
        self.rolloff = rolloff
        self.x = Signal(width, name="x")
        self.y = Signal(width, name="y")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        # Placeholder: passthrough with register
        o_r = Signal(self.width, name="o_r")
        m.d.sync += o_r.eq(self.x)
        m.d.comb += self.y.eq(o_r)
        return m
