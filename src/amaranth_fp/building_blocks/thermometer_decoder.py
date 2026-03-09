"""Thermometer decoder."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["ThermometerDecoder"]


class ThermometerDecoder(PipelinedComponent):
    """Thermometer decoder.

    Parameters
    ----------
    width : int
        Number of input bits.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.x = Signal(width, name="x")
        self.o = Signal(1 << width, name="o")
        self.latency = 0

    def elaborate(self, platform) -> Module:
        m = Module()
        m.d.comb += self.o.eq((1 << self.x) - 1)
        return m
