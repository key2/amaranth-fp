"""Weighted bit representation for bit heap compression."""
from __future__ import annotations

from amaranth import *

__all__ = ["WeightedBit"]


class WeightedBit:
    """A single bit with an associated column weight in a bit heap.

    Parameters
    ----------
    signal : Signal(1)
        The 1-bit signal.
    weight : int
        Column weight (power of 2).
    cycle : int
        Pipeline stage when this bit is available.
    """

    def __init__(self, signal: Signal, weight: int, cycle: int = 0):
        self.signal = signal
        self.weight = weight
        self.cycle = cycle

    def __repr__(self):
        return f"WeightedBit({self.signal.name}, w={self.weight}, c={self.cycle})"
