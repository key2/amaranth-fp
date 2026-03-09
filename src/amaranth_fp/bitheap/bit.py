"""Bit abstraction for bit heap."""
from __future__ import annotations

from amaranth import *

__all__ = ["Bit"]


class Bit:
    """Represents a single bit in a bit heap column.

    Parameters
    ----------
    signal : Signal(1)
    column : int
        Column index (weight).
    uid : int
        Unique identifier.
    """

    def __init__(self, signal: Signal, column: int, uid: int = 0):
        self.signal = signal
        self.column = column
        self.uid = uid
