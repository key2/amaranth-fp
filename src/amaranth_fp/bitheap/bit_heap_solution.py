"""Bit heap solution representation."""
from __future__ import annotations

__all__ = ["BitHeapSolution"]


class BitHeapSolution:
    """Stores a solved bit heap compression plan.

    Attributes
    ----------
    stages : list
        List of compression stages applied.
    total_latency : int
        Total latency of compression.
    """

    def __init__(self):
        self.stages: list = []
        self.total_latency = 0
