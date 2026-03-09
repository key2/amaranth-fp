"""Posit number test helper."""
from __future__ import annotations

__all__ = ["PositNumber"]


class PositNumber:
    """Helper for posit number encoding/decoding in tests.

    Parameters
    ----------
    n : int
        Posit width.
    es : int
        Exponent size.
    """

    def __init__(self, n: int = 8, es: int = 0):
        self.n = n
        self.es = es

    def encode(self, value: float) -> int:
        """Encode float to posit (simplified)."""
        if value == 0:
            return 0
        sign = 1 if value < 0 else 0
        v = abs(value)
        # Simplified: just return truncated int representation
        raw = int(v * (1 << (self.n - 2))) & ((1 << self.n) - 1)
        if sign:
            raw = ((1 << self.n) - raw) & ((1 << self.n) - 1)
        return raw

    def decode(self, bits: int) -> float:
        """Decode posit to float (simplified)."""
        if bits == 0:
            return 0.0
        if bits == (1 << (self.n - 1)):
            return float('nan')
        sign = bits >> (self.n - 1)
        if sign:
            bits = ((1 << self.n) - bits) & ((1 << self.n) - 1)
        return bits / (1 << (self.n - 2)) * (-1 if sign else 1)
