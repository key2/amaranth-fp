"""Posit format descriptor."""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PositFormat"]


@dataclass(frozen=True)
class PositFormat:
    """Posit number format descriptor.

    Parameters
    ----------
    n : int
        Total bit width.
    es : int
        Exponent field size.
    """

    n: int
    es: int

    def __post_init__(self) -> None:
        if self.n < 3:
            raise ValueError(f"Posit width must be >= 3, got {self.n}")
        if self.es < 0:
            raise ValueError(f"Exponent size must be >= 0, got {self.es}")

    @property
    def useed(self) -> int:
        """useed = 2^(2^es)."""
        return 1 << (1 << self.es)

    @property
    def max_value(self) -> float:
        """Maximum representable positive value."""
        return float(self.useed ** (self.n - 2))

    @property
    def min_positive(self) -> float:
        """Minimum positive value."""
        return float(self.useed ** (-(self.n - 2)))
