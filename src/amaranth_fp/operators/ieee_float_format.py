"""IEEE 754 float format utilities."""
from __future__ import annotations

__all__ = ["IEEEFloatFormat"]


class IEEEFloatFormat:
    """Describes an IEEE 754 floating-point format.

    Parameters
    ----------
    we : int
        Exponent width.
    wf : int
        Fraction width (without implicit bit).
    """

    def __init__(self, we: int = 8, wf: int = 23):
        self.we = we
        self.wf = wf

    @property
    def width(self) -> int:
        return 1 + self.we + self.wf

    @property
    def bias(self) -> int:
        return (1 << (self.we - 1)) - 1

    @classmethod
    def binary16(cls):
        return cls(5, 10)

    @classmethod
    def binary32(cls):
        return cls(8, 23)

    @classmethod
    def binary64(cls):
        return cls(11, 52)
