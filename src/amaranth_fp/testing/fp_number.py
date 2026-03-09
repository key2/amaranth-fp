"""FloPoCo-format FP number encoding/decoding utility."""
from __future__ import annotations

import struct

__all__ = ["FPNumber"]


class FPNumber:
    """Encode/decode FloPoCo internal FP format.

    FloPoCo format: [exception(2)] [sign(1)] [exponent(we)] [fraction(wf)]

    Parameters
    ----------
    we : int
    wf : int
    """

    def __init__(self, we: int = 8, wf: int = 23) -> None:
        self.we = we
        self.wf = wf
        self.width = 2 + 1 + we + wf

    def encode(self, value: float) -> int:
        """Encode a Python float into FloPoCo FP bit pattern."""
        if value != value:  # NaN
            return 3 << (1 + self.we + self.wf)
        if value == 0.0:
            return 0
        if value == float("inf"):
            return (2 << (1 + self.we + self.wf))
        if value == float("-inf"):
            return (2 << (1 + self.we + self.wf)) | (1 << (self.we + self.wf))

        sign = 1 if value < 0 else 0
        value = abs(value)

        bias = (1 << (self.we - 1)) - 1
        import math
        exp = int(math.floor(math.log2(value)))
        mantissa = value / (2.0 ** exp) - 1.0
        frac = int(round(mantissa * (1 << self.wf)))

        exp_biased = exp + bias
        if exp_biased < 0:
            return 0  # underflow to zero
        if exp_biased >= (1 << self.we):
            return (2 << (1 + self.we + self.wf)) | (sign << (self.we + self.wf))

        exc = 1  # normal
        return (exc << (1 + self.we + self.wf)) | (sign << (self.we + self.wf)) | (exp_biased << self.wf) | frac

    def decode(self, bits: int) -> float:
        """Decode a FloPoCo FP bit pattern to Python float."""
        exc = (bits >> (1 + self.we + self.wf)) & 3
        if exc == 0:
            return 0.0
        if exc == 2:
            sign = (bits >> (self.we + self.wf)) & 1
            return float("-inf") if sign else float("inf")
        if exc == 3:
            return float("nan")

        sign = (bits >> (self.we + self.wf)) & 1
        exp_biased = (bits >> self.wf) & ((1 << self.we) - 1)
        frac = bits & ((1 << self.wf) - 1)

        bias = (1 << (self.we - 1)) - 1
        exp = exp_biased - bias
        mantissa = 1.0 + frac / (1 << self.wf)
        result = mantissa * (2.0 ** exp)
        return -result if sign else result
