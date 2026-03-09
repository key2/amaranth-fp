"""IEEE 754 number encoding/decoding utility."""
from __future__ import annotations

import struct

__all__ = ["IEEENumber"]


class IEEENumber:
    """Encode/decode IEEE 754 floating-point numbers.

    Supports single (8,23) and double (11,52) precision,
    as well as arbitrary (we, wf) formats.

    Parameters
    ----------
    we : int
    wf : int
    """

    def __init__(self, we: int = 8, wf: int = 23) -> None:
        self.we = we
        self.wf = wf
        self.width = 1 + we + wf

    def encode(self, value: float) -> int:
        """Encode a Python float to IEEE bit pattern."""
        if self.we == 8 and self.wf == 23:
            return struct.unpack(">I", struct.pack(">f", value))[0]
        if self.we == 11 and self.wf == 52:
            return struct.unpack(">Q", struct.pack(">d", value))[0]
        # Generic encoding
        import math
        if value != value:  # NaN
            return ((1 << self.we) - 1) << self.wf | 1
        if value == 0.0:
            return 0
        sign = 0
        if value < 0:
            sign = 1
            value = -value
        if value == float("inf"):
            return (sign << (self.we + self.wf)) | (((1 << self.we) - 1) << self.wf)
        bias = (1 << (self.we - 1)) - 1
        exp = int(math.floor(math.log2(value)))
        frac = int(round((value / (2.0 ** exp) - 1.0) * (1 << self.wf)))
        exp_biased = exp + bias
        if exp_biased <= 0:
            return 0
        if exp_biased >= (1 << self.we) - 1:
            return (sign << (self.we + self.wf)) | (((1 << self.we) - 1) << self.wf)
        return (sign << (self.we + self.wf)) | (exp_biased << self.wf) | frac

    def decode(self, bits: int) -> float:
        """Decode an IEEE bit pattern to Python float."""
        if self.we == 8 and self.wf == 23:
            return struct.unpack(">f", struct.pack(">I", bits & 0xFFFFFFFF))[0]
        if self.we == 11 and self.wf == 52:
            return struct.unpack(">d", struct.pack(">Q", bits & 0xFFFFFFFFFFFFFFFF))[0]
        sign = (bits >> (self.we + self.wf)) & 1
        exp_biased = (bits >> self.wf) & ((1 << self.we) - 1)
        frac = bits & ((1 << self.wf) - 1)
        if exp_biased == 0 and frac == 0:
            return -0.0 if sign else 0.0
        if exp_biased == (1 << self.we) - 1:
            if frac:
                return float("nan")
            return float("-inf") if sign else float("inf")
        bias = (1 << (self.we - 1)) - 1
        exp = exp_biased - bias
        mantissa = 1.0 + frac / (1 << self.wf)
        result = mantissa * (2.0 ** exp)
        return -result if sign else result
