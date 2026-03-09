"""Floating-point format definitions for amaranth-fp.

Defines FPFormat (parameterized FP width) and layout helpers for both
IEEE 754 and FloPoCo internal representations.
"""
from __future__ import annotations

from dataclasses import dataclass

from amaranth.lib.data import StructLayout


@dataclass(frozen=True)
class FPFormat:
    """Floating-point format parameterized by exponent and fraction widths.

    Attributes:
        we: Exponent field width in bits.
        wf: Fraction (mantissa) field width in bits.
    """

    we: int
    wf: int

    def __post_init__(self) -> None:
        if self.we < 2:
            raise ValueError(f"Exponent width must be >= 2, got {self.we}")
        if self.wf < 1:
            raise ValueError(f"Fraction width must be >= 1, got {self.wf}")

    # --- Predefined formats ---------------------------------------------------

    @classmethod
    def half(cls) -> FPFormat:
        """IEEE 754 half-precision (binary16): 5-bit exponent, 10-bit fraction."""
        return cls(we=5, wf=10)

    @classmethod
    def single(cls) -> FPFormat:
        """IEEE 754 single-precision (binary32): 8-bit exponent, 23-bit fraction."""
        return cls(we=8, wf=23)

    @classmethod
    def double(cls) -> FPFormat:
        """IEEE 754 double-precision (binary64): 11-bit exponent, 52-bit fraction."""
        return cls(we=11, wf=52)

    @classmethod
    def custom(cls, we: int, wf: int) -> FPFormat:
        """Create a custom floating-point format."""
        return cls(we=we, wf=wf)

    # --- Derived properties ---------------------------------------------------

    @property
    def width(self) -> int:
        """Total bit width of the internal (FloPoCo) representation.

        Layout: 2 (exception) + 1 (sign) + we (exponent) + wf (fraction).
        """
        return 2 + 1 + self.we + self.wf

    @property
    def ieee_width(self) -> int:
        """Total bit width of the IEEE 754 representation.

        Layout: 1 (sign) + we (exponent) + wf (fraction).
        """
        return 1 + self.we + self.wf

    @property
    def bias(self) -> int:
        """Exponent bias: 2^(we-1) - 1."""
        return (1 << (self.we - 1)) - 1

    @property
    def emin(self) -> int:
        """Minimum exponent (for normal numbers): 1 - bias."""
        return 1 - self.bias

    @property
    def emax(self) -> int:
        """Maximum exponent (for normal numbers): bias."""
        return self.bias


def float_to_flopoco(val: float, we: int, wf: int, bias: int) -> int:
    """Encode a Python float as a FloPoCo internal format integer (elaboration-time helper).

    Layout: [exception(2)] [sign(1)] [exponent(we)] [fraction(wf)]
    """
    import math as _math
    if _math.isnan(val):
        return 0b11 << (1 + we + wf)
    if val == 0.0:
        sign = 1 if _math.copysign(1.0, val) < 0 else 0
        return (0b00 << (1 + we + wf)) | (sign << (we + wf))
    if _math.isinf(val):
        sign = 1 if val < 0 else 0
        return (0b10 << (1 + we + wf)) | (sign << (we + wf))
    sign = 1 if val < 0 else 0
    fabs = abs(val)
    e = _math.floor(_math.log2(fabs))
    sig = fabs / (2.0 ** e)
    frac = int(round((sig - 1.0) * (1 << wf)))
    if frac >= (1 << wf):
        frac = 0
        e += 1
    eb = e + bias
    if eb < 1:
        return (0b00 << (1 + we + wf)) | (sign << (we + wf))
    if eb >= (1 << we) - 1:
        return (0b10 << (1 + we + wf)) | (sign << (we + wf))
    return (0b01 << (1 + we + wf)) | (sign << (we + wf)) | (eb << wf) | frac


def ieee_layout(fmt: FPFormat) -> StructLayout:
    """Return an Amaranth StructLayout for IEEE 754 encoding.

    Fields (LSB to MSB): ``mantissa`` (wf), ``exponent`` (we), ``sign`` (1).
    """
    return StructLayout({
        "mantissa": fmt.wf,
        "exponent": fmt.we,
        "sign": 1,
    })


def internal_layout(fmt: FPFormat) -> StructLayout:
    """Return an Amaranth StructLayout for FloPoCo internal encoding.

    Fields (LSB to MSB): ``mantissa`` (wf), ``exponent`` (we),
    ``sign`` (1), ``exception`` (2).

    Exception encoding:
        - 00: zero
        - 01: normal
        - 10: infinity
        - 11: NaN
    """
    return StructLayout({
        "mantissa": fmt.wf,
        "exponent": fmt.we,
        "sign": 1,
        "exception": 2,
    })
