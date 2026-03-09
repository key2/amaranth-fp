"""Shared test helpers and fixtures for amaranth-fp tests."""

import pytest

from amaranth_fp.format import FPFormat
from amaranth_fp.testing.sollya_reference import has_sollya


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "sollya: mark test as requiring Sollya")


def pytest_collection_modifyitems(config, items):
    """Skip Sollya-only tests when Sollya is not available."""
    if has_sollya():
        return
    skip_sollya = pytest.mark.skip(reason="Sollya not available, using mpmath fallback")
    for item in items:
        if "sollya" in item.keywords:
            item.add_marker(skip_sollya)


def encode_fp(fmt: FPFormat, sign: int, exponent: int, mantissa: int, exception: int = 0b01) -> int:
    """Encode a value in internal FloPoCo format.

    Layout (LSB to MSB): mantissa(wf) | exponent(we) | sign(1) | exception(2)
    """
    return (
        (exception << (1 + fmt.we + fmt.wf))
        | (sign << (fmt.we + fmt.wf))
        | (exponent << fmt.wf)
        | mantissa
    )


def fp_zero(fmt: FPFormat, sign: int = 0) -> int:
    """Encode +0 or -0."""
    return encode_fp(fmt, sign, 0, 0, 0b00)


def fp_inf(fmt: FPFormat, sign: int = 0) -> int:
    """Encode +inf or -inf."""
    return encode_fp(fmt, sign, 0, 0, 0b10)


def fp_nan(fmt: FPFormat) -> int:
    """Encode NaN."""
    return encode_fp(fmt, 0, 0, 0, 0b11)


def fp_normal(fmt: FPFormat, sign: int, exponent: int, mantissa: int) -> int:
    """Encode a normal number."""
    return encode_fp(fmt, sign, exponent, mantissa, 0b01)


def fp_one(fmt: FPFormat, sign: int = 0) -> int:
    """Encode 1.0 (or -1.0): exponent=bias, mantissa=0."""
    return fp_normal(fmt, sign, fmt.bias, 0)


def decode_exc(fmt: FPFormat, value: int) -> int:
    """Extract exception field from internal format."""
    return (value >> (1 + fmt.we + fmt.wf)) & 0b11


def decode_sign(fmt: FPFormat, value: int) -> int:
    """Extract sign field from internal format."""
    return (value >> (fmt.we + fmt.wf)) & 1


def decode_exp(fmt: FPFormat, value: int) -> int:
    """Extract exponent field from internal format."""
    return (value >> fmt.wf) & ((1 << fmt.we) - 1)


def decode_mant(fmt: FPFormat, value: int) -> int:
    """Extract mantissa field from internal format."""
    return value & ((1 << fmt.wf) - 1)


def encode_ieee(fmt: FPFormat, sign: int, exponent: int, mantissa: int) -> int:
    """Encode an IEEE 754 value. Layout (LSB to MSB): mantissa(wf) | exponent(we) | sign(1)."""
    return (sign << (fmt.we + fmt.wf)) | (exponent << fmt.wf) | mantissa
