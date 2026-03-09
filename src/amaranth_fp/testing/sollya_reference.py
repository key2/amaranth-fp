"""Golden reference model using Sollya/mpmath for mathematically correct FP results.

Tries to import PythonSollya first, falls back to mpmath for arbitrary-precision
computation. Provides correctly-rounded results for all basic FP operations.
"""
from __future__ import annotations

import math
import struct
from typing import Tuple

from amaranth_fp.format import FPFormat

# --- Backend detection --------------------------------------------------------

_has_sollya = False
_has_mpmath = False

try:
    import sollya  # type: ignore[import-untyped]
    # Verify it's the real PythonSollya, not a namespace package shadow
    if hasattr(sollya, 'SollyaObject'):
        _has_sollya = True
except ImportError:
    pass

try:
    import mpmath
    _has_mpmath = True
except ImportError:
    pass

__all__ = ["SollyaReference", "has_sollya", "has_mpmath"]


def has_sollya() -> bool:
    """Return True if PythonSollya is available."""
    return _has_sollya


def has_mpmath() -> bool:
    """Return True if mpmath is available."""
    return _has_mpmath


# Exception codes for FloPoCo internal format
EXC_ZERO = 0b00
EXC_NORMAL = 0b01
EXC_INF = 0b10
EXC_NAN = 0b11


class SollyaReference:
    """Golden reference model using Sollya for mathematically correct FP results.

    Falls back to mpmath when Sollya is not installed. Computes correctly-rounded
    results and encodes them in FloPoCo internal format.

    Parameters
    ----------
    fmt : FPFormat
        Target floating-point format.
    """

    def __init__(self, fmt: FPFormat) -> None:
        self.fmt = fmt
        self.has_sollya = _has_sollya

        if _has_mpmath:
            # Use enough precision for correct rounding in any format
            mpmath.mp.dps = max(100, fmt.wf * 2)

    # --- Internal format encoding/decoding ------------------------------------

    def float_to_internal(self, value: float) -> int:
        """Convert a Python float to FloPoCo internal format encoding.

        Returns an integer with layout:
        [exception(2)] [sign(1)] [exponent(we)] [fraction(wf)]
        """
        fmt = self.fmt

        # NaN
        if math.isnan(value):
            return EXC_NAN << (1 + fmt.we + fmt.wf)

        # Zero (preserve sign)
        if value == 0.0:
            sign = 1 if math.copysign(1.0, value) < 0 else 0
            return (EXC_ZERO << (1 + fmt.we + fmt.wf)) | (sign << (fmt.we + fmt.wf))

        # Infinity
        if math.isinf(value):
            sign = 1 if value < 0 else 0
            return (EXC_INF << (1 + fmt.we + fmt.wf)) | (sign << (fmt.we + fmt.wf))

        sign = 1 if value < 0 else 0
        fabs = abs(value)

        # Compute exponent and mantissa
        exp_unbiased = math.floor(math.log2(fabs))

        # Clamp for denormals
        if exp_unbiased < fmt.emin:
            # Denormal: encode as zero in FloPoCo (FloPoCo doesn't support denormals natively)
            return (EXC_ZERO << (1 + fmt.we + fmt.wf)) | (sign << (fmt.we + fmt.wf))

        # Overflow check
        max_exp = fmt.emax
        if exp_unbiased > max_exp:
            return (EXC_INF << (1 + fmt.we + fmt.wf)) | (sign << (fmt.we + fmt.wf))

        # Normal number: significand = 1.fraction
        significand = fabs / (2.0 ** exp_unbiased)
        frac = significand - 1.0
        frac_bits = int(round(frac * (1 << fmt.wf)))

        # Handle rounding overflow (e.g., frac rounds up to 1.0)
        if frac_bits >= (1 << fmt.wf):
            frac_bits = 0
            exp_unbiased += 1
            if exp_unbiased > max_exp:
                return (EXC_INF << (1 + fmt.we + fmt.wf)) | (sign << (fmt.we + fmt.wf))

        exp_biased = exp_unbiased + fmt.bias

        return (
            (EXC_NORMAL << (1 + fmt.we + fmt.wf))
            | (sign << (fmt.we + fmt.wf))
            | (exp_biased << fmt.wf)
            | frac_bits
        )

    def internal_to_float(self, encoded: int) -> float:
        """Convert FloPoCo internal format encoding to Python float."""
        fmt = self.fmt
        exc = (encoded >> (1 + fmt.we + fmt.wf)) & 0b11
        sign = (encoded >> (fmt.we + fmt.wf)) & 1

        if exc == EXC_ZERO:
            return -0.0 if sign else 0.0
        if exc == EXC_INF:
            return float("-inf") if sign else float("inf")
        if exc == EXC_NAN:
            return float("nan")

        exp_biased = (encoded >> fmt.wf) & ((1 << fmt.we) - 1)
        frac = encoded & ((1 << fmt.wf) - 1)

        exp_unbiased = exp_biased - fmt.bias
        significand = 1.0 + frac / (1 << fmt.wf)
        result = significand * (2.0 ** exp_unbiased)
        return -result if sign else result

    def decode_fields(self, encoded: int) -> Tuple[int, int, int, int]:
        """Decode internal format into (exception, sign, exponent, mantissa)."""
        fmt = self.fmt
        exc = (encoded >> (1 + fmt.we + fmt.wf)) & 0b11
        sign = (encoded >> (fmt.we + fmt.wf)) & 1
        exp = (encoded >> fmt.wf) & ((1 << fmt.we) - 1)
        mant = encoded & ((1 << fmt.wf) - 1)
        return (exc, sign, exp, mant)

    # --- Correctly-rounded operations -----------------------------------------

    def _round_to_format(self, value: float | object) -> float:
        """Round an arbitrary-precision value to the target FP format.

        Uses Sollya if available, otherwise mpmath with high precision
        then converts to the target format width.
        """
        if _has_sollya:
            # Use Sollya's round() for provably correct rounding
            return self._sollya_round(value)
        elif _has_mpmath:
            return self._mpmath_round(value)
        else:
            # Last resort: Python float (only correct for double precision)
            return float(value)

    def _sollya_round(self, value: object) -> float:
        """Round using Sollya (provably correct)."""
        fmt = self.fmt
        # PythonSollya uses binary16/binary32/binary64 naming
        if fmt.we == 5 and fmt.wf == 10:
            sol_fmt = sollya.binary16
        elif fmt.we == 8 and fmt.wf == 23:
            sol_fmt = sollya.binary32
        elif fmt.we == 11 and fmt.wf == 52:
            sol_fmt = sollya.binary64
        else:
            # Custom format: fall back to mpmath-based rounding
            return self._mpmath_round(float(value))
        result = sollya.round(value, sol_fmt, sollya.RN)
        return float(result)

    def _mpmath_round(self, value: object) -> float:
        """Round using mpmath (high precision, then round to target format).

        For standard IEEE formats, we use struct.pack/unpack for exact rounding.
        For custom formats, we simulate the rounding manually.
        """
        fmt = self.fmt
        # Convert to Python float first (this handles mpmath types)
        fval = float(value)

        if math.isnan(fval) or math.isinf(fval) or fval == 0.0:
            return fval

        # For half precision, manually round
        if fmt.we == 5 and fmt.wf == 10:
            return self._round_to_half(fval)
        elif fmt.we == 8 and fmt.wf == 23:
            # Single precision: use struct for exact rounding
            return struct.unpack('f', struct.pack('f', fval))[0]
        elif fmt.we == 11 and fmt.wf == 52:
            # Double precision: Python float is already double
            return fval
        else:
            return self._round_to_custom(fval)

    def _round_to_half(self, value: float) -> float:
        """Round a float to IEEE 754 half precision (binary16)."""
        # Use struct with half-precision format (Python 3.6+)
        try:
            return struct.unpack('e', struct.pack('e', value))[0]
        except (struct.error, OverflowError):
            if abs(value) > 65504:
                return math.copysign(float('inf'), value)
            return 0.0

    def _round_to_custom(self, value: float) -> float:
        """Round to a custom FP format using manual rounding (round-to-nearest-even)."""
        fmt = self.fmt
        if value == 0.0:
            return value

        sign = -1 if value < 0 else 1
        fabs = abs(value)

        exp = math.floor(math.log2(fabs))

        # Clamp to format range
        if exp > fmt.emax:
            return sign * float('inf')
        if exp < fmt.emin:
            return sign * 0.0  # Flush denormals to zero

        # Compute the quantum (ULP at this exponent)
        ulp = 2.0 ** (exp - fmt.wf)

        # Round to nearest even
        scaled = fabs / ulp
        rounded = round(scaled)  # Python's round does banker's rounding
        result = rounded * ulp

        # Check for overflow after rounding
        max_normal = (2.0 - 2.0 ** (-fmt.wf)) * (2.0 ** fmt.emax)
        if result > max_normal:
            return sign * float('inf')

        return sign * result

    def _compute_high_prec(self, op: str, a: float, b: float = 0.0, c: float = 0.0) -> float:
        """Compute an operation in high precision, then round to target format."""
        if _has_sollya:
            return self._compute_sollya(op, a, b, c)
        elif _has_mpmath:
            return self._compute_mpmath(op, a, b, c)
        else:
            raise RuntimeError("Neither Sollya nor mpmath is available")

    def _compute_sollya(self, op: str, a: float, b: float, c: float) -> float:
        """Compute using Sollya."""
        sa = sollya.SollyaObject(a)
        sb = sollya.SollyaObject(b)
        sc = sollya.SollyaObject(c)

        if op == "add":
            expr = sa + sb
        elif op == "mul":
            expr = sa * sb
        elif op == "div":
            expr = sa / sb
        elif op == "sqrt":
            expr = sollya.sqrt(sa)
        elif op == "fma":
            expr = sa * sb + sc
        elif op == "exp":
            expr = sollya.exp(sa)
        elif op == "log":
            expr = sollya.log(sa)
        else:
            raise ValueError(f"Unknown operation: {op}")

        return self._sollya_round(expr)

    def _compute_mpmath(self, op: str, a: float, b: float, c: float) -> float:
        """Compute using mpmath with high precision, then round."""
        ma = mpmath.mpf(a)
        mb = mpmath.mpf(b)
        mc = mpmath.mpf(c)

        if op == "add":
            result = ma + mb
        elif op == "mul":
            result = ma * mb
        elif op == "div":
            if b == 0.0:
                if a == 0.0:
                    return float('nan')
                return math.copysign(float('inf'), a * b) if b == 0.0 else float('nan')
            result = ma / mb
        elif op == "sqrt":
            if a < 0:
                return float('nan')
            if a == 0.0:
                return a  # preserve sign of zero
            result = mpmath.sqrt(ma)
        elif op == "fma":
            # FMA: a*b + c with single rounding
            result = ma * mb + mc
        elif op == "exp":
            result = mpmath.exp(ma)
        elif op == "log":
            if a <= 0:
                if a == 0.0:
                    return float('-inf')
                return float('nan')
            result = mpmath.log(ma)
        else:
            raise ValueError(f"Unknown operation: {op}")

        return self._round_to_format(result)

    def _handle_special_binary(self, op: str, a: float, b: float) -> float | None:
        """Handle IEEE 754 special cases for binary operations. Returns None if not special."""
        a_nan = math.isnan(a)
        b_nan = math.isnan(b)
        a_inf = math.isinf(a)
        b_inf = math.isinf(b)

        if a_nan or b_nan:
            return float('nan')

        if op == "add":
            if a_inf and b_inf:
                if math.copysign(1, a) != math.copysign(1, b):
                    return float('nan')  # inf + (-inf)
                return a
            if a_inf:
                return a
            if b_inf:
                return b
        elif op == "mul":
            if a_inf:
                if b == 0.0:
                    return float('nan')
                return math.copysign(float('inf'), math.copysign(1, a) * math.copysign(1, b))
            if b_inf:
                if a == 0.0:
                    return float('nan')
                return math.copysign(float('inf'), math.copysign(1, a) * math.copysign(1, b))
        elif op == "div":
            if a_inf and b_inf:
                return float('nan')
            if a_inf:
                return math.copysign(float('inf'), math.copysign(1, a) * math.copysign(1, b))
            if b_inf:
                return math.copysign(0.0, math.copysign(1, a) * math.copysign(1, b))
            if b == 0.0:
                if a == 0.0:
                    return float('nan')
                return math.copysign(float('inf'), math.copysign(1, a) * math.copysign(1, b))

        return None  # not a special case

    # --- Public FP operations -------------------------------------------------

    def fp_add(self, a: float, b: float) -> Tuple[int, int, int, int]:
        """Compute a+b with correct rounding, return (exception, sign, exponent, mantissa)."""
        special = self._handle_special_binary("add", a, b)
        if special is not None:
            return self.decode_fields(self.float_to_internal(special))
        result = self._compute_high_prec("add", a, b)
        return self.decode_fields(self.float_to_internal(result))

    def fp_mul(self, a: float, b: float) -> Tuple[int, int, int, int]:
        """Compute a*b with correct rounding."""
        special = self._handle_special_binary("mul", a, b)
        if special is not None:
            return self.decode_fields(self.float_to_internal(special))
        result = self._compute_high_prec("mul", a, b)
        return self.decode_fields(self.float_to_internal(result))

    def fp_div(self, a: float, b: float) -> Tuple[int, int, int, int]:
        """Compute a/b with correct rounding."""
        special = self._handle_special_binary("div", a, b)
        if special is not None:
            return self.decode_fields(self.float_to_internal(special))
        result = self._compute_high_prec("div", a, b)
        return self.decode_fields(self.float_to_internal(result))

    def fp_sqrt(self, a: float) -> Tuple[int, int, int, int]:
        """Compute sqrt(a) with correct rounding."""
        if math.isnan(a):
            return self.decode_fields(self.float_to_internal(float('nan')))
        if a < 0:
            return self.decode_fields(self.float_to_internal(float('nan')))
        if math.isinf(a):
            if a > 0:
                return self.decode_fields(self.float_to_internal(float('inf')))
            return self.decode_fields(self.float_to_internal(float('nan')))
        if a == 0.0:
            return self.decode_fields(self.float_to_internal(a))
        result = self._compute_high_prec("sqrt", a)
        return self.decode_fields(self.float_to_internal(result))

    def fp_fma(self, a: float, b: float, c: float) -> Tuple[int, int, int, int]:
        """Compute a*b+c with single rounding (fused multiply-add)."""
        if math.isnan(a) or math.isnan(b) or math.isnan(c):
            return self.decode_fields(self.float_to_internal(float('nan')))
        # inf * 0 + anything = NaN
        if (math.isinf(a) and b == 0.0) or (a == 0.0 and math.isinf(b)):
            return self.decode_fields(self.float_to_internal(float('nan')))
        if math.isinf(a) or math.isinf(b):
            product_sign = math.copysign(1, a) * math.copysign(1, b)
            product = math.copysign(float('inf'), product_sign)
            if math.isinf(c) and math.copysign(1, product) != math.copysign(1, c):
                return self.decode_fields(self.float_to_internal(float('nan')))
            return self.decode_fields(self.float_to_internal(product))
        if math.isinf(c):
            return self.decode_fields(self.float_to_internal(c))
        result = self._compute_high_prec("fma", a, b, c)
        return self.decode_fields(self.float_to_internal(result))

    def fp_exp(self, a: float) -> Tuple[int, int, int, int]:
        """Compute exp(a) with correct rounding."""
        if math.isnan(a):
            return self.decode_fields(self.float_to_internal(float('nan')))
        if math.isinf(a):
            if a > 0:
                return self.decode_fields(self.float_to_internal(float('inf')))
            return self.decode_fields(self.float_to_internal(0.0))
        result = self._compute_high_prec("exp", a)
        return self.decode_fields(self.float_to_internal(result))

    def fp_log(self, a: float) -> Tuple[int, int, int, int]:
        """Compute log(a) with correct rounding."""
        if math.isnan(a):
            return self.decode_fields(self.float_to_internal(float('nan')))
        if a < 0:
            return self.decode_fields(self.float_to_internal(float('nan')))
        if a == 0.0:
            return self.decode_fields(self.float_to_internal(float('-inf')))
        if math.isinf(a) and a > 0:
            return self.decode_fields(self.float_to_internal(float('inf')))
        result = self._compute_high_prec("log", a)
        return self.decode_fields(self.float_to_internal(result))
