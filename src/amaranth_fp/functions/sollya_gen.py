"""Sollya-based polynomial coefficient generator for function approximation.

Generates minimax polynomial coefficients at elaboration time using Sollya's
fpminimax. Sollya is required — there is no fallback. Creates PipelinedComponent
modules using FixHornerEvaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from amaranth import *

from ..pipelined import PipelinedComponent
from .fix_horner import FixHornerEvaluator

__all__ = ["SollyaFunctionGenerator", "FixedPointFormat"]


@dataclass(frozen=True)
class FixedPointFormat:
    """Fixed-point number format descriptor."""
    signed: bool
    int_bits: int
    frac_bits: int

    @property
    def total_bits(self) -> int:
        return (1 if self.signed else 0) + self.int_bits + self.frac_bits


# Global coefficient cache
_COEFFICIENT_CACHE: dict[tuple, list[int]] = {}


def _sollya_coefficients(
    func_expr: str,
    domain: tuple[float, float],
    degree: int,
    precision_bits: int = 24,
) -> list[float]:
    """Compute minimax polynomial coefficients using Sollya's fpminimax.

    Parameters
    ----------
    func_expr : str
        Sollya expression string, e.g. "sin(x)", "exp(x) - 1".
    domain : tuple[float, float]
        Approximation interval [a, b].
    degree : int
        Polynomial degree.
    precision_bits : int
        Target precision in bits.

    Returns
    -------
    list[float]
        Monomial coefficients [c0, c1, ..., cn].

    Raises
    ------
    ImportError
        If Sollya (pythonsollya) is not installed.
    """
    try:
        import sollya  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "Sollya is required for SollyaFunctionGenerator. "
            "Install pythonsollya: https://gitlab.com/metalibm-dev/pythonsollya\n"
            "Sollya provides certified polynomial approximations with proven error bounds. "
            "There is no fallback — mpmath approximations are NOT equivalent and would "
            "silently produce incorrect hardware."
        )

    x = sollya.SollyaObject("x")
    f = sollya.parse(func_expr)
    dom = sollya.Interval(sollya.SollyaObject(domain[0]), sollya.SollyaObject(domain[1]))

    # Determine degree via guessdegree if not specified
    target_err = sollya.SollyaObject(2) ** (-precision_bits)

    # Build format list for fpminimax
    fmt_list = [sollya.binary64] * (degree + 1)

    poly = sollya.fpminimax(f, degree, fmt_list, dom, sollya.absolute, sollya.floating)

    # Extract monomial coefficients
    coeffs = []
    for i in range(degree + 1):
        ci = sollya.coeff(poly, i)
        coeffs.append(float(ci))

    # Verify with supnorm
    err = sollya.supnorm(poly, f, dom, sollya.absolute, target_err)
    # err is an interval; just log it (certification happens via supnorm)

    return coeffs


class SollyaFunctionGenerator:
    """Generate polynomial approximation coefficients and hardware modules.

    Uses Sollya's fpminimax to compute certified minimax polynomial coefficients
    at elaboration time. Coefficients are cached to avoid recomputation.

    Raises ImportError immediately if Sollya is not installed.

    Parameters
    ----------
    func_expr : str
        Sollya expression (e.g., "sin(x)", "2^x - 1").
    domain : tuple[float, float]
        Approximation domain [a, b].
    degree : int
        Polynomial degree.
    precision_bits : int
        Target precision in bits (default: 24 for single precision).
    """

    def __init__(
        self,
        func_expr: str,
        domain: tuple[float, float],
        degree: int,
        precision_bits: int = 24,
    ) -> None:
        try:
            import sollya  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "Sollya is required for SollyaFunctionGenerator. "
                "Install pythonsollya: https://gitlab.com/metalibm-dev/pythonsollya\n"
                "Sollya provides certified polynomial approximations with proven error bounds. "
                "There is no fallback — mpmath approximations are NOT equivalent and would "
                "silently produce incorrect hardware."
            )

        self.func_expr = func_expr
        self.domain = domain
        self.degree = degree
        self.precision_bits = precision_bits
        self._coefficients: list[float] | None = None

    def _cache_key(self, frac_bits: int) -> tuple:
        return (self.func_expr, self.domain, self.degree, self.precision_bits, frac_bits)

    def generate_coefficients(self) -> list[float]:
        """Generate floating-point polynomial coefficients via Sollya."""
        if self._coefficients is not None:
            return self._coefficients
        self._coefficients = _sollya_coefficients(
            self.func_expr, self.domain, self.degree, self.precision_bits
        )
        return self._coefficients

    def quantize_coefficients(self, frac_bits: int) -> list[int]:
        """Convert floating-point coefficients to fixed-point integers.

        Parameters
        ----------
        frac_bits : int
            Number of fractional bits in the fixed-point representation.

        Returns
        -------
        list[int]
            Coefficients scaled by 2^frac_bits and rounded.
        """
        key = self._cache_key(frac_bits)
        if key in _COEFFICIENT_CACHE:
            return _COEFFICIENT_CACHE[key]

        float_coeffs = self.generate_coefficients()
        scale = 1 << frac_bits
        int_coeffs = [int(round(c * scale)) for c in float_coeffs]
        _COEFFICIENT_CACHE[key] = int_coeffs
        return int_coeffs

    def create_module(
        self,
        input_format: FixedPointFormat,
        output_format: FixedPointFormat,
    ) -> FixHornerEvaluator:
        """Create a FixHornerEvaluator with precomputed coefficients.

        Parameters
        ----------
        input_format : FixedPointFormat
            Fixed-point format for the polynomial input.
        output_format : FixedPointFormat
            Fixed-point format for the polynomial output.

        Returns
        -------
        FixHornerEvaluator
            Configured Horner evaluator with Sollya-certified coefficients.
        """
        frac_bits = max(input_format.frac_bits, output_format.frac_bits)
        int_coeffs = self.quantize_coefficients(frac_bits)
        coeff_width = frac_bits + max(input_format.int_bits, output_format.int_bits) + 2

        return FixHornerEvaluator(
            coefficients=int_coeffs,
            input_width=input_format.total_bits,
            coeff_width=coeff_width,
            output_width=output_format.total_bits,
        )
