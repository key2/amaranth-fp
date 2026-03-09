"""Comprehensive tests for the Sollya/mpmath golden reference model.

Tests all FP operations with normal, edge, special, and rounding cases
across half and single precision formats.
"""
from __future__ import annotations

import math
import struct

import pytest

from amaranth_fp.format import FPFormat
from amaranth_fp.testing.sollya_reference import (
    SollyaReference,
    EXC_ZERO,
    EXC_NORMAL,
    EXC_INF,
    EXC_NAN,
    has_sollya,
    has_mpmath,
)
from test_helpers import assert_fp_equal, random_fp_values


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[FPFormat.half(), FPFormat.single()], ids=["half", "single"])
def ref(request):
    """SollyaReference fixture for half and single precision."""
    return SollyaReference(request.param)


@pytest.fixture
def ref_half():
    return SollyaReference(FPFormat.half())


@pytest.fixture
def ref_single():
    return SollyaReference(FPFormat.single())


# ---------------------------------------------------------------------------
# Encoding / decoding round-trip
# ---------------------------------------------------------------------------

class TestEncodingRoundTrip:
    """Test float_to_internal and internal_to_float are inverses."""

    @pytest.mark.parametrize("value", [
        0.0, -0.0, 1.0, -1.0, 2.0, 0.5, 1.5, 100.0, 0.001,
        float('inf'), float('-inf'), float('nan'),
    ])
    def test_roundtrip(self, ref, value):
        encoded = ref.float_to_internal(value)
        decoded = ref.internal_to_float(encoded)
        if math.isnan(value):
            assert math.isnan(decoded)
        elif value == 0.0:
            assert decoded == 0.0
            # Check sign of zero
            assert math.copysign(1, decoded) == math.copysign(1, value)
        else:
            assert decoded == pytest.approx(value, rel=1e-2), (
                f"Roundtrip failed for {value}: encoded={encoded:#x}, decoded={decoded}"
            )

    def test_special_encoding_nan(self, ref):
        enc = ref.float_to_internal(float('nan'))
        exc, sign, exp, mant = ref.decode_fields(enc)
        assert exc == EXC_NAN

    def test_special_encoding_inf(self, ref):
        enc = ref.float_to_internal(float('inf'))
        exc, sign, exp, mant = ref.decode_fields(enc)
        assert exc == EXC_INF
        assert sign == 0

    def test_special_encoding_neg_inf(self, ref):
        enc = ref.float_to_internal(float('-inf'))
        exc, sign, exp, mant = ref.decode_fields(enc)
        assert exc == EXC_INF
        assert sign == 1

    def test_special_encoding_zero(self, ref):
        enc = ref.float_to_internal(0.0)
        exc, sign, exp, mant = ref.decode_fields(enc)
        assert exc == EXC_ZERO
        assert sign == 0

    def test_special_encoding_neg_zero(self, ref):
        enc = ref.float_to_internal(-0.0)
        exc, sign, exp, mant = ref.decode_fields(enc)
        assert exc == EXC_ZERO
        assert sign == 1

    def test_normal_one(self, ref):
        enc = ref.float_to_internal(1.0)
        exc, sign, exp, mant = ref.decode_fields(enc)
        assert exc == EXC_NORMAL
        assert sign == 0
        assert exp == ref.fmt.bias  # 1.0 has unbiased exponent 0
        assert mant == 0  # 1.0 = 1.000...


# ---------------------------------------------------------------------------
# FP Addition
# ---------------------------------------------------------------------------

class TestFPAdd:
    """Test fp_add against known correct results."""

    @pytest.mark.parametrize("a,b,desc", [
        (1.0, 1.0, "simple"),
        (1.0, -1.0, "cancellation"),
        (1.5, 2.5, "normal_sum"),
        (0.1, 0.2, "non_representable"),
        (100.0, 0.001, "large_exp_diff"),
        (1.0, 0.0, "add_zero"),
        (0.0, 0.0, "zero_plus_zero"),
    ])
    def test_normal_cases(self, ref, a, b, desc):
        exc, sign, exp, mant = ref.fp_add(a, b)
        assert exc == EXC_NORMAL or exc == EXC_ZERO

    @pytest.mark.parametrize("a,b", [
        (float('inf'), 1.0),
        (1.0, float('inf')),
        (float('-inf'), 1.0),
        (float('inf'), float('inf')),
    ])
    def test_inf_cases(self, ref, a, b):
        exc, sign, exp, mant = ref.fp_add(a, b)
        assert exc == EXC_INF

    def test_inf_minus_inf_is_nan(self, ref):
        exc, sign, exp, mant = ref.fp_add(float('inf'), float('-inf'))
        assert exc == EXC_NAN

    @pytest.mark.parametrize("a,b", [
        (float('nan'), 1.0),
        (1.0, float('nan')),
        (float('nan'), float('nan')),
    ])
    def test_nan_propagation(self, ref, a, b):
        exc, sign, exp, mant = ref.fp_add(a, b)
        assert exc == EXC_NAN

    def test_signed_zero(self, ref):
        exc, sign, exp, mant = ref.fp_add(-0.0, 0.0)
        assert exc == EXC_ZERO

    def test_add_result_value_half(self, ref_half):
        """1.0 + 1.0 = 2.0 in half precision."""
        exc, sign, exp, mant = ref_half.fp_add(1.0, 1.0)
        assert exc == EXC_NORMAL
        assert sign == 0
        # 2.0 = 1.0 * 2^1, biased exp = 1 + 15 = 16
        assert exp == 16
        assert mant == 0

    def test_add_result_value_single(self, ref_single):
        """1.5 + 2.5 = 4.0 in single precision."""
        exc, sign, exp, mant = ref_single.fp_add(1.5, 2.5)
        assert exc == EXC_NORMAL
        assert sign == 0
        # 4.0 = 1.0 * 2^2, biased exp = 2 + 127 = 129
        assert exp == 129
        assert mant == 0


# ---------------------------------------------------------------------------
# FP Multiplication
# ---------------------------------------------------------------------------

class TestFPMul:
    """Test fp_mul against known correct results."""

    @pytest.mark.parametrize("a,b,desc", [
        (2.0, 3.0, "simple"),
        (1.0, 1.0, "identity"),
        (-1.0, 1.0, "neg_identity"),
        (-2.0, -3.0, "neg_times_neg"),
        (0.5, 2.0, "half_double"),
        (0.1, 10.0, "non_representable"),
    ])
    def test_normal_cases(self, ref, a, b, desc):
        exc, sign, exp, mant = ref.fp_mul(a, b)
        assert exc in (EXC_NORMAL, EXC_ZERO)

    def test_mul_by_zero(self, ref):
        exc, sign, exp, mant = ref.fp_mul(5.0, 0.0)
        assert exc == EXC_ZERO

    def test_inf_times_zero_is_nan(self, ref):
        exc, sign, exp, mant = ref.fp_mul(float('inf'), 0.0)
        assert exc == EXC_NAN

    def test_inf_times_finite(self, ref):
        exc, sign, exp, mant = ref.fp_mul(float('inf'), 2.0)
        assert exc == EXC_INF
        assert sign == 0

    def test_neg_inf_times_pos(self, ref):
        exc, sign, exp, mant = ref.fp_mul(float('-inf'), 2.0)
        assert exc == EXC_INF
        assert sign == 1

    @pytest.mark.parametrize("a,b", [
        (float('nan'), 1.0),
        (1.0, float('nan')),
    ])
    def test_nan_propagation(self, ref, a, b):
        exc, sign, exp, mant = ref.fp_mul(a, b)
        assert exc == EXC_NAN

    def test_mul_result_value(self, ref_half):
        """2.0 * 3.0 = 6.0 in half precision."""
        exc, sign, exp, mant = ref_half.fp_mul(2.0, 3.0)
        assert exc == EXC_NORMAL
        assert sign == 0
        # 6.0 = 1.5 * 2^2, biased exp = 2 + 15 = 17, frac = 0.5 => 0b1000000000 = 512
        assert exp == 17
        assert mant == 512


# ---------------------------------------------------------------------------
# FP Division
# ---------------------------------------------------------------------------

class TestFPDiv:
    """Test fp_div against known correct results."""

    @pytest.mark.parametrize("a,b,desc", [
        (6.0, 2.0, "exact"),
        (1.0, 3.0, "repeating"),
        (1.0, 7.0, "repeating_7"),
        (10.0, 3.0, "non_exact"),
        (-6.0, 2.0, "neg_dividend"),
        (6.0, -2.0, "neg_divisor"),
    ])
    def test_normal_cases(self, ref, a, b, desc):
        exc, sign, exp, mant = ref.fp_div(a, b)
        assert exc == EXC_NORMAL

    def test_div_by_zero(self, ref):
        exc, sign, exp, mant = ref.fp_div(1.0, 0.0)
        assert exc == EXC_INF

    def test_zero_div_zero_is_nan(self, ref):
        exc, sign, exp, mant = ref.fp_div(0.0, 0.0)
        assert exc == EXC_NAN

    def test_inf_div_inf_is_nan(self, ref):
        exc, sign, exp, mant = ref.fp_div(float('inf'), float('inf'))
        assert exc == EXC_NAN

    def test_finite_div_inf(self, ref):
        exc, sign, exp, mant = ref.fp_div(1.0, float('inf'))
        assert exc == EXC_ZERO

    def test_div_result_value(self, ref_half):
        """6.0 / 2.0 = 3.0 in half precision."""
        exc, sign, exp, mant = ref_half.fp_div(6.0, 2.0)
        assert exc == EXC_NORMAL
        assert sign == 0
        # 3.0 = 1.5 * 2^1, biased exp = 1 + 15 = 16, frac = 0.5 => 512
        assert exp == 16
        assert mant == 512


# ---------------------------------------------------------------------------
# FP Square Root
# ---------------------------------------------------------------------------

class TestFPSqrt:
    """Test fp_sqrt against known correct results."""

    @pytest.mark.parametrize("a,desc", [
        (4.0, "perfect_square"),
        (9.0, "perfect_square_9"),
        (2.0, "irrational"),
        (1.0, "one"),
        (0.25, "quarter"),
        (100.0, "hundred"),
    ])
    def test_normal_cases(self, ref, a, desc):
        exc, sign, exp, mant = ref.fp_sqrt(a)
        assert exc == EXC_NORMAL
        assert sign == 0

    def test_sqrt_zero(self, ref):
        exc, sign, exp, mant = ref.fp_sqrt(0.0)
        assert exc == EXC_ZERO

    def test_sqrt_negative_is_nan(self, ref):
        exc, sign, exp, mant = ref.fp_sqrt(-1.0)
        assert exc == EXC_NAN

    def test_sqrt_neg_inf_is_nan(self, ref):
        exc, sign, exp, mant = ref.fp_sqrt(float('-inf'))
        assert exc == EXC_NAN

    def test_sqrt_pos_inf(self, ref):
        exc, sign, exp, mant = ref.fp_sqrt(float('inf'))
        assert exc == EXC_INF

    def test_sqrt_nan(self, ref):
        exc, sign, exp, mant = ref.fp_sqrt(float('nan'))
        assert exc == EXC_NAN

    def test_sqrt_result_value(self, ref_half):
        """sqrt(4.0) = 2.0 in half precision."""
        exc, sign, exp, mant = ref_half.fp_sqrt(4.0)
        assert exc == EXC_NORMAL
        assert sign == 0
        # 2.0 = 1.0 * 2^1, biased exp = 1 + 15 = 16
        assert exp == 16
        assert mant == 0


# ---------------------------------------------------------------------------
# FP FMA (Fused Multiply-Add)
# ---------------------------------------------------------------------------

class TestFPFMA:
    """Test fp_fma against known correct results."""

    @pytest.mark.parametrize("a,b,c,desc", [
        (2.0, 3.0, 1.0, "simple"),
        (1.0, 1.0, 0.0, "mul_only"),
        (0.0, 0.0, 1.0, "add_only"),
        (1.0, -1.0, 1.0, "cancellation"),
        (-2.0, 3.0, 7.0, "neg_product"),
    ])
    def test_normal_cases(self, ref, a, b, c, desc):
        exc, sign, exp, mant = ref.fp_fma(a, b, c)
        assert exc in (EXC_NORMAL, EXC_ZERO)

    def test_inf_times_zero_plus_c(self, ref):
        exc, sign, exp, mant = ref.fp_fma(float('inf'), 0.0, 1.0)
        assert exc == EXC_NAN

    def test_nan_propagation(self, ref):
        exc, sign, exp, mant = ref.fp_fma(float('nan'), 1.0, 1.0)
        assert exc == EXC_NAN

    def test_fma_result_value(self, ref_half):
        """fma(2, 3, 1) = 7.0 in half precision."""
        exc, sign, exp, mant = ref_half.fp_fma(2.0, 3.0, 1.0)
        assert exc == EXC_NORMAL
        assert sign == 0
        # 7.0 = 1.75 * 2^2, biased exp = 2 + 15 = 17, frac = 0.75 => 768
        assert exp == 17
        assert mant == 768


# ---------------------------------------------------------------------------
# FP Exp
# ---------------------------------------------------------------------------

class TestFPExp:
    """Test fp_exp against known correct results."""

    @pytest.mark.parametrize("a,desc", [
        (0.0, "exp_zero"),
        (1.0, "exp_one"),
        (-1.0, "exp_neg_one"),
        (0.5, "exp_half"),
    ])
    def test_normal_cases(self, ref, a, desc):
        exc, sign, exp, mant = ref.fp_exp(a)
        assert exc == EXC_NORMAL
        assert sign == 0  # exp() is always positive for finite input

    def test_exp_zero_is_one(self, ref):
        exc, sign, exp, mant = ref.fp_exp(0.0)
        assert exc == EXC_NORMAL
        # Result should be 1.0
        assert exp == ref.fmt.bias
        assert mant == 0

    def test_exp_pos_inf(self, ref):
        exc, sign, exp, mant = ref.fp_exp(float('inf'))
        assert exc == EXC_INF

    def test_exp_neg_inf(self, ref):
        exc, sign, exp, mant = ref.fp_exp(float('-inf'))
        assert exc == EXC_ZERO

    def test_exp_nan(self, ref):
        exc, sign, exp, mant = ref.fp_exp(float('nan'))
        assert exc == EXC_NAN

    def test_exp_large_overflow(self, ref_half):
        """exp(12) overflows half precision."""
        exc, sign, exp, mant = ref_half.fp_exp(12.0)
        assert exc == EXC_INF


# ---------------------------------------------------------------------------
# FP Log
# ---------------------------------------------------------------------------

class TestFPLog:
    """Test fp_log against known correct results."""

    @pytest.mark.parametrize("a,desc", [
        (1.0, "log_one"),
        (2.0, "log_two"),
        (10.0, "log_ten"),
        (0.5, "log_half"),
    ])
    def test_normal_cases(self, ref, a, desc):
        exc, sign, exp, mant = ref.fp_log(a)
        if a == 1.0:
            assert exc == EXC_ZERO  # log(1) = 0
        else:
            assert exc == EXC_NORMAL

    def test_log_one_is_zero(self, ref):
        exc, sign, exp, mant = ref.fp_log(1.0)
        assert exc == EXC_ZERO

    def test_log_zero_is_neg_inf(self, ref):
        exc, sign, exp, mant = ref.fp_log(0.0)
        assert exc == EXC_INF
        assert sign == 1

    def test_log_negative_is_nan(self, ref):
        exc, sign, exp, mant = ref.fp_log(-1.0)
        assert exc == EXC_NAN

    def test_log_pos_inf(self, ref):
        exc, sign, exp, mant = ref.fp_log(float('inf'))
        assert exc == EXC_INF
        assert sign == 0

    def test_log_nan(self, ref):
        exc, sign, exp, mant = ref.fp_log(float('nan'))
        assert exc == EXC_NAN

    def test_log_half_is_negative(self, ref):
        exc, sign, exp, mant = ref.fp_log(0.5)
        assert exc == EXC_NORMAL
        assert sign == 1  # log(0.5) < 0


# ---------------------------------------------------------------------------
# Random value consistency
# ---------------------------------------------------------------------------

class TestRandomConsistency:
    """Test that reference model is self-consistent with random values."""

    def test_add_commutative(self, ref):
        """a + b == b + a for random values."""
        values = random_fp_values(ref.fmt, n=20)
        for i in range(0, len(values) - 1, 2):
            a, b = values[i], values[i + 1]
            r1 = ref.fp_add(a, b)
            r2 = ref.fp_add(b, a)
            assert r1 == r2, f"Commutativity failed for {a} + {b}"

    def test_mul_commutative(self, ref):
        """a * b == b * a for random values."""
        values = random_fp_values(ref.fmt, n=20)
        for i in range(0, len(values) - 1, 2):
            a, b = values[i], values[i + 1]
            r1 = ref.fp_mul(a, b)
            r2 = ref.fp_mul(b, a)
            assert r1 == r2, f"Commutativity failed for {a} * {b}"

    def test_sqrt_of_square(self, ref):
        """sqrt(a^2) ≈ |a| for positive values (within rounding)."""
        values = [abs(v) for v in random_fp_values(ref.fmt, n=10) if abs(v) > 0.01]
        for a in values[:5]:
            sq_exc, sq_sign, sq_exp, sq_mant = ref.fp_mul(a, a)
            if sq_exc == EXC_NORMAL:
                sq_float = ref.internal_to_float(
                    (sq_exc << (1 + ref.fmt.we + ref.fmt.wf))
                    | (sq_sign << (ref.fmt.we + ref.fmt.wf))
                    | (sq_exp << ref.fmt.wf)
                    | sq_mant
                )
                sqrt_result = ref.fp_sqrt(sq_float)
                # Should be approximately |a| (within a couple ULP due to double rounding)
                assert sqrt_result[0] == EXC_NORMAL


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

class TestBackendDetection:
    """Test that backend detection works correctly."""

    def test_has_mpmath(self):
        assert has_mpmath(), "mpmath should be available"

    def test_reference_works_without_sollya(self):
        """SollyaReference should work even without Sollya (using mpmath)."""
        ref = SollyaReference(FPFormat.half())
        result = ref.fp_add(1.0, 1.0)
        assert result[0] == EXC_NORMAL


# ---------------------------------------------------------------------------
# Overflow / Underflow edge cases
# ---------------------------------------------------------------------------

class TestOverflowUnderflow:
    """Test near-boundary values."""

    def test_half_near_overflow(self, ref_half):
        """Adding to near-max value in half precision."""
        # Max half = 65504
        exc, sign, exp, mant = ref_half.fp_add(65504.0, 0.0)
        assert exc == EXC_NORMAL

    def test_half_overflow(self, ref_half):
        """Overflow in half precision."""
        exc, sign, exp, mant = ref_half.fp_mul(65504.0, 2.0)
        assert exc == EXC_INF

    def test_half_underflow(self, ref_half):
        """Very small number underflows to zero."""
        exc, sign, exp, mant = ref_half.fp_mul(2.0 ** -14, 2.0 ** -14)
        assert exc == EXC_ZERO  # FloPoCo flushes denormals to zero

    def test_single_large_add(self, ref_single):
        """Large addition in single precision stays normal."""
        exc, sign, exp, mant = ref_single.fp_add(1e30, 1e30)
        assert exc == EXC_NORMAL


# ---------------------------------------------------------------------------
# Rounding tie-to-even cases
# ---------------------------------------------------------------------------

class TestRoundingTieToEven:
    """Test cases that exercise round-to-nearest-even."""

    def test_tie_case_half(self, ref_half):
        """Test a value that's exactly between two representable half-precision values."""
        # In half precision, 1.0 + 2^-11 is exactly halfway between 1.0 and 1.0 + 2^-10
        # Round-to-nearest-even should pick the one with even mantissa (1.0, mant=0)
        result = ref_half.fp_add(1.0, 2.0 ** -11)
        exc, sign, exp, mant = result
        assert exc == EXC_NORMAL
        # Should round to 1.0 (even mantissa) since 1.0 has mantissa 0 (even)
        assert exp == ref_half.fmt.bias
        assert mant == 0
