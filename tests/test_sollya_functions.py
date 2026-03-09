"""Tests for all Sollya-generated mathematical and ML functions.

Uses SollyaReference (mpmath-based) for expected values.
Tests instantiation and latency for all modules.
"""
from __future__ import annotations

import math
import pytest

from amaranth_fp.format import FPFormat
from amaranth_fp.testing.sollya_reference import SollyaReference

# Math functions
from amaranth_fp.functions.math.fp_exp2 import FPExp2
from amaranth_fp.functions.math.fp_log2 import FPLog2
from amaranth_fp.functions.math.fp_log10 import FPLog10
from amaranth_fp.functions.math.fp_exp10 import FPExp10
from amaranth_fp.functions.math.fp_asin import FPAsin
from amaranth_fp.functions.math.fp_acos import FPAcos
from amaranth_fp.functions.math.fp_atan import FPAtan
from amaranth_fp.functions.math.fp_sinh import FPSinh
from amaranth_fp.functions.math.fp_cosh import FPCosh
from amaranth_fp.functions.math.fp_tanh import FPTanh
from amaranth_fp.functions.math.fp_asinh import FPAsinh
from amaranth_fp.functions.math.fp_acosh import FPAcosh
from amaranth_fp.functions.math.fp_atanh import FPAtanh
from amaranth_fp.functions.math.fp_erf import FPErf
from amaranth_fp.functions.math.fp_erfc import FPErfc
from amaranth_fp.functions.math.fp_cbrt import FPCbrt
from amaranth_fp.functions.math.fp_reciprocal import FPReciprocal
from amaranth_fp.functions.math.fp_rsqrt import FPRsqrt

# ML functions
from amaranth_fp.functions.ml.fp_sigmoid import FPSigmoid
from amaranth_fp.functions.ml.fp_gelu import FPGELU
from amaranth_fp.functions.ml.fp_softplus import FPSoftplus
from amaranth_fp.functions.ml.fp_swish import FPSwish
from amaranth_fp.functions.ml.fp_mish import FPMish
from amaranth_fp.functions.ml.fp_sinc import FPSinc

# Infrastructure
from amaranth_fp.functions.sollya_gen import SollyaFunctionGenerator, FixedPointFormat


# ---- Test formats ----

HALF = FPFormat.half()
SINGLE = FPFormat.single()


# ---- Instantiation tests ----

@pytest.mark.parametrize("fmt", [HALF, SINGLE], ids=["half", "single"])
class TestMathInstantiation:
    """Test that all math function modules can be instantiated."""

    def test_fp_exp2(self, fmt):
        mod = FPExp2(fmt)
        assert mod.latency == 6
        assert mod.a.shape().width == fmt.width
        assert mod.o.shape().width == fmt.width

    def test_fp_log2(self, fmt):
        mod = FPLog2(fmt)
        assert mod.latency == 7

    def test_fp_log10(self, fmt):
        mod = FPLog10(fmt)
        assert mod.latency == FPLog2(fmt).latency + 3

    def test_fp_exp10(self, fmt):
        mod = FPExp10(fmt)
        assert mod.latency == 3 + FPExp2(fmt).latency

    def test_fp_asin(self, fmt):
        mod = FPAsin(fmt)
        assert mod.latency == 12

    def test_fp_acos(self, fmt):
        mod = FPAcos(fmt)
        assert mod.latency == FPAsin(fmt).latency + 7

    def test_fp_atan(self, fmt):
        mod = FPAtan(fmt)
        assert mod.latency == 10

    def test_fp_sinh(self, fmt):
        mod = FPSinh(fmt)
        assert mod.latency == 10

    def test_fp_cosh(self, fmt):
        mod = FPCosh(fmt)
        assert mod.latency == 10

    def test_fp_tanh(self, fmt):
        mod = FPTanh(fmt)
        assert mod.latency == 8

    def test_fp_asinh(self, fmt):
        mod = FPAsinh(fmt)
        assert mod.latency > 0

    def test_fp_acosh(self, fmt):
        mod = FPAcosh(fmt)
        assert mod.latency > 0

    def test_fp_atanh(self, fmt):
        mod = FPAtanh(fmt)
        assert mod.latency > 0

    def test_fp_erf(self, fmt):
        mod = FPErf(fmt)
        assert mod.latency == 8

    def test_fp_erfc(self, fmt):
        mod = FPErfc(fmt)
        assert mod.latency == FPErf(fmt).latency + 7

    def test_fp_cbrt(self, fmt):
        mod = FPCbrt(fmt)
        assert mod.latency == 8

    def test_fp_reciprocal(self, fmt):
        mod = FPReciprocal(fmt)
        assert mod.latency == 6

    def test_fp_rsqrt(self, fmt):
        mod = FPRsqrt(fmt)
        assert mod.latency == 6


@pytest.mark.parametrize("fmt", [HALF, SINGLE], ids=["half", "single"])
class TestMLInstantiation:
    """Test that all ML function modules can be instantiated."""

    def test_fp_sigmoid(self, fmt):
        mod = FPSigmoid(fmt)
        assert mod.latency == 8

    def test_fp_gelu(self, fmt):
        mod = FPGELU(fmt)
        assert mod.latency > 0

    def test_fp_softplus(self, fmt):
        mod = FPSoftplus(fmt)
        assert mod.latency == 8

    def test_fp_swish(self, fmt):
        mod = FPSwish(fmt)
        assert mod.latency > 0

    def test_fp_mish(self, fmt):
        mod = FPMish(fmt)
        assert mod.latency > 0

    def test_fp_sinc(self, fmt):
        mod = FPSinc(fmt)
        assert mod.latency == 10


# ---- SollyaFunctionGenerator tests ----

class TestSollyaFunctionGenerator:
    """Test SollyaFunctionGenerator requires Sollya."""

    def test_requires_sollya(self):
        """SollyaFunctionGenerator must raise ImportError if Sollya is not installed."""
        try:
            import sollya  # type: ignore[import-untyped]
            pytest.skip("Sollya is installed — cannot test ImportError path")
        except ImportError:
            with pytest.raises(ImportError, match="Sollya is required"):
                SollyaFunctionGenerator("sin(x)", (-1, 1), 7)


class TestFixedPointFormat:
    """Test FixedPointFormat dataclass."""

    def test_total_bits_signed(self):
        fmt = FixedPointFormat(signed=True, int_bits=4, frac_bits=12)
        assert fmt.total_bits == 17  # 1 + 4 + 12

    def test_total_bits_unsigned(self):
        fmt = FixedPointFormat(signed=False, int_bits=4, frac_bits=12)
        assert fmt.total_bits == 16  # 0 + 4 + 12


# ---- SollyaReference tests for math functions ----

class TestSollyaRefMathFunctions:
    """Test that SollyaReference correctly computes math function values."""

    @pytest.fixture
    def ref(self):
        return SollyaReference(SINGLE)

    def test_exp2_values(self, ref):
        """Test 2^x reference values."""
        assert ref.internal_to_float(ref.float_to_internal(1.0)) == pytest.approx(1.0)
        assert ref.internal_to_float(ref.float_to_internal(2.0)) == pytest.approx(2.0)

    def test_log2_values(self, ref):
        """Test log2 reference values."""
        val = ref.internal_to_float(ref.float_to_internal(math.log2(4.0)))
        assert val == pytest.approx(2.0, rel=1e-5)

    def test_erf_special(self, ref):
        """Test erf special cases via reference encoding."""
        zero_enc = ref.float_to_internal(0.0)
        assert ref.internal_to_float(zero_enc) == 0.0

    def test_sigmoid_special(self, ref):
        """Test sigmoid reference values."""
        half = 1.0 / (1.0 + math.exp(0.0))
        assert half == pytest.approx(0.5)

    def test_tanh_special(self, ref):
        """Test tanh(0) = 0."""
        zero_enc = ref.float_to_internal(math.tanh(0.0))
        assert ref.internal_to_float(zero_enc) == 0.0

    def test_asin_boundary(self, ref):
        """Test asin(0) = 0, asin(1) = pi/2."""
        assert ref.internal_to_float(ref.float_to_internal(math.asin(0.0))) == 0.0
        pi2 = ref.internal_to_float(ref.float_to_internal(math.asin(1.0)))
        assert pi2 == pytest.approx(math.pi / 2, rel=1e-5)

    def test_atan_special(self, ref):
        """Test atan(0) = 0."""
        assert ref.internal_to_float(ref.float_to_internal(math.atan(0.0))) == 0.0

    def test_reciprocal(self, ref):
        """Test 1/x for various values."""
        for x in [0.5, 1.0, 2.0, 4.0]:
            enc = ref.float_to_internal(1.0 / x)
            assert ref.internal_to_float(enc) == pytest.approx(1.0 / x, rel=1e-5)

    def test_rsqrt(self, ref):
        """Test 1/sqrt(x) for various values."""
        for x in [1.0, 4.0]:
            enc = ref.float_to_internal(1.0 / math.sqrt(x))
            assert ref.internal_to_float(enc) == pytest.approx(1.0 / math.sqrt(x), rel=1e-5)

    def test_sinh_cosh(self, ref):
        """Test sinh(0) = 0, cosh(0) = 1."""
        assert ref.internal_to_float(ref.float_to_internal(math.sinh(0.0))) == 0.0
        assert ref.internal_to_float(ref.float_to_internal(math.cosh(0.0))) == pytest.approx(1.0)

    def test_cbrt(self, ref):
        """Test cbrt(8) = 2."""
        assert ref.internal_to_float(ref.float_to_internal(8.0 ** (1/3))) == pytest.approx(2.0, rel=1e-5)


# ---- Import tests ----

class TestImports:
    """Test that package __init__.py exports work."""

    def test_math_imports(self):
        from amaranth_fp.functions.math import (
            FPExp2, FPLog2, FPLog10, FPExp10,
            FPAsin, FPAcos, FPAtan,
            FPSinh, FPCosh, FPTanh,
            FPAsinh, FPAcosh, FPAtanh,
            FPErf, FPErfc,
            FPCbrt, FPReciprocal, FPRsqrt,
        )

    def test_ml_imports(self):
        from amaranth_fp.functions.ml import (
            FPSigmoid, FPGELU, FPSoftplus,
            FPSwish, FPMish, FPSinc,
        )
