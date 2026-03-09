"""Hardware simulation tests for all 24 math/ML floating-point operators.

Each test instantiates the operator, runs a clocked Amaranth simulation with
proper latency waits, and compares the output against mpmath reference values
via SollyaReference.
"""
from __future__ import annotations

import math
import pytest
import mpmath
from amaranth import *
from amaranth.sim import Simulator

from amaranth_fp.format import FPFormat
from amaranth_fp.testing.sollya_reference import SollyaReference, EXC_ZERO, EXC_NORMAL, EXC_INF, EXC_NAN

# --- Math operators ---
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

# --- ML operators ---
from amaranth_fp.functions.ml.fp_sigmoid import FPSigmoid
from amaranth_fp.functions.ml.fp_gelu import FPGELU
from amaranth_fp.functions.ml.fp_softplus import FPSoftplus
from amaranth_fp.functions.ml.fp_swish import FPSwish
from amaranth_fp.functions.ml.fp_mish import FPMish
from amaranth_fp.functions.ml.fp_sinc import FPSinc

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

FMT = FPFormat.half()  # Use half-precision for fast simulation


def _ref():
    return SollyaReference(FMT)


def _exc(encoded: int) -> int:
    """Extract exception field from encoded FloPoCo value."""
    return (encoded >> (1 + FMT.we + FMT.wf)) & 0b11


def _to_float(encoded: int) -> float:
    return _ref().internal_to_float(encoded)


def simulate_unary(dut, input_val: float) -> int:
    """Run a unary FP operator through clocked simulation, return raw output."""
    ref = _ref()
    encoded_input = ref.float_to_internal(input_val)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    result = [None]

    async def testbench(ctx):
        ctx.set(dut.a, encoded_input)
        # Wait latency + 1 extra cycle for output register
        for _ in range(dut.latency + 1):
            await ctx.tick()
        result[0] = ctx.get(dut.o)

    sim.add_testbench(testbench)
    with sim.write_vcd("test_hw.vcd"):
        sim.run()

    return result[0]


def assert_close(got_encoded: int, expected_float: float, tol_ulps: int = 4):
    """Assert hardware result is within tol_ulps of expected value."""
    ref = _ref()
    got_exc = _exc(got_encoded)
    expected_encoded = ref.float_to_internal(expected_float)
    exp_exc = _exc(expected_encoded)

    # Special value checks
    if math.isnan(expected_float):
        assert got_exc == EXC_NAN, f"Expected NaN, got exc={got_exc}"
        return
    if math.isinf(expected_float):
        assert got_exc == EXC_INF, f"Expected Inf, got exc={got_exc}"
        return
    if expected_float == 0.0:
        assert got_exc == EXC_ZERO, f"Expected zero, got exc={got_exc}"
        return

    # For normal values, check ULP distance
    got_float = ref.internal_to_float(got_encoded)
    if expected_float == 0.0:
        assert abs(got_float) < 1e-6
        return

    # Calculate ULP distance approximately
    rel_err = abs(got_float - expected_float) / max(abs(expected_float), 1e-30)
    # Half precision has ~10 mantissa bits → 1 ULP ≈ 2^-10 ≈ 0.001
    ulp_size = 2.0 ** (-FMT.wf)
    ulp_dist = rel_err / ulp_size if ulp_size > 0 else rel_err

    assert ulp_dist <= tol_ulps, (
        f"Result {got_float} too far from expected {expected_float}: "
        f"~{ulp_dist:.1f} ULPs (max {tol_ulps})"
    )


def assert_exc(got_encoded: int, expected_exc: int):
    """Assert only the exception code."""
    assert _exc(got_encoded) == expected_exc, (
        f"Expected exc={expected_exc}, got exc={_exc(got_encoded)} "
        f"(encoded=0x{got_encoded:x})"
    )


# =========================================================================== #
# Math function tests
# =========================================================================== #

class TestFPExp2Hardware:
    def test_exp2_zero(self):
        """2^0 = 1"""
        dut = FPExp2(FMT)
        r = simulate_unary(dut, 0.0)
        assert_exc(r, EXC_NORMAL)

    def test_exp2_one(self):
        """2^1 = 2"""
        dut = FPExp2(FMT)
        r = simulate_unary(dut, 1.0)
        assert_close(r, 2.0)

    def test_exp2_neg_inf(self):
        """2^(-inf) = 0"""
        dut = FPExp2(FMT)
        r = simulate_unary(dut, float('-inf'))
        assert_exc(r, EXC_ZERO)


class TestFPLog2Hardware:
    def test_log2_one(self):
        """log2(1) = 0"""
        dut = FPLog2(FMT)
        r = simulate_unary(dut, 1.0)
        assert_exc(r, EXC_ZERO)

    def test_log2_two(self):
        """log2(2) = 1"""
        dut = FPLog2(FMT)
        r = simulate_unary(dut, 2.0)
        assert_close(r, 1.0)

    def test_log2_zero(self):
        """log2(0) = -inf"""
        dut = FPLog2(FMT)
        r = simulate_unary(dut, 0.0)
        assert_exc(r, EXC_INF)


class TestFPLog10Hardware:
    def test_log10_one(self):
        """log10(1) = 0"""
        dut = FPLog10(FMT)
        r = simulate_unary(dut, 1.0)
        assert_exc(r, EXC_ZERO)

    def test_log10_ten(self):
        """log10(10) = 1"""
        dut = FPLog10(FMT)
        r = simulate_unary(dut, 10.0)
        assert_close(r, 1.0)


class TestFPExp10Hardware:
    def test_exp10_zero(self):
        """10^0 = 1"""
        dut = FPExp10(FMT)
        r = simulate_unary(dut, 0.0)
        assert_exc(r, EXC_NORMAL)

    def test_exp10_one(self):
        """10^1 = 10"""
        dut = FPExp10(FMT)
        r = simulate_unary(dut, 1.0)
        assert_close(r, 10.0)


class TestFPAsinHardware:
    def test_asin_zero(self):
        """asin(0) = 0"""
        dut = FPAsin(FMT)
        r = simulate_unary(dut, 0.0)
        assert_exc(r, EXC_ZERO)

    def test_asin_half(self):
        """asin(0.5) ≈ 0.5236"""
        dut = FPAsin(FMT)
        r = simulate_unary(dut, 0.5)
        assert_close(r, float(mpmath.asin(0.5)), tol_ulps=8)

    def test_asin_one(self):
        """asin(1) = pi/2"""
        dut = FPAsin(FMT)
        r = simulate_unary(dut, 1.0)
        assert_close(r, float(mpmath.asin(1.0)), tol_ulps=8)


class TestFPAcosHardware:
    def test_acos_one(self):
        """acos(1) = 0"""
        dut = FPAcos(FMT)
        r = simulate_unary(dut, 1.0)
        assert_close(r, 0.0, tol_ulps=8)

    def test_acos_zero(self):
        """acos(0) = pi/2"""
        dut = FPAcos(FMT)
        r = simulate_unary(dut, 0.0)
        assert_close(r, float(mpmath.acos(0.0)), tol_ulps=8)


class TestFPAtanHardware:
    def test_atan_zero(self):
        """atan(0) = 0"""
        dut = FPAtan(FMT)
        r = simulate_unary(dut, 0.0)
        assert_exc(r, EXC_ZERO)

    def test_atan_one(self):
        """atan(1) = pi/4"""
        dut = FPAtan(FMT)
        r = simulate_unary(dut, 1.0)
        assert_close(r, float(mpmath.atan(1.0)), tol_ulps=8)


class TestFPSinhHardware:
    def test_sinh_zero(self):
        """sinh(0) = 0"""
        dut = FPSinh(FMT)
        r = simulate_unary(dut, 0.0)
        assert_exc(r, EXC_ZERO)

    def test_sinh_one(self):
        """sinh(1) ≈ 1.1752"""
        dut = FPSinh(FMT)
        r = simulate_unary(dut, 1.0)
        assert_close(r, float(mpmath.sinh(1.0)), tol_ulps=8)


class TestFPCoshHardware:
    def test_cosh_zero(self):
        """cosh(0) = 1"""
        dut = FPCosh(FMT)
        r = simulate_unary(dut, 0.0)
        assert_close(r, 1.0, tol_ulps=4)

    def test_cosh_one(self):
        """cosh(1) ≈ 1.5431"""
        dut = FPCosh(FMT)
        r = simulate_unary(dut, 1.0)
        assert_close(r, float(mpmath.cosh(1.0)), tol_ulps=8)


class TestFPTanhHardware:
    def test_tanh_zero(self):
        """tanh(0) = 0"""
        dut = FPTanh(FMT)
        r = simulate_unary(dut, 0.0)
        assert_exc(r, EXC_ZERO)

    def test_tanh_one(self):
        """tanh(1) ≈ 0.7616"""
        dut = FPTanh(FMT)
        r = simulate_unary(dut, 1.0)
        assert_close(r, float(mpmath.tanh(1.0)), tol_ulps=8)

    def test_tanh_large(self):
        """tanh(large) → ±1"""
        dut = FPTanh(FMT)
        r = simulate_unary(dut, 10.0)
        assert_close(r, 1.0, tol_ulps=4)


class TestFPAsinhHardware:
    def test_asinh_zero(self):
        """asinh(0) = 0"""
        dut = FPAsinh(FMT)
        r = simulate_unary(dut, 0.0)
        assert_exc(r, EXC_ZERO)

    def test_asinh_one(self):
        """asinh(1) ≈ 0.8814"""
        dut = FPAsinh(FMT)
        r = simulate_unary(dut, 1.0)
        assert_close(r, float(mpmath.asinh(1.0)), tol_ulps=8)


class TestFPAcoshHardware:
    def test_acosh_one(self):
        """acosh(1) = 0"""
        dut = FPAcosh(FMT)
        r = simulate_unary(dut, 1.0)
        assert_close(r, 0.0, tol_ulps=4)

    def test_acosh_two(self):
        """acosh(2) ≈ 1.3170"""
        dut = FPAcosh(FMT)
        r = simulate_unary(dut, 2.0)
        assert_close(r, float(mpmath.acosh(2.0)), tol_ulps=8)


class TestFPAtanhHardware:
    def test_atanh_zero(self):
        """atanh(0) = 0"""
        dut = FPAtanh(FMT)
        r = simulate_unary(dut, 0.0)
        assert_exc(r, EXC_ZERO)

    def test_atanh_half(self):
        """atanh(0.5) ≈ 0.5493"""
        dut = FPAtanh(FMT)
        r = simulate_unary(dut, 0.5)
        assert_close(r, float(mpmath.atanh(0.5)), tol_ulps=8)


class TestFPErfHardware:
    def test_erf_zero(self):
        """erf(0) = 0"""
        dut = FPErf(FMT)
        r = simulate_unary(dut, 0.0)
        assert_exc(r, EXC_ZERO)

    def test_erf_one(self):
        """erf(1) ≈ 0.8427"""
        dut = FPErf(FMT)
        r = simulate_unary(dut, 1.0)
        assert_close(r, float(mpmath.erf(1.0)), tol_ulps=8)


class TestFPErfcHardware:
    def test_erfc_zero(self):
        """erfc(0) = 1"""
        dut = FPErfc(FMT)
        r = simulate_unary(dut, 0.0)
        assert_close(r, 1.0, tol_ulps=4)

    def test_erfc_one(self):
        """erfc(1) ≈ 0.1573"""
        dut = FPErfc(FMT)
        r = simulate_unary(dut, 1.0)
        assert_close(r, float(mpmath.erfc(1.0)), tol_ulps=8)


class TestFPCbrtHardware:
    def test_cbrt_zero(self):
        """cbrt(0) = 0"""
        dut = FPCbrt(FMT)
        r = simulate_unary(dut, 0.0)
        assert_exc(r, EXC_ZERO)

    def test_cbrt_one(self):
        """cbrt(1) = 1"""
        dut = FPCbrt(FMT)
        r = simulate_unary(dut, 1.0)
        assert_close(r, 1.0, tol_ulps=4)

    def test_cbrt_eight(self):
        """cbrt(8) = 2"""
        dut = FPCbrt(FMT)
        r = simulate_unary(dut, 8.0)
        assert_close(r, 2.0, tol_ulps=4)


class TestFPReciprocalHardware:
    def test_recip_one(self):
        """1/1 = 1"""
        dut = FPReciprocal(FMT)
        r = simulate_unary(dut, 1.0)
        assert_close(r, 1.0, tol_ulps=4)

    def test_recip_two(self):
        """1/2 = 0.5"""
        dut = FPReciprocal(FMT)
        r = simulate_unary(dut, 2.0)
        assert_close(r, 0.5, tol_ulps=4)

    def test_recip_zero(self):
        """1/0 = inf"""
        dut = FPReciprocal(FMT)
        r = simulate_unary(dut, 0.0)
        assert_exc(r, EXC_INF)


class TestFPRsqrtHardware:
    def test_rsqrt_one(self):
        """1/sqrt(1) = 1"""
        dut = FPRsqrt(FMT)
        r = simulate_unary(dut, 1.0)
        assert_close(r, 1.0, tol_ulps=4)

    def test_rsqrt_four(self):
        """1/sqrt(4) = 0.5"""
        dut = FPRsqrt(FMT)
        r = simulate_unary(dut, 4.0)
        assert_close(r, 0.5, tol_ulps=4)

    def test_rsqrt_zero(self):
        """1/sqrt(0) = inf"""
        dut = FPRsqrt(FMT)
        r = simulate_unary(dut, 0.0)
        assert_exc(r, EXC_INF)


# =========================================================================== #
# ML function tests
# =========================================================================== #

class TestFPSigmoidHardware:
    def test_sigmoid_zero(self):
        """sigmoid(0) = 0.5"""
        dut = FPSigmoid(FMT)
        r = simulate_unary(dut, 0.0)
        assert_close(r, 0.5, tol_ulps=8)

    def test_sigmoid_large_pos(self):
        """sigmoid(large) → 1"""
        dut = FPSigmoid(FMT)
        r = simulate_unary(dut, 10.0)
        assert_close(r, 1.0, tol_ulps=8)

    def test_sigmoid_large_neg(self):
        """sigmoid(-large) → 0"""
        dut = FPSigmoid(FMT)
        r = simulate_unary(dut, -10.0)
        assert_exc(r, EXC_ZERO)


class TestFPGELUHardware:
    def test_gelu_zero(self):
        """GELU(0) = 0"""
        dut = FPGELU(FMT)
        r = simulate_unary(dut, 0.0)
        assert_exc(r, EXC_ZERO)

    def test_gelu_one(self):
        """GELU(1) ≈ 0.8412"""
        dut = FPGELU(FMT)
        r = simulate_unary(dut, 1.0)
        expected = float(1.0 * 0.5 * (1.0 + float(mpmath.erf(1.0 / mpmath.sqrt(2)))))
        assert_close(r, expected, tol_ulps=16)


class TestFPSoftplusHardware:
    def test_softplus_zero(self):
        """softplus(0) = ln(2) ≈ 0.6931"""
        dut = FPSoftplus(FMT)
        r = simulate_unary(dut, 0.0)
        assert_close(r, float(mpmath.log(2)), tol_ulps=8)

    def test_softplus_large(self):
        """softplus(large) ≈ x"""
        dut = FPSoftplus(FMT)
        r = simulate_unary(dut, 10.0)
        assert_close(r, 10.0, tol_ulps=8)


class TestFPSwishHardware:
    def test_swish_zero(self):
        """swish(0) = 0"""
        dut = FPSwish(FMT)
        r = simulate_unary(dut, 0.0)
        assert_exc(r, EXC_ZERO)

    def test_swish_one(self):
        """swish(1) = 1 * sigmoid(1) ≈ 0.7311"""
        dut = FPSwish(FMT)
        r = simulate_unary(dut, 1.0)
        expected = 1.0 / (1.0 + math.exp(-1.0))
        assert_close(r, expected, tol_ulps=16)


class TestFPMishHardware:
    def test_mish_zero(self):
        """mish(0) = 0"""
        dut = FPMish(FMT)
        r = simulate_unary(dut, 0.0)
        assert_exc(r, EXC_ZERO)

    def test_mish_one(self):
        """mish(1) = tanh(softplus(1))"""
        dut = FPMish(FMT)
        r = simulate_unary(dut, 1.0)
        sp = float(mpmath.log(1 + mpmath.exp(1)))
        expected = float(mpmath.tanh(sp))
        assert_close(r, expected, tol_ulps=16)


class TestFPSincHardware:
    def test_sinc_zero(self):
        """sinc(0) = 1 (by convention)"""
        dut = FPSinc(FMT)
        r = simulate_unary(dut, 0.0)
        assert_close(r, 1.0, tol_ulps=4)

    def test_sinc_one(self):
        """sinc(1) = sin(1)/1 ≈ 0.8415"""
        dut = FPSinc(FMT)
        r = simulate_unary(dut, 1.0)
        assert_close(r, float(mpmath.sin(1.0)), tol_ulps=8)
