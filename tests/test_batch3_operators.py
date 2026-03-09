"""Tests for batch 3 operators."""
import pytest
from amaranth import *
from amaranth.sim import Simulator

# Primitives
from amaranth_fp.primitives import GenericLut, GenericMult, GenericMux, RowAdder, BooleanEquation

# New operators
from amaranth_fp.operators import (
    LNSAddSub, Cotran, CotranHybrid, LNSAtanPow, LNSLogSinCos,
    IntIntKCM, CRFPConstMult, FixAtan2ByRecipMultAtan, FixSinPoly,
    Atan2Table, IEEEFPExp, FPLogIterative, FPSqrtPoly, FixNormNaive, TaoSort,
)

# New conversions
from amaranth_fp.conversions import PIF2Posit, Posit2PIF

# New functions
from amaranth_fp.functions import FixFunctionByVaryingPiecewisePoly, HOTBM, TableOperator

# New filters
from amaranth_fp.filters import FixRootRaisedCosine, IntFIRTransposed

# New complex
from amaranth_fp.complex import IntTwiddleMultiplier, IntFFTButterfly

# Test utilities
from amaranth_fp.testing import FPNumber, IEEENumber


def _sim_cycles(dut, n=5):
    """Run n clock cycles in a sim and return."""
    sim = Simulator(dut)
    sim.add_clock(1e-6)

    async def bench(ctx):
        for _ in range(n):
            await ctx.tick()

    sim.add_testbench(bench)
    with sim.write_vcd("test_batch3.vcd"):
        sim.run()


class TestPrimitives:
    def test_generic_lut_instantiate(self):
        contents = [i * 2 for i in range(16)]
        dut = GenericLut(4, 8, contents)
        assert dut.latency == 1
        _sim_cycles(dut)

    def test_generic_mult_instantiate(self):
        dut = GenericMult(8, 8)
        assert dut.latency == 1
        assert len(dut.o) == 16
        _sim_cycles(dut)

    def test_generic_mux_instantiate(self):
        dut = GenericMux(8, 4)
        assert dut.latency == 1
        assert len(dut.inputs) == 4
        _sim_cycles(dut)

    def test_row_adder_instantiate(self):
        dut = RowAdder(8, 3)
        assert dut.latency == 1
        _sim_cycles(dut)

    def test_boolean_equation_instantiate(self):
        dut = BooleanEquation(3, 0b10010110)  # XOR of 3 bits
        assert dut.latency == 1
        _sim_cycles(dut)


class TestLNSOperators:
    def test_lns_add_sub(self):
        dut = LNSAddSub(16)
        assert dut.latency == 2
        _sim_cycles(dut)

    def test_cotran(self):
        dut = Cotran(16)
        assert dut.latency == 1
        _sim_cycles(dut)

    def test_cotran_hybrid(self):
        dut = CotranHybrid(16)
        assert dut.latency == 2
        _sim_cycles(dut)

    def test_lns_atan_pow(self):
        dut = LNSAtanPow(16)
        assert dut.latency == 1
        _sim_cycles(dut)

    def test_lns_log_sin_cos(self):
        dut = LNSLogSinCos(16)
        assert dut.latency == 1
        _sim_cycles(dut)


class TestConversions:
    def test_pif2posit(self):
        dut = PIF2Posit(16, 1)
        assert dut.latency == 1
        _sim_cycles(dut)

    def test_posit2pif(self):
        dut = Posit2PIF(16, 1)
        assert dut.latency == 1
        _sim_cycles(dut)


class TestConstMult:
    def test_int_int_kcm(self):
        dut = IntIntKCM(16, 42)
        assert dut.latency == 1
        _sim_cycles(dut)

    def test_cr_fp_const_mult(self):
        dut = CRFPConstMult(8, 23, 3.14159)
        assert dut.latency == 2
        _sim_cycles(dut)


class TestFixFunctions:
    def test_varying_piecewise_poly(self):
        dut = FixFunctionByVaryingPiecewisePoly(8, 8)
        assert dut.latency == 2
        _sim_cycles(dut)

    def test_hotbm(self):
        dut = HOTBM(8, 8, order=2)
        assert dut.latency == 1
        _sim_cycles(dut)

    def test_table_operator(self):
        dut = TableOperator(4, 8, [i * 3 for i in range(16)])
        assert dut.latency == 1
        _sim_cycles(dut)


class TestFilters:
    def test_fix_root_raised_cosine(self):
        dut = FixRootRaisedCosine(16)
        assert dut.latency == 1
        _sim_cycles(dut)

    def test_int_fir_transposed(self):
        dut = IntFIRTransposed(16, [1, 2, 1])
        assert dut.latency == 1
        _sim_cycles(dut)


class TestTrig:
    def test_fix_atan2_by_recip_mult_atan(self):
        dut = FixAtan2ByRecipMultAtan(16)
        assert dut.latency == 3
        _sim_cycles(dut)

    def test_fix_sin_poly(self):
        dut = FixSinPoly(16)
        assert dut.latency == 2
        _sim_cycles(dut)

    def test_atan2_table(self):
        dut = Atan2Table(4, 8)
        assert dut.latency == 1
        _sim_cycles(dut)


class TestExpLog:
    def test_ieee_fp_exp(self):
        dut = IEEEFPExp(8, 23)
        assert dut.latency == 3
        _sim_cycles(dut)

    def test_fp_log_iterative(self):
        dut = FPLogIterative(8, 23, n_iterations=4)
        assert dut.latency == 4
        _sim_cycles(dut)


class TestSqrt:
    def test_fp_sqrt_poly(self):
        dut = FPSqrtPoly(8, 23)
        assert dut.latency == 2
        _sim_cycles(dut)


class TestComplex:
    def test_int_twiddle_multiplier(self):
        dut = IntTwiddleMultiplier(16, n=8, k=0)
        assert dut.latency == 1
        _sim_cycles(dut)

    def test_int_fft_butterfly(self):
        dut = IntFFTButterfly(16)
        assert dut.latency == 1
        _sim_cycles(dut)


class TestNorms:
    def test_fix_norm_naive(self):
        dut = FixNormNaive(16)
        assert dut.latency == 2
        _sim_cycles(dut)


class TestSort:
    def test_tao_sort(self):
        dut = TaoSort(8, n_inputs=4)
        assert dut.latency >= 1
        _sim_cycles(dut)


class TestUtilities:
    def test_fp_number_encode_decode(self):
        fp = FPNumber(8, 23)
        for val in [0.0, 1.0, -1.0, 3.14]:
            bits = fp.encode(val)
            decoded = fp.decode(bits)
            if val == 0.0:
                assert decoded == 0.0
            else:
                assert abs(decoded - val) / abs(val) < 1e-5

    def test_fp_number_special(self):
        fp = FPNumber(8, 23)
        import math
        assert fp.decode(fp.encode(float("inf"))) == float("inf")
        assert fp.decode(fp.encode(float("-inf"))) == float("-inf")
        assert math.isnan(fp.decode(fp.encode(float("nan"))))
        assert fp.decode(fp.encode(0.0)) == 0.0

    def test_ieee_number_single(self):
        ieee = IEEENumber(8, 23)
        for val in [0.0, 1.0, -1.0, 3.14]:
            bits = ieee.encode(val)
            decoded = ieee.decode(bits)
            if val == 0.0:
                assert decoded == 0.0
            else:
                assert abs(decoded - val) / abs(val) < 1e-5

    def test_ieee_number_double(self):
        ieee = IEEENumber(11, 52)
        bits = ieee.encode(2.718281828)
        decoded = ieee.decode(bits)
        assert abs(decoded - 2.718281828) < 1e-8
