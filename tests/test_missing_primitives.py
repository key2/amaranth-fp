"""Tests for all newly implemented FloPoCo primitives."""
import pytest
from amaranth import *
from amaranth.sim import Simulator


def sim_check(dut, setup, check, *, clks=10):
    """Helper: run sync sim, apply setup, then check outputs."""
    sim = Simulator(dut)
    sim.add_clock(1e-6)
    results = {}

    async def proc(ctx):
        for _ in range(2):
            await ctx.tick()
        setup(ctx)
        for _ in range(clks):
            await ctx.tick()
        check(ctx, results)

    sim.add_testbench(proc)
    with sim.write_vcd("test_missing.vcd"):
        sim.run()
    return results


# ── BitHeap ──────────────────────────────────────────────────────────

class TestBitHeap:
    def test_weighted_bit(self):
        from amaranth_fp.bitheap import WeightedBit
        s = Signal(1, name="b")
        wb = WeightedBit(s, weight=3, cycle=0)
        assert wb.weight == 3
        assert wb.cycle == 0

    def test_bit(self):
        from amaranth_fp.bitheap import Bit
        s = Signal(1)
        b = Bit(s, column=5, uid=1)
        assert b.column == 5

    def test_compression_strategy(self):
        from amaranth_fp.bitheap import CompressionStrategy
        cs = CompressionStrategy(64)
        assert cs.heap_width == 64

    def test_bit_heap_solution(self):
        from amaranth_fp.bitheap import BitHeapSolution
        sol = BitHeapSolution()
        assert sol.total_latency == 0

    def test_first_fitting(self):
        from amaranth_fp.bitheap import FirstFittingCompressionStrategy
        ff = FirstFittingCompressionStrategy(32)
        assert ff.heap_width == 32

    def test_max_efficiency(self):
        from amaranth_fp.bitheap import MaxEfficiencyCompressionStrategy
        me = MaxEfficiencyCompressionStrategy(32)
        assert me.heap_width == 32

    def test_parandeh_afshar(self):
        from amaranth_fp.bitheap import ParandehAfsharCompressionStrategy
        pa = ParandehAfsharCompressionStrategy(32)
        assert pa.heap_width == 32


# ── Building Blocks ──────────────────────────────────────────────────

class TestBuildingBlocks:
    def test_one_hot_decoder(self):
        from amaranth_fp.building_blocks import OneHotDecoder
        dut = OneHotDecoder(3)
        assert dut.latency == 0

    def test_thermometer_decoder(self):
        from amaranth_fp.building_blocks import ThermometerDecoder
        dut = ThermometerDecoder(3)
        assert dut.latency == 0

    def test_lzoc3(self):
        from amaranth_fp.building_blocks import LZOC3
        dut = LZOC3(8)
        assert dut.latency == 1


# ── Complex ──────────────────────────────────────────────────────────

class TestComplex:
    def test_fix_complex_kcm(self):
        from amaranth_fp.complex import FixComplexKCM
        dut = FixComplexKCM(1, -8, 0.5, 0.25)
        assert dut.latency == 2

    def test_fix_fft_fully_pa(self):
        from amaranth_fp.complex import FixFFTFullyPA
        dut = FixFFTFullyPA(4, 1, -8)
        assert dut.latency == 4

    def test_int_fft(self):
        from amaranth_fp.complex import IntFFT
        dut = IntFFT(4, 16)
        assert dut.latency == 4

    def test_int_fft_level_dit2(self):
        from amaranth_fp.complex import IntFFTLevelDIT2
        dut = IntFFTLevelDIT2(16, 4)
        assert dut.latency == 2

    def test_int_twiddle_mult_alt(self):
        from amaranth_fp.complex import IntTwiddleMultiplierAlternative
        dut = IntTwiddleMultiplierAlternative(16)
        assert dut.latency == 2


# ── DAG ──────────────────────────────────────────────────────────────

class TestDAG:
    def test_dag_operator(self):
        from amaranth_fp.dag import DAGOperator
        dut = DAGOperator(16)
        dut.add_node("op1")
        dut.add_edge("op1", "op2")
        assert len(dut.nodes) == 1
        assert len(dut.edges) == 1


# ── Integer ──────────────────────────────────────────────────────────

class TestInteger:
    def test_carry_gen_circuit(self):
        from amaranth_fp.integer import CarryGenerationCircuit
        dut = CarryGenerationCircuit(8)
        assert dut.latency == 1

    def test_fix_multi_adder(self):
        from amaranth_fp.integer import FixMultiAdder
        dut = FixMultiAdder(8, 4)
        assert len(dut.inputs) == 4

    def test_int_multi_adder(self):
        from amaranth_fp.integer import IntMultiAdder
        dut = IntMultiAdder(8, 3)
        assert dut.latency == 2

    def test_base_multiplier(self):
        from amaranth_fp.integer import BaseMultiplier
        dut = BaseMultiplier(8, 8)
        assert dut.latency == 2

    def test_dsp_block(self):
        from amaranth_fp.integer import DSPBlock
        dut = DSPBlock(18, 25)
        assert dut.latency == 3

    def test_fix_mult_add(self):
        from amaranth_fp.integer import FixMultAdd
        dut = FixMultAdd(8)
        assert dut.latency == 3

    def test_int_multiplier_lut(self):
        from amaranth_fp.integer import IntMultiplierLUT
        dut = IntMultiplierLUT(4, 4)
        assert dut.latency == 1

    def test_base_squarer_lut(self):
        from amaranth_fp.integer import BaseSquarerLUT
        dut = BaseSquarerLUT(4)
        assert dut.latency == 1


# ── Operators ────────────────────────────────────────────────────────

class TestNewOperators:
    def test_fp_add_single_path(self):
        from amaranth_fp.operators import FPAddSinglePath
        dut = FPAddSinglePath(8, 23)
        assert dut.latency == 4

    def test_fix_real_const_mult(self):
        from amaranth_fp.operators import FixRealConstMult
        dut = FixRealConstMult(1, -8, -16, 3.14)
        assert dut.latency == 2

    def test_int_const_mult_shift_add(self):
        from amaranth_fp.operators import IntConstMultShiftAdd
        dut = IntConstMultShiftAdd(8, 5)
        assert dut.latency == 2

    def test_fix_resize(self):
        from amaranth_fp.operators import FixResize
        dut = FixResize(1, -8, 2, -16)
        assert dut.latency == 1

    def test_shift_reg(self):
        from amaranth_fp.operators import ShiftReg
        dut = ShiftReg(8, 5)
        assert dut.latency == 5

    def test_fix_atan2_bivariate(self):
        from amaranth_fp.operators import FixAtan2ByBivariateApprox
        dut = FixAtan2ByBivariateApprox(1, -8)
        assert dut.latency == 4

    def test_fix_atan2_cordic(self):
        from amaranth_fp.operators import FixAtan2ByCORDIC
        dut = FixAtan2ByCORDIC(1, -8)
        assert dut.latency == 10  # 1 - (-8) + 1 = 10

    def test_fix_sincos_cordic(self):
        from amaranth_fp.operators import FixSinCosCORDIC
        dut = FixSinCosCORDIC(1, -8)
        assert dut.latency == 10

    def test_const_div3(self):
        from amaranth_fp.operators import ConstDiv3ForSinPoly
        dut = ConstDiv3ForSinPoly(16)
        assert dut.latency == 1

    def test_exp(self):
        from amaranth_fp.operators import Exp
        dut = Exp(1, -8, 4, -8)
        assert dut.latency == 4

    def test_ieee_float_format(self):
        from amaranth_fp.operators import IEEEFloatFormat
        f = IEEEFloatFormat(8, 23)
        assert f.width == 32
        assert f.bias == 127

    def test_ieee_float_format_presets(self):
        from amaranth_fp.operators import IEEEFloatFormat
        assert IEEEFloatFormat.binary16().width == 16
        assert IEEEFloatFormat.binary32().width == 32
        assert IEEEFloatFormat.binary64().width == 64

    def test_log_sin_cos(self):
        from amaranth_fp.operators import LogSinCos
        dut = LogSinCos(16)
        assert dut.latency == 3

    def test_fix_2d_norm(self):
        from amaranth_fp.operators import Fix2DNorm
        dut = Fix2DNorm(1, -8)
        assert dut.latency == 3

    def test_fix_3d_norm(self):
        from amaranth_fp.operators import Fix3DNorm
        dut = Fix3DNorm(1, -8)
        assert dut.latency == 3

    def test_sort_wrapper(self):
        from amaranth_fp.operators import SortWrapper
        dut = SortWrapper(8, 4)
        assert len(dut.inputs) == 4
        assert len(dut.outputs) == 4

    def test_fix_constant(self):
        from amaranth_fp.operators import FixConstant
        dut = FixConstant(1, -8, 0.5)
        assert dut.latency == 0


# ── Posit ────────────────────────────────────────────────────────────

class TestPosit:
    def test_pif_add(self):
        from amaranth_fp.posit import PIFAdd
        dut = PIFAdd(16)
        assert dut.latency == 3

    def test_pif2fix(self):
        from amaranth_fp.posit import PIF2Fix
        dut = PIF2Fix(16, 32)
        assert dut.latency == 2

    def test_posit_exp(self):
        from amaranth_fp.posit import PositExp
        dut = PositExp(16, 2)
        assert dut.latency == 4

    def test_posit_function(self):
        from amaranth_fp.posit import PositFunction
        dut = PositFunction(16, 2, "sin(x)")
        assert dut.latency == 3

    def test_posit_function_by_table(self):
        from amaranth_fp.posit import PositFunctionByTable
        dut = PositFunctionByTable(8, 0, "x^2")
        assert dut.latency == 1

    def test_posit2posit(self):
        from amaranth_fp.posit import Posit2Posit
        dut = Posit2Posit(8, 0, 16, 1)
        assert dut.latency == 2


# ── Primitives ───────────────────────────────────────────────────────

class TestPrimitives:
    def test_primitive_base(self):
        from amaranth_fp.primitives import Primitive
        p = Primitive("test")
        assert p.prim_name == "test"

    def test_intel_lcell(self):
        from amaranth_fp.primitives.intel import IntelLCELL
        dut = IntelLCELL(0xAAAA)
        assert dut.latency == 0

    def test_intel_rccm(self):
        from amaranth_fp.primitives.intel import IntelRCCM
        dut = IntelRCCM(8, 5)
        assert dut.latency == 1

    def test_intel_ternary_adder(self):
        from amaranth_fp.primitives.intel import IntelTernaryAdder
        dut = IntelTernaryAdder(8)
        assert dut.latency == 1

    def test_xilinx_carry4(self):
        from amaranth_fp.primitives.xilinx import XilinxCARRY4
        dut = XilinxCARRY4()
        assert dut.latency == 0

    def test_xilinx_lut5(self):
        from amaranth_fp.primitives.xilinx import XilinxLUT5
        dut = XilinxLUT5(init=0xDEADBEEF)
        assert dut.latency == 0

    def test_xilinx_lut6(self):
        from amaranth_fp.primitives.xilinx import XilinxLUT6
        dut = XilinxLUT6()
        assert dut.latency == 0

    def test_xilinx_muxf7(self):
        from amaranth_fp.primitives.xilinx import XilinxMUXF7
        dut = XilinxMUXF7()
        assert dut.latency == 0

    def test_xilinx_muxf8(self):
        from amaranth_fp.primitives.xilinx import XilinxMUXF8
        dut = XilinxMUXF8()
        assert dut.latency == 0

    def test_xilinx_fdce(self):
        from amaranth_fp.primitives.xilinx import XilinxFDCE
        dut = XilinxFDCE()
        assert dut.latency == 1

    def test_xilinx_cfglut5(self):
        from amaranth_fp.primitives.xilinx import XilinxCFGLUT5
        dut = XilinxCFGLUT5(init=0xFF)
        assert dut.latency == 0

    def test_xilinx_lookahead8(self):
        from amaranth_fp.primitives.xilinx import XilinxLOOKAHEAD8
        dut = XilinxLOOKAHEAD8()
        assert dut.latency == 0

    def test_xilinx_generic_mux(self):
        from amaranth_fp.primitives.xilinx import XilinxGenericMux
        dut = XilinxGenericMux(width=4, sel_bits=2)
        assert dut.latency == 0

    def test_xilinx_n2m_decoder(self):
        from amaranth_fp.primitives.xilinx import XilinxN2MDecoder
        dut = XilinxN2MDecoder(n=3)
        assert dut.latency == 0

    def test_xilinx_four_to_two_compressor(self):
        from amaranth_fp.primitives.xilinx import XilinxFourToTwoCompressor
        dut = XilinxFourToTwoCompressor(width=4)
        assert dut.latency == 0

    def test_xilinx_ternary_add_sub(self):
        from amaranth_fp.primitives.xilinx import XilinxTernaryAddSub
        dut = XilinxTernaryAddSub(width=8)
        assert dut.latency == 0

    def test_xilinx_gpc(self):
        from amaranth_fp.primitives.xilinx import XilinxGPC
        dut = XilinxGPC(column_heights=(3, 2))
        assert dut.latency == 0


# ── Functions ────────────────────────────────────────────────────────

class TestFunctions:
    def test_kcm_table(self):
        from amaranth_fp.functions import KCMTable
        dut = KCMTable(4, 8, 3)
        assert dut.latency == 1
        assert len(dut.contents) == 16

    def test_alpha(self):
        from amaranth_fp.functions import ALPHA
        dut = ALPHA(-4, -8)
        assert dut.latency == 2

    def test_fix_function_by_piecewise_poly(self):
        from amaranth_fp.functions import FixFunctionByPiecewisePoly
        coeffs = [[1, 2], [3, 4]]  # 2 segments, degree 1
        dut = FixFunctionByPiecewisePoly(8, 8, 2, 1, coeffs)
        assert dut.latency >= 1


# ── Filters ──────────────────────────────────────────────────────────

class TestFilters:
    def test_fix_iir_shift_add(self):
        from amaranth_fp.filters import FixIIRShiftAdd
        dut = FixIIRShiftAdd(1, -8, [0.5, 0.25, 0.125])
        assert dut.latency == 4


# ── Sorting ──────────────────────────────────────────────────────────

class TestSorting:
    def test_bitonic_sort(self):
        from amaranth_fp.sorting import BitonicSort
        dut = BitonicSort(8, 4)
        assert len(dut.inputs) == 4
        assert len(dut.outputs) == 4

    def test_optimal_depth_sort(self):
        from amaranth_fp.sorting import OptimalDepthSort
        dut = OptimalDepthSort(8, 4)
        assert len(dut.inputs) == 4


# ── Testing utilities ────────────────────────────────────────────────

class TestTestingUtils:
    def test_posit_number(self):
        from amaranth_fp.testing import PositNumber
        pn = PositNumber(8, 0)
        assert pn.encode(0) == 0
        assert pn.decode(0) == 0.0

    def test_register_sandwich(self):
        from amaranth_fp.testing import RegisterSandwich
        dut = RegisterSandwich(8)
        assert dut.latency == 2

    def test_test_bench(self):
        from amaranth_fp.testing import TestBench
        from amaranth_fp.operators import FPAbs
        from amaranth_fp.format import FPFormat
        inner = FPAbs(FPFormat(8, 23))
        tb = TestBench(inner)
        assert tb.dut is inner
