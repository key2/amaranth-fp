# amaranth-fp Documentation

Floating-point operator generator for [Amaranth HDL](https://amaranth-lang.org/), a complete port of [FloPoCo](http://flopoco.org/) to Python/Amaranth.

**amaranth-fp** provides **190+ parameterized, pipelined components** across **14 modules** covering floating-point arithmetic, transcendental functions, integer arithmetic, complex/FFT, DSP filters, function approximation, bit heap compression, posit arithmetic, LNS operators, sorting networks, format conversions, FPGA primitives, and testing infrastructure. All operators use a FloPoCo-style internal format with explicit exception flags for efficient FPGA synthesis.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [FP Format & Internal Format](#fp-format--internal-format)
- [FPGA Targets](#fpga-targets)
- [Pipelining](#pipelining)
- [Core FP Arithmetic](#core-fp-arithmetic)
- [FP Variants](#fp-variants)
- [Transcendentals](#transcendentals)
- [Constant Multipliers](#constant-multipliers)
- [Integer Arithmetic](#integer-arithmetic)
- [Complex / FFT](#complex--fft)
- [Filters / DSP](#filters--dsp)
- [Function Approximation](#function-approximation)
- [BitHeap Infrastructure](#bitheap-infrastructure)
- [Posit Number System](#posit-number-system)
- [LNS Operators](#lns-operators)
- [Norms](#norms)
- [Sorting](#sorting)
- [Primitive Components](#primitive-components)
- [Building Blocks](#building-blocks)
- [Format Conversions](#format-conversions)
- [DAG Operator Composition](#dag-operator-composition)
- [Testing Infrastructure](#testing-infrastructure)
- [Testing Guide](#testing-guide)
- [Complete API Summary Table](#complete-api-summary-table)

---

## Installation

```bash
git clone <repo-url>
cd amaranth-fp
pdm install          # basic install
pdm install -G dev   # with dev dependencies (pytest)
```

**Requirements:** Python ≥ 3.9, Amaranth ≥ 0.5

---

## Quick Start

```python
from amaranth import *
from amaranth_fp.format import FPFormat
from amaranth_fp.operators import FPAdd
from amaranth_fp.conversions import InputIEEE, OutputIEEE

fmt = FPFormat.half()  # 5-bit exponent, 10-bit fraction

class FPAddPipeline(Elaboratable):
    def __init__(self, fmt):
        self.fmt = fmt
        self.ieee_a = Signal(fmt.ieee_width)
        self.ieee_b = Signal(fmt.ieee_width)
        self.ieee_o = Signal(fmt.ieee_width)

    def elaborate(self, platform):
        m = Module()
        fmt = self.fmt
        m.submodules.in_a = in_a = InputIEEE(fmt)
        m.submodules.in_b = in_b = InputIEEE(fmt)
        m.d.comb += [in_a.ieee_in.eq(self.ieee_a), in_b.ieee_in.eq(self.ieee_b)]
        m.submodules.add = add = FPAdd(fmt)
        m.d.comb += [add.a.eq(in_a.fp_out), add.b.eq(in_b.fp_out)]
        m.submodules.out = out = OutputIEEE(fmt)
        m.d.comb += [out.fp_in.eq(add.o), self.ieee_o.eq(out.ieee_out)]
        return m

# Total latency: InputIEEE(1) + FPAdd(7) + OutputIEEE(1) = 9 cycles

from amaranth.back.verilog import convert
top = FPAddPipeline(fmt)
output = convert(top, ports=[top.ieee_a, top.ieee_b, top.ieee_o])
```

---

## FP Format & Internal Format

**File:** [`src/amaranth_fp/format.py`](../src/amaranth_fp/format.py)

### `FPFormat` class

Frozen dataclass parameterized by `we` (exponent width) and `wf` (fraction width). Validation: `we >= 2`, `wf >= 1`.

| Factory Method | we | wf | IEEE Name |
|---|---|---|---|
| `FPFormat.half()` | 5 | 10 | binary16 |
| `FPFormat.single()` | 8 | 23 | binary32 |
| `FPFormat.double()` | 11 | 52 | binary64 |
| `FPFormat.custom(we, wf)` | any | any | custom |

| Property | Formula | Description |
|---|---|---|
| `width` | `2 + 1 + we + wf` | Total internal (FloPoCo) bit width |
| `ieee_width` | `1 + we + wf` | Total IEEE 754 bit width |
| `bias` | `2^(we-1) - 1` | Exponent bias |
| `emin` | `1 - bias` | Minimum normal exponent |
| `emax` | `bias` | Maximum normal exponent |

### Internal Format Bit Layout (LSB to MSB)

```
┌──────────────┬──────────────┬──────┬───────────┐
│ mantissa     │ exponent     │ sign │ exception │
│ (wf bits)    │ (we bits)    │ (1)  │ (2 bits)  │
└──────────────┴──────────────┴──────┴───────────┘
```

| Exception Code | Meaning |
|---|---|
| `00` | Zero |
| `01` | Normal number |
| `10` | Infinity |
| `11` | NaN |

### Layout Functions

- **`ieee_layout(fmt)`** → `StructLayout`: `mantissa(wf)`, `exponent(we)`, `sign(1)`
- **`internal_layout(fmt)`** → `StructLayout`: `mantissa(wf)`, `exponent(we)`, `sign(1)`, `exception(2)`

---

## FPGA Targets

**File:** [`src/amaranth_fp/targets.py`](../src/amaranth_fp/targets.py)

`FPGATarget` is a frozen dataclass with timing parameters for pipeline-stage estimation. Method `cycles_for_delay(delay_ns)` computes pipeline stages needed.

| Factory | FPGA | LUT (ns) | Carry (ns) | DSP (ns) | FF (ns) | Max MHz |
|---|---|---|---|---|---|---|
| `Kintex7()` | Xilinx Kintex-7 | 0.043 | 0.049 | 2.892 | 0.216 | 741 |
| `VirtexUltrascalePlus()` | Xilinx VU+ | 0.035 | 0.015 | — | 0.150 | 775 |
| `Zynq7000()` | Xilinx Zynq-7000 | 0.124 | 0.114 | — | 0.518 | 500 |
| `StratixV()` | Intel Stratix V | 0.433 | 0.022 | 1.875 | 0.156 | 717 |
| `Versal()` | Xilinx Versal | 0.124 | 0.114 | — | 0.518 | 500 |
| `GenericTarget()` | Conservative default | 0.500 | 0.150 | 3.000 | 0.500 | 200 |

---

## Pipelining

### `PipelinedComponent` base class

**File:** [`src/amaranth_fp/pipelined.py`](../src/amaranth_fp/pipelined.py)

Base class for all pipelined operators. Provides `sync_to(m, sig, target_cycle)`, `add_latency(sig, cycles)`, `get_latency(sig)`.

### `PipelineHelper` utility

**File:** [`src/amaranth_fp/pipeline.py`](../src/amaranth_fp/pipeline.py)

```python
pipe = PipelineHelper(m, stages=3)
a, b = pipe.stage(a, b)   # stage 0→1
a, b = pipe.stage(a, b)   # stage 1→2
```

Methods: `stage(*signals)`, `delay(signal, n=1)`, properties `current_stage`, `stages`.

### `FPOperator` base class

**File:** [`src/amaranth_fp/operator.py`](../src/amaranth_fp/operator.py)

Abstract base extending `Component` with `fmt: FPFormat`, `pipeline_stages: int`, and `pipeline_register(m, signal, stage)`.

---

## Core FP Arithmetic

All operators use internal FloPoCo format. Port widths are `fmt.width` bits.

| Component | File | Inputs | Output | Latency | Algorithm |
|---|---|---|---|---|---|
| `FPAdd` | [`fp_add.py`](../src/amaranth_fp/operators/fp_add.py) | `a`, `b` | `o` | 7 | Single-path adder with LZC normalization |
| `FPSub` | [`fp_sub.py`](../src/amaranth_fp/operators/fp_sub.py) | `a`, `b` | `o` | 7 | Negates b sign, delegates to FPAdd |
| `FPMul` | [`fp_mul.py`](../src/amaranth_fp/operators/fp_mul.py) | `a`, `b` | `o` | 5 | Significand multiply + RNE rounding |
| `FPDiv` | [`fp_div.py`](../src/amaranth_fp/operators/fp_div.py) | `a`, `b` | `o` | 6 | Non-restoring division |
| `FPSqrt` | [`fp_sqrt.py`](../src/amaranth_fp/operators/fp_sqrt.py) | `a` | `o` | 5 | Binary restoring square root |
| `FPFMA` | [`fp_fma.py`](../src/amaranth_fp/operators/fp_fma.py) | `a`, `b`, `c` | `o` | 9 | Fused multiply-add: a×b+c |
| `FPSquare` | [`fp_square.py`](../src/amaranth_fp/operators/fp_square.py) | `a` | `o` | 5 | Delegates to FPMul |
| `FPAbs` | [`fp_abs.py`](../src/amaranth_fp/operators/fp_abs.py) | `a` | `o` | 1 | Clears sign bit |
| `FPNeg` | [`fp_neg.py`](../src/amaranth_fp/operators/fp_neg.py) | `a` | `o` | 1 | Flips sign bit |
| `FPMin` | [`fp_minmax.py`](../src/amaranth_fp/operators/fp_minmax.py) | `a`, `b` | `o` | 3 | Uses FPComparator |
| `FPMax` | [`fp_minmax.py`](../src/amaranth_fp/operators/fp_minmax.py) | `a`, `b` | `o` | 3 | Uses FPComparator |
| `FPComparator` | [`fp_cmp.py`](../src/amaranth_fp/operators/fp_cmp.py) | `a`, `b` | `lt`, `eq`, `gt`, `unordered` | 2 | Sign/magnitude compare |

---

## FP Variants

| Component | File | Description | Latency |
|---|---|---|---|
| `FPAddDualPath` | [`fp_add_dual_path.py`](../src/amaranth_fp/operators/fp_add_dual_path.py) | Dual-path adder (close/far path) for improved frequency | 7 |
| `FPAddSinglePath` | [`fp_add_single_path.py`](../src/amaranth_fp/operators/fp_add_single_path.py) | Explicit single-path adder variant | 7 |
| `FPAddSub` | [`fp_add_sub.py`](../src/amaranth_fp/operators/fp_add_sub.py) | Combined add/subtract with operation select | 7 |
| `FPAdd3Input` | [`fp_add3.py`](../src/amaranth_fp/operators/fp_add3.py) | 3-input adder chaining two FPAdd | 14 |
| `FPMultKaratsuba` | [`fp_mult_karatsuba.py`](../src/amaranth_fp/operators/fp_mult_karatsuba.py) | Karatsuba-based FP multiplier for wide formats | 6 |
| `IEEEFPAdd` | [`ieee_fp_add.py`](../src/amaranth_fp/operators/ieee_fp_add.py) | IEEE-native FP addition | 7 |
| `IEEEFPFMA` | [`ieee_fp_fma.py`](../src/amaranth_fp/operators/ieee_fp_fma.py) | IEEE-native fused multiply-add | 9 |
| `IEEEFPExp` | [`ieee_fp_exp.py`](../src/amaranth_fp/operators/ieee_fp_exp.py) | IEEE-native exponential | 8 |
| `FPDotProduct` | [`fp_dot_product.py`](../src/amaranth_fp/operators/fp_dot_product.py) | N-element dot product via multiply + tree reduction | 5+ceil(log2(n))×7 |
| `FPRealKCM` | [`fp_real_kcm.py`](../src/amaranth_fp/operators/fp_real_kcm.py) | FP × real constant via KCM | 3 |
| `FPSqrtPoly` | [`fp_sqrt_poly.py`](../src/amaranth_fp/operators/fp_sqrt_poly.py) | Polynomial-based FP sqrt | 4 |
| `FPLogIterative` | [`fp_log_iterative.py`](../src/amaranth_fp/operators/fp_log_iterative.py) | Iterative logarithm | 8 |
| `IEEEFloatFormat` | [`ieee_float_format.py`](../src/amaranth_fp/operators/ieee_float_format.py) | IEEE float format descriptor | — |

---

## Transcendentals

| Component | File | Description | Latency |
|---|---|---|---|
| `FPExp` | [`fp_exp.py`](../src/amaranth_fp/operators/fp_exp.py) | Floating-point e^x via table + range reduction | 8 |
| `FPLog` | [`fp_log.py`](../src/amaranth_fp/operators/fp_log.py) | Floating-point ln(x) via table + decomposition | 8 |
| `FPPow` | [`fp_pow.py`](../src/amaranth_fp/operators/fp_pow.py) | x^y = exp(y·log(x)), chains FPLog→FPMul→FPExp | 21 |
| `Exp` | [`exp.py`](../src/amaranth_fp/operators/exp.py) | Generic exponential operator | 8 |
| `FixSinCos` | [`fix_sincos.py`](../src/amaranth_fp/operators/fix_sincos.py) | CORDIC rotation mode sin/cos | width+2 |
| `FixSinCosPoly` | [`fix_sincos_poly.py`](../src/amaranth_fp/operators/fix_sincos_poly.py) | Polynomial-based sin/cos | varies |
| `FixSinCosCORDIC` | [`fix_sincos_cordic.py`](../src/amaranth_fp/operators/fix_sincos_cordic.py) | Explicit CORDIC sin/cos | width+2 |
| `FixSinOrCos` | [`fix_sin_or_cos.py`](../src/amaranth_fp/operators/fix_sin_or_cos.py) | Sin-only or cos-only evaluation | varies |
| `FixSinPoly` | [`fix_sin_poly.py`](../src/amaranth_fp/operators/fix_sin_poly.py) | Polynomial sine approximation | varies |
| `LogSinCos` | [`log_sin_cos.py`](../src/amaranth_fp/operators/log_sin_cos.py) | Logarithmic domain sin/cos | varies |
| `ConstDiv3ForSinPoly` | [`const_div3_for_sin_poly.py`](../src/amaranth_fp/operators/const_div3_for_sin_poly.py) | Constant ÷3 helper for sin polynomial | 1 |
| `FixAtan2` | [`fix_atan2.py`](../src/amaranth_fp/operators/fix_atan2.py) | CORDIC vectoring mode atan2 | width+2 |
| `FixAtan2ByCORDIC` | [`fix_atan2_cordic.py`](../src/amaranth_fp/operators/fix_atan2_cordic.py) | CORDIC-based atan2 | width+2 |
| `FixAtan2ByBivariateApprox` | [`fix_atan2_bivariate.py`](../src/amaranth_fp/operators/fix_atan2_bivariate.py) | Bivariate approximation atan2 | varies |
| `FixAtan2ByRecipMultAtan` | [`fix_atan2_by_recip_mult_atan.py`](../src/amaranth_fp/operators/fix_atan2_by_recip_mult_atan.py) | Reciprocal-multiply atan2 | varies |
| `Atan2Table` | [`atan2_table.py`](../src/amaranth_fp/operators/atan2_table.py) | Table-based atan2 lookup | 1 |

---

## Constant Multipliers

| Component | File | Description | Latency |
|---|---|---|---|
| `FPConstMult` | [`fp_const_mult.py`](../src/amaranth_fp/operators/fp_const_mult.py) | FP × known constant (decomposed at elaboration) | 3 |
| `CRFPConstMult` | [`cr_fp_const_mult.py`](../src/amaranth_fp/operators/cr_fp_const_mult.py) | Correctly-rounded FP constant multiplier | 3 |
| `FPConstDiv` | [`fp_const_div.py`](../src/amaranth_fp/operators/fp_const_div.py) | FP ÷ integer constant via reciprocal multiply | 3 |
| `FixRealKCM` | [`fix_real_kcm.py`](../src/amaranth_fp/operators/fix_real_kcm.py) | Fixed-point × real constant via KCM tables | 2 |
| `FixRealConstMult` | [`fix_real_const_mult.py`](../src/amaranth_fp/operators/fix_real_const_mult.py) | Fixed-point × real constant multiplier | 2 |
| `FixRealShiftAdd` | [`fix_real_shift_add.py`](../src/amaranth_fp/operators/fix_real_shift_add.py) | Shift-and-add constant multiplier | 1 |
| `FixFixConstMult` | [`fix_fix_const_mult.py`](../src/amaranth_fp/operators/fix_fix_const_mult.py) | Fixed × fixed constant multiplier | 1 |
| `FixConstant` | [`fix_constant.py`](../src/amaranth_fp/operators/fix_constant.py) | Fixed-point constant generator | 0 |
| `IntConstMult` | [`int_const_mult.py`](../src/amaranth_fp/operators/int_const_mult.py) | Integer × constant via CSD shift-add | 1 |
| `IntConstMultShiftAdd` | [`int_const_mult_shift_add.py`](../src/amaranth_fp/operators/int_const_mult_shift_add.py) | Shift-add integer constant multiplier | 1 |
| `IntConstDiv` | [`int_const_div.py`](../src/amaranth_fp/operators/int_const_div.py) | Integer ÷ constant | 2 |
| `IntIntKCM` | [`int_int_kcm.py`](../src/amaranth_fp/operators/int_int_kcm.py) | Integer KCM multiplier | 2 |

---

## Integer Arithmetic

**Module:** `amaranth_fp.integer`

| Component | File | Description | Latency |
|---|---|---|---|
| `IntAdder` | [`int_adder.py`](../src/amaranth_fp/integer/int_adder.py) | General integer adder | 1 |
| `IntAddSub` | [`int_add_sub.py`](../src/amaranth_fp/integer/int_add_sub.py) | Combined integer add/subtract | 1 |
| `IntMultiplier` | [`int_multiplier.py`](../src/amaranth_fp/integer/int_multiplier.py) | General integer multiplier | 2 |
| `IntMultiplierLUT` | [`int_multiplier_lut.py`](../src/amaranth_fp/integer/int_multiplier_lut.py) | LUT-based small multiplier | 1 |
| `IntSquarer` | [`int_squarer.py`](../src/amaranth_fp/integer/int_squarer.py) | Integer squaring | 1 |
| `IntComparator` | [`int_comparator.py`](../src/amaranth_fp/integer/int_comparator.py) | Integer comparator | 1 |
| `IntConstComparator` | [`int_const_comparator.py`](../src/amaranth_fp/integer/int_const_comparator.py) | Compare integer against constant | 0 |
| `IntMultiAdder` | [`int_multi_adder.py`](../src/amaranth_fp/integer/int_multi_adder.py) | Multi-operand integer adder | 2 |
| `IntDualAddSub` | [`int_dual_add_sub.py`](../src/amaranth_fp/operators/int_dual_add_sub.py) | Dual add/subtract unit | 1 |
| `BaseMultiplier` | [`base_multiplier.py`](../src/amaranth_fp/integer/base_multiplier.py) | Base multiplier block for tiling | 1 |
| `BaseSquarerLUT` | [`base_squarer_lut.py`](../src/amaranth_fp/integer/base_squarer_lut.py) | LUT-based base squarer | 1 |
| `CarryGenCircuit` | [`carry_gen_circuit.py`](../src/amaranth_fp/integer/carry_gen_circuit.py) | Carry generation logic | 0 |
| `DSPBlock` | [`dsp_block.py`](../src/amaranth_fp/integer/dsp_block.py) | DSP block wrapper | 1 |
| `FixMultAdd` | [`fix_mult_add.py`](../src/amaranth_fp/integer/fix_mult_add.py) | Fixed-point multiply-add | 2 |
| `FixMultiAdder` | [`fix_multi_adder.py`](../src/amaranth_fp/integer/fix_multi_adder.py) | Fixed-point multi-operand adder | 2 |

---

## Complex / FFT

**Module:** `amaranth_fp.complex`

| Component | File | Description | Latency |
|---|---|---|---|
| `FixComplexAdder` | [`fix_complex_adder.py`](../src/amaranth_fp/complex/fix_complex_adder.py) | Fixed-point complex addition | 1 |
| `FixComplexKCM` | [`fix_complex_kcm.py`](../src/amaranth_fp/complex/fix_complex_kcm.py) | Fixed-point complex constant multiplier | 2 |
| `FixComplexMult` | [`fix_complex_mult.py`](../src/amaranth_fp/complex/fix_complex_mult.py) | Fixed-point complex multiplication | 3 |
| `FixFFT` | [`fix_fft.py`](../src/amaranth_fp/complex/fix_fft.py) | Fixed-point FFT | varies |
| `FixFFTFullyPA` | [`fix_fft_fully_pa.py`](../src/amaranth_fp/complex/fix_fft_fully_pa.py) | Fully pipelined FFT | varies |
| `FixR2Butterfly` | [`fix_r2_butterfly.py`](../src/amaranth_fp/complex/fix_r2_butterfly.py) | Radix-2 butterfly | 3 |
| `FPComplexAdder` | [`fp_complex_adder.py`](../src/amaranth_fp/complex/fp_complex_adder.py) | Floating-point complex addition | 7 |
| `FPComplexMult` | [`fp_complex_mult.py`](../src/amaranth_fp/complex/fp_complex_mult.py) | Floating-point complex multiplication | 12 |
| `IntFFT` | [`int_fft.py`](../src/amaranth_fp/complex/int_fft.py) | Integer FFT | varies |
| `IntFFTButterfly` | [`int_fft_butterfly.py`](../src/amaranth_fp/complex/int_fft_butterfly.py) | Integer FFT butterfly | 2 |
| `IntFFTLevelDIT2` | [`int_fft_level_dit2.py`](../src/amaranth_fp/complex/int_fft_level_dit2.py) | DIT-2 FFT level | varies |
| `IntTwiddleMultiplier` | [`int_twiddle_multiplier.py`](../src/amaranth_fp/complex/int_twiddle_multiplier.py) | Integer twiddle factor multiplier | 2 |
| `IntTwiddleMultAlt` | [`int_twiddle_mult_alt.py`](../src/amaranth_fp/complex/int_twiddle_mult_alt.py) | Alternative twiddle multiplier | 2 |

---

## Filters / DSP

**Module:** `amaranth_fp.filters`

| Component | File | Description | Latency |
|---|---|---|---|
| `FixFIR` | [`fix_fir.py`](../src/amaranth_fp/filters/fix_fir.py) | Fixed-point FIR filter | n_taps |
| `FixIIR` | [`fix_iir.py`](../src/amaranth_fp/filters/fix_iir.py) | Fixed-point IIR filter | varies |
| `FixIIRShiftAdd` | [`fix_iir_shift_add.py`](../src/amaranth_fp/filters/fix_iir_shift_add.py) | Shift-add IIR filter | varies |
| `FixSOPC` | [`fix_sopc.py`](../src/amaranth_fp/filters/fix_sopc.py) | Sum of products with constants | 2 |
| `FixHalfSine` | [`fix_half_sine.py`](../src/amaranth_fp/filters/fix_half_sine.py) | Half-sine pulse shaping filter | varies |
| `FixRootRaisedCosine` | [`fix_root_raised_cosine.py`](../src/amaranth_fp/filters/fix_root_raised_cosine.py) | Root-raised-cosine filter | varies |
| `IntFIRTransposed` | [`int_fir_transposed.py`](../src/amaranth_fp/filters/int_fir_transposed.py) | Integer transposed FIR filter | n_taps |

---

## Function Approximation

**Module:** `amaranth_fp.functions`

| Component | File | Description | Latency |
|---|---|---|---|
| `Table` | [`table.py`](../src/amaranth_fp/functions/table.py) | Generic ROM lookup table | 1 |
| `TableOperator` | [`table_operator.py`](../src/amaranth_fp/functions/table_operator.py) | Operator wrapper around Table | 1 |
| `KCMTable` | [`kcm_table.py`](../src/amaranth_fp/functions/kcm_table.py) | KCM coefficient table | 1 |
| `FixFunctionByTable` | [`fix_function_by_table.py`](../src/amaranth_fp/functions/fix_function_by_table.py) | Exhaustive table function evaluation | 1 |
| `FixFunctionByPiecewisePoly` | [`fix_function_by_poly.py`](../src/amaranth_fp/functions/fix_function_by_poly.py) | Piecewise polynomial approximation | degree+2 |
| `FixFunctionBySimplePoly` | [`fix_function_by_simple_poly.py`](../src/amaranth_fp/functions/fix_function_by_simple_poly.py) | Simple polynomial approximation | degree+1 |
| `FixFunctionByVaryingPiecewisePoly` | [`fix_function_by_varying_piecewise_poly.py`](../src/amaranth_fp/functions/fix_function_by_varying_piecewise_poly.py) | Non-uniform segment polynomial | degree+2 |
| `FixFunctionByMultipartite` | [`fix_function_by_multipartite.py`](../src/amaranth_fp/functions/fix_function_by_multipartite.py) | Multipartite table method | 2 |
| `FixHornerEvaluator` | [`fix_horner.py`](../src/amaranth_fp/functions/fix_horner.py) | Pipelined Horner polynomial evaluator | n_coeffs-1 |
| `HOTBM` | [`hotbm.py`](../src/amaranth_fp/functions/hotbm.py) | High-Order Table-Based Method | 2 |
| `Alpha` | [`alpha.py`](../src/amaranth_fp/functions/alpha.py) | Alpha function evaluation | varies |

---

## BitHeap Infrastructure

**Module:** `amaranth_fp.bitheap`

| Component | File | Description |
|---|---|---|
| `Bit` | [`bit.py`](../src/amaranth_fp/bitheap/bit.py) | Single bit in a bit heap |
| `WeightedBit` | [`weighted_bit.py`](../src/amaranth_fp/bitheap/weighted_bit.py) | Bit with column weight |
| `BitHeap` | [`bit_heap.py`](../src/amaranth_fp/bitheap/bit_heap.py) | Bit heap data structure for multi-operand addition |
| `BitHeapSolution` | [`bit_heap_solution.py`](../src/amaranth_fp/bitheap/bit_heap_solution.py) | Solution representation for compressed bit heap |
| `Compressor` | [`compressor.py`](../src/amaranth_fp/bitheap/compressor.py) | Generic compressor (e.g., 3:2, 4:2) |
| `CompressionStrategy` | [`compression_strategy.py`](../src/amaranth_fp/bitheap/compression_strategy.py) | Abstract compression strategy |
| `FirstFittingStrategy` | [`first_fitting_strategy.py`](../src/amaranth_fp/bitheap/first_fitting_strategy.py) | First-fit compressor placement |
| `MaxEfficiencyStrategy` | [`max_efficiency_strategy.py`](../src/amaranth_fp/bitheap/max_efficiency_strategy.py) | Maximum efficiency compressor selection |
| `ParandehAfsharStrategy` | [`parandeh_afshar_strategy.py`](../src/amaranth_fp/bitheap/parandeh_afshar_strategy.py) | Parandeh-Afshar compression strategy |
| `DiffCompressedTable` | [`diff_compressed_table.py`](../src/amaranth_fp/bitheap/diff_compressed_table.py) | Differentially compressed lookup table |
| `DualTable` | [`dual_table.py`](../src/amaranth_fp/bitheap/dual_table.py) | Dual-output lookup table |

---

## Posit Number System

**Module:** `amaranth_fp.posit`

| Component | File | Description | Latency |
|---|---|---|---|
| `PositFormat` | [`posit_format.py`](../src/amaranth_fp/posit/posit_format.py) | Posit format descriptor | — |
| `Posit2FP` | [`posit2fp.py`](../src/amaranth_fp/posit/posit2fp.py) | Posit → floating-point conversion | 3 |
| `Posit2Posit` | [`posit2posit.py`](../src/amaranth_fp/posit/posit2posit.py) | Posit precision conversion | 2 |
| `PositAdd` | [`posit_add.py`](../src/amaranth_fp/posit/posit_add.py) | Posit addition | 5 |
| `PositExp` | [`posit_exp.py`](../src/amaranth_fp/posit/posit_exp.py) | Posit exponential | varies |
| `PositFunction` | [`posit_function.py`](../src/amaranth_fp/posit/posit_function.py) | Generic posit function evaluation | varies |
| `PositFunctionByTable` | [`posit_function_by_table.py`](../src/amaranth_fp/posit/posit_function_by_table.py) | Table-based posit function | 1 |
| `PIFAdd` | [`pif_add.py`](../src/amaranth_fp/posit/pif_add.py) | PIF (Posit Internal Format) addition | 5 |
| `PIF2Fix` | [`pif2fix.py`](../src/amaranth_fp/posit/pif2fix.py) | PIF → fixed-point conversion | 2 |

---

## LNS Operators

Logarithmic Number System: `sign(1) | log_value(width-1)` in fixed-point.

| Component | File | Description | Latency |
|---|---|---|---|
| `LNSMul` | [`lns_ops.py`](../src/amaranth_fp/operators/lns_ops.py) | LNS multiplication (add log values) | 1 |
| `LNSAdd` | [`lns_ops.py`](../src/amaranth_fp/operators/lns_ops.py) | LNS addition via table lookup | 3 |
| `LNSDiv` | [`lns_div.py`](../src/amaranth_fp/operators/lns_div.py) | LNS division (subtract log values) | 1 |
| `LNSSqrt` | [`lns_sqrt.py`](../src/amaranth_fp/operators/lns_sqrt.py) | LNS square root (halve log value) | 1 |
| `LNSAddSub` | [`lns_add_sub.py`](../src/amaranth_fp/operators/lns_add_sub.py) | Combined LNS add/subtract | 3 |
| `Cotran` | [`cotran.py`](../src/amaranth_fp/operators/cotran.py) | Cotransformation function for LNS | 2 |
| `CotranHybrid` | [`cotran.py`](../src/amaranth_fp/operators/cotran.py) | Hybrid cotransformation | 2 |
| `LNSAtanPow` | [`lns_atan_pow.py`](../src/amaranth_fp/operators/lns_atan_pow.py) | LNS atan/power function | varies |
| `LNSLogSinCos` | [`lns_atan_pow.py`](../src/amaranth_fp/operators/lns_atan_pow.py) | LNS-domain sin/cos | varies |

---

## Norms

| Component | File | Description | Latency |
|---|---|---|---|
| `FixNorm` | [`fix_norm.py`](../src/amaranth_fp/operators/fix_norm.py) | Fixed-point 2D/3D vector norm via Newton-Raphson sqrt | 6 |
| `FixNormNaive` | [`fix_norm_naive.py`](../src/amaranth_fp/operators/fix_norm_naive.py) | Naive fixed-point norm | 4 |
| `Fix2DNorm` | [`fix_2d_norm.py`](../src/amaranth_fp/operators/fix_2d_norm.py) | 2D vector norm | 6 |
| `Fix3DNorm` | [`fix_3d_norm.py`](../src/amaranth_fp/operators/fix_3d_norm.py) | 3D vector norm | 6 |
| `Fix2DNormCORDIC` | [`fix_2d_norm_cordic.py`](../src/amaranth_fp/operators/fix_2d_norm_cordic.py) | CORDIC-based 2D norm | width+2 |
| `Fix3DNormCORDIC` | [`fix_3d_norm_cordic.py`](../src/amaranth_fp/operators/fix_3d_norm_cordic.py) | CORDIC-based 3D norm | width+2 |
| `FixSumOfSquares` | [`fix_sum_of_squares.py`](../src/amaranth_fp/operators/fix_sum_of_squares.py) | Sum of squares computation | 3 |

---

## Sorting

**Module:** `amaranth_fp.sorting` + `amaranth_fp.operators`

| Component | File | Description | Latency |
|---|---|---|---|
| `SortingNetwork` | [`sorting_network.py`](../src/amaranth_fp/operators/sorting_network.py) | Parameterized sorting network | depth |
| `BitonicSort` | [`bitonic_sort.py`](../src/amaranth_fp/sorting/bitonic_sort.py) | Bitonic sorting network | log²(n) |
| `OptimalDepthSort` | [`optimal_depth_sort.py`](../src/amaranth_fp/sorting/optimal_depth_sort.py) | Optimal-depth sorting network | optimal |
| `TaoSort` | [`tao_sort.py`](../src/amaranth_fp/operators/tao_sort.py) | Tao sorting network | varies |
| `SortWrapper` | [`sort_wrapper.py`](../src/amaranth_fp/operators/sort_wrapper.py) | Sort wrapper for integration | varies |

---

## Primitive Components

### Generic Primitives

**Module:** `amaranth_fp.primitives`

| Component | File | Description |
|---|---|---|
| `Primitive` | [`primitive.py`](../src/amaranth_fp/primitives/primitive.py) | Abstract primitive base class |
| `GenericLUT` | [`generic_lut.py`](../src/amaranth_fp/primitives/generic_lut.py) | Generic lookup table primitive |
| `GenericMult` | [`generic_mult.py`](../src/amaranth_fp/primitives/generic_mult.py) | Generic multiplier primitive |
| `GenericMux` | [`generic_mux.py`](../src/amaranth_fp/primitives/generic_mux.py) | Generic multiplexer primitive |
| `BooleanEquation` | [`boolean_equation.py`](../src/amaranth_fp/primitives/boolean_equation.py) | Boolean equation primitive |
| `RowAdder` | [`row_adder.py`](../src/amaranth_fp/primitives/row_adder.py) | Row adder primitive |

### Intel FPGA Primitives

**Module:** `amaranth_fp.primitives.intel`

| Component | File | Description |
|---|---|---|
| `IntelLCELL` | [`intel_lcell.py`](../src/amaranth_fp/primitives/intel/intel_lcell.py) | Intel LCELL primitive |
| `IntelRCCM` | [`intel_rccm.py`](../src/amaranth_fp/primitives/intel/intel_rccm.py) | Intel RCCM primitive |
| `IntelTernaryAdder` | [`intel_ternary_adder.py`](../src/amaranth_fp/primitives/intel/intel_ternary_adder.py) | Intel ternary adder primitive |

### Xilinx FPGA Primitives

**Module:** `amaranth_fp.primitives.xilinx`

| Component | File | Description |
|---|---|---|
| `XilinxCarry4` | [`xilinx_carry4.py`](../src/amaranth_fp/primitives/xilinx/xilinx_carry4.py) | Xilinx CARRY4 primitive |
| `XilinxLUT5` | [`xilinx_lut5.py`](../src/amaranth_fp/primitives/xilinx/xilinx_lut5.py) | Xilinx LUT5 primitive |
| `XilinxLUT6` | [`xilinx_lut6.py`](../src/amaranth_fp/primitives/xilinx/xilinx_lut6.py) | Xilinx LUT6 primitive |
| `XilinxMUXF7` | [`xilinx_muxf7.py`](../src/amaranth_fp/primitives/xilinx/xilinx_muxf7.py) | Xilinx MUXF7 primitive |
| `XilinxMUXF8` | [`xilinx_muxf8.py`](../src/amaranth_fp/primitives/xilinx/xilinx_muxf8.py) | Xilinx MUXF8 primitive |
| `XilinxFDCE` | [`xilinx_fdce.py`](../src/amaranth_fp/primitives/xilinx/xilinx_fdce.py) | Xilinx FDCE flip-flop |
| `XilinxCFGLUT5` | [`xilinx_cfglut5.py`](../src/amaranth_fp/primitives/xilinx/xilinx_cfglut5.py) | Xilinx configurable LUT5 |
| `XilinxLookahead8` | [`xilinx_lookahead8.py`](../src/amaranth_fp/primitives/xilinx/xilinx_lookahead8.py) | Xilinx 8-bit lookahead |
| `XilinxGenericMux` | [`xilinx_generic_mux.py`](../src/amaranth_fp/primitives/xilinx/xilinx_generic_mux.py) | Xilinx generic multiplexer |
| `XilinxN2MDecoder` | [`xilinx_n2m_decoder.py`](../src/amaranth_fp/primitives/xilinx/xilinx_n2m_decoder.py) | Xilinx N-to-M decoder |
| `XilinxFourToTwoCompressor` | [`xilinx_four_to_two_compressor.py`](../src/amaranth_fp/primitives/xilinx/xilinx_four_to_two_compressor.py) | Xilinx 4:2 compressor |
| `XilinxTernaryAddSub` | [`xilinx_ternary_add_sub.py`](../src/amaranth_fp/primitives/xilinx/xilinx_ternary_add_sub.py) | Xilinx ternary add/subtract |
| `XilinxGPC` | [`xilinx_gpc.py`](../src/amaranth_fp/primitives/xilinx/xilinx_gpc.py) | Xilinx generalized parallel counter |

---

## Building Blocks

**Module:** `amaranth_fp.building_blocks`

| Component | File | Description | Latency |
|---|---|---|---|
| `Shifter` | [`shifter.py`](../src/amaranth_fp/building_blocks/shifter.py) | Log₂-staged barrel shifter | 0 (comb) |
| `LeadingZeroCounter` | [`lzc.py`](../src/amaranth_fp/building_blocks/lzc.py) | Recursive divide-and-conquer LZC | 0 (comb) |
| `LZOC3` | [`lzoc3.py`](../src/amaranth_fp/building_blocks/lzoc3.py) | 3-input leading zero/one counter | 0 (comb) |
| `Normalizer` | [`normalizer.py`](../src/amaranth_fp/building_blocks/normalizer.py) | LZC + left shift normalization | 0 (comb) |
| `RoundingUnit` | [`rounding.py`](../src/amaranth_fp/building_blocks/rounding.py) | IEEE 754 RNE rounding (guard/round/sticky) | 0 (comb) |
| `OneHotDecoder` | [`one_hot_decoder.py`](../src/amaranth_fp/building_blocks/one_hot_decoder.py) | One-hot decoder | 0 (comb) |
| `ThermometerDecoder` | [`thermometer_decoder.py`](../src/amaranth_fp/building_blocks/thermometer_decoder.py) | Thermometer code decoder | 0 (comb) |

Additional operators in `amaranth_fp.operators`:

| Component | File | Description | Latency |
|---|---|---|---|
| `ShiftReg` | [`shift_reg.py`](../src/amaranth_fp/operators/shift_reg.py) | Shift register | n |
| `FixResize` | [`fix_resize.py`](../src/amaranth_fp/operators/fix_resize.py) | Fixed-point resize/truncate | 0 |

---

## Format Conversions

**Module:** `amaranth_fp.conversions`

| Component | File | Description | Latency |
|---|---|---|---|
| `InputIEEE` | [`input_ieee.py`](../src/amaranth_fp/conversions/input_ieee.py) | IEEE 754 → FloPoCo internal format | 1 |
| `OutputIEEE` | [`output_ieee.py`](../src/amaranth_fp/conversions/output_ieee.py) | FloPoCo internal → IEEE 754 | 1 |
| `Fix2FP` | [`fix2fp.py`](../src/amaranth_fp/conversions/fix2fp.py) | Fixed-point → floating-point | 3 |
| `FP2Fix` | [`fp2fix.py`](../src/amaranth_fp/conversions/fp2fix.py) | Floating-point → fixed-point | 3 |
| `FPResize` | [`fp_resize.py`](../src/amaranth_fp/conversions/fp_resize.py) | FP precision conversion with rounding | 2 |
| `PIF2Posit` | [`pif2posit.py`](../src/amaranth_fp/conversions/pif2posit.py) | PIF → posit encoding | 2 |
| `Posit2PIF` | [`posit2pif.py`](../src/amaranth_fp/conversions/posit2pif.py) | Posit → PIF decoding | 2 |

---

## DAG Operator Composition

**Module:** `amaranth_fp.dag`

| Component | File | Description |
|---|---|---|
| `DAGOperator` | [`dag_operator.py`](../src/amaranth_fp/dag/dag_operator.py) | DAG-based operator composition for building complex dataflow graphs from primitive operators |

Allows defining operator graphs as directed acyclic graphs, automatically handling signal routing and pipeline alignment between connected operators.

---

## Testing Infrastructure

**Module:** `amaranth_fp.testing`

| Component | File | Description |
|---|---|---|
| `FPNumber` | [`fp_number.py`](../src/amaranth_fp/testing/fp_number.py) | FloPoCo internal format number helper for encoding/decoding test values |
| `IEEENumber` | [`ieee_number.py`](../src/amaranth_fp/testing/ieee_number.py) | IEEE 754 number helper for test value encode/decode |
| `PositNumber` | [`posit_number.py`](../src/amaranth_fp/testing/posit_number.py) | Posit number helper for test value encode/decode |
| `RegisterSandwich` | `register_sandwich.py` | Test harness wrapping operators with input/output registers |
| `TestBench` | `test_bench.py` | Reusable test bench utilities for Amaranth simulation |

---

## Testing Guide

### Running Tests

```bash
pdm run pytest           # run all tests
pdm run pytest -v        # verbose output
pdm run pytest tests/test_operators.py  # specific file
```

### Writing a New Test

```python
from amaranth.sim import Simulator
from amaranth_fp.format import FPFormat
from amaranth_fp.operators import FPMul
from conftest import fp_normal, decode_exc, decode_exp, decode_mant

FMT = FPFormat.half()

def test_my_mul():
    dut = FPMul(FMT)
    async def bench(ctx):
        ctx.set(dut.a, fp_normal(FMT, 0, 16, 0))       # 2.0
        ctx.set(dut.b, fp_normal(FMT, 0, 16, 0b1000000000))  # 3.0
        for _ in range(dut.latency):
            await ctx.tick()
        result = ctx.get(dut.o)
        assert decode_exc(FMT, result) == 0b01   # normal

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(bench)
    with sim.write_vcd("test_my_mul.vcd"):
        sim.run()
```

---

## Complete API Summary Table

| # | Component | Module | Latency |
|---|---|---|---|
| | **Core FP Arithmetic** | | |
| 1 | `FPAdd` | `operators` | 7 |
| 2 | `FPSub` | `operators` | 7 |
| 3 | `FPMul` | `operators` | 5 |
| 4 | `FPDiv` | `operators` | 6 |
| 5 | `FPSqrt` | `operators` | 5 |
| 6 | `FPFMA` | `operators` | 9 |
| 7 | `FPSquare` | `operators` | 5 |
| 8 | `FPAbs` | `operators` | 1 |
| 9 | `FPNeg` | `operators` | 1 |
| 10 | `FPMin` | `operators` | 3 |
| 11 | `FPMax` | `operators` | 3 |
| 12 | `FPComparator` | `operators` | 2 |
| | **FP Variants** | | |
| 13 | `FPAddDualPath` | `operators` | 7 |
| 14 | `FPAddSinglePath` | `operators` | 7 |
| 15 | `FPAddSub` | `operators` | 7 |
| 16 | `FPAdd3Input` | `operators` | 14 |
| 17 | `FPMultKaratsuba` | `operators` | 6 |
| 18 | `IEEEFPAdd` | `operators` | 7 |
| 19 | `IEEEFPFMA` | `operators` | 9 |
| 20 | `IEEEFPExp` | `operators` | 8 |
| 21 | `FPDotProduct` | `operators` | 5+ceil(log2(n))×7 |
| 22 | `FPRealKCM` | `operators` | 3 |
| 23 | `FPSqrtPoly` | `operators` | 4 |
| 24 | `FPLogIterative` | `operators` | 8 |
| 25 | `IEEEFloatFormat` | `operators` | — |
| | **Transcendentals** | | |
| 26 | `FPExp` | `operators` | 8 |
| 27 | `FPLog` | `operators` | 8 |
| 28 | `FPPow` | `operators` | 21 |
| 29 | `Exp` | `operators` | 8 |
| 30 | `FixSinCos` | `operators` | width+2 |
| 31 | `FixSinCosPoly` | `operators` | varies |
| 32 | `FixSinCosCORDIC` | `operators` | width+2 |
| 33 | `FixSinOrCos` | `operators` | varies |
| 34 | `FixSinPoly` | `operators` | varies |
| 35 | `LogSinCos` | `operators` | varies |
| 36 | `ConstDiv3ForSinPoly` | `operators` | 1 |
| 37 | `FixAtan2` | `operators` | width+2 |
| 38 | `FixAtan2ByCORDIC` | `operators` | width+2 |
| 39 | `FixAtan2ByBivariateApprox` | `operators` | varies |
| 40 | `FixAtan2ByRecipMultAtan` | `operators` | varies |
| 41 | `Atan2Table` | `operators` | 1 |
| | **Constant Multipliers** | | |
| 42 | `FPConstMult` | `operators` | 3 |
| 43 | `CRFPConstMult` | `operators` | 3 |
| 44 | `FPConstDiv` | `operators` | 3 |
| 45 | `FixRealKCM` | `operators` | 2 |
| 46 | `FixRealConstMult` | `operators` | 2 |
| 47 | `FixRealShiftAdd` | `operators` | 1 |
| 48 | `FixFixConstMult` | `operators` | 1 |
| 49 | `FixConstant` | `operators` | 0 |
| 50 | `IntConstMult` | `operators` | 1 |
| 51 | `IntConstMultShiftAdd` | `operators` | 1 |
| 52 | `IntConstDiv` | `operators` | 2 |
| 53 | `IntIntKCM` | `operators` | 2 |
| | **Integer Arithmetic** | | |
| 54 | `IntAdder` | `integer` | 1 |
| 55 | `IntAddSub` | `integer` | 1 |
| 56 | `IntMultiplier` | `integer` | 2 |
| 57 | `IntMultiplierLUT` | `integer` | 1 |
| 58 | `IntSquarer` | `integer` | 1 |
| 59 | `IntComparator` | `integer` | 1 |
| 60 | `IntConstComparator` | `integer` | 0 |
| 61 | `IntMultiAdder` | `integer` | 2 |
| 62 | `IntDualAddSub` | `operators` | 1 |
| 63 | `BaseMultiplier` | `integer` | 1 |
| 64 | `BaseSquarerLUT` | `integer` | 1 |
| 65 | `CarryGenCircuit` | `integer` | 0 |
| 66 | `DSPBlock` | `integer` | 1 |
| 67 | `FixMultAdd` | `integer` | 2 |
| 68 | `FixMultiAdder` | `integer` | 2 |
| | **Complex / FFT** | | |
| 69 | `FixComplexAdder` | `complex` | 1 |
| 70 | `FixComplexKCM` | `complex` | 2 |
| 71 | `FixComplexMult` | `complex` | 3 |
| 72 | `FixFFT` | `complex` | varies |
| 73 | `FixFFTFullyPA` | `complex` | varies |
| 74 | `FixR2Butterfly` | `complex` | 3 |
| 75 | `FPComplexAdder` | `complex` | 7 |
| 76 | `FPComplexMult` | `complex` | 12 |
| 77 | `IntFFT` | `complex` | varies |
| 78 | `IntFFTButterfly` | `complex` | 2 |
| 79 | `IntFFTLevelDIT2` | `complex` | varies |
| 80 | `IntTwiddleMultiplier` | `complex` | 2 |
| 81 | `IntTwiddleMultAlt` | `complex` | 2 |
| | **Filters / DSP** | | |
| 82 | `FixFIR` | `filters` | n_taps |
| 83 | `FixIIR` | `filters` | varies |
| 84 | `FixIIRShiftAdd` | `filters` | varies |
| 85 | `FixSOPC` | `filters` | 2 |
| 86 | `FixHalfSine` | `filters` | varies |
| 87 | `FixRootRaisedCosine` | `filters` | varies |
| 88 | `IntFIRTransposed` | `filters` | n_taps |
| | **Function Approximation** | | |
| 89 | `Table` | `functions` | 1 |
| 90 | `TableOperator` | `functions` | 1 |
| 91 | `KCMTable` | `functions` | 1 |
| 92 | `FixFunctionByTable` | `functions` | 1 |
| 93 | `FixFunctionByPiecewisePoly` | `functions` | degree+2 |
| 94 | `FixFunctionBySimplePoly` | `functions` | degree+1 |
| 95 | `FixFunctionByVaryingPiecewisePoly` | `functions` | degree+2 |
| 96 | `FixFunctionByMultipartite` | `functions` | 2 |
| 97 | `FixHornerEvaluator` | `functions` | n_coeffs-1 |
| 98 | `HOTBM` | `functions` | 2 |
| 99 | `Alpha` | `functions` | varies |
| | **BitHeap Infrastructure** | | |
| 100 | `Bit` | `bitheap` | — |
| 101 | `WeightedBit` | `bitheap` | — |
| 102 | `BitHeap` | `bitheap` | — |
| 103 | `BitHeapSolution` | `bitheap` | — |
| 104 | `Compressor` | `bitheap` | — |
| 105 | `CompressionStrategy` | `bitheap` | — |
| 106 | `FirstFittingStrategy` | `bitheap` | — |
| 107 | `MaxEfficiencyStrategy` | `bitheap` | — |
| 108 | `ParandehAfsharStrategy` | `bitheap` | — |
| 109 | `DiffCompressedTable` | `bitheap` | — |
| 110 | `DualTable` | `bitheap` | — |
| | **Posit Number System** | | |
| 111 | `PositFormat` | `posit` | — |
| 112 | `Posit2FP` | `posit` | 3 |
| 113 | `Posit2Posit` | `posit` | 2 |
| 114 | `PositAdd` | `posit` | 5 |
| 115 | `PositExp` | `posit` | varies |
| 116 | `PositFunction` | `posit` | varies |
| 117 | `PositFunctionByTable` | `posit` | 1 |
| 118 | `PIFAdd` | `posit` | 5 |
| 119 | `PIF2Fix` | `posit` | 2 |
| | **LNS Operators** | | |
| 120 | `LNSMul` | `operators` | 1 |
| 121 | `LNSAdd` | `operators` | 3 |
| 122 | `LNSDiv` | `operators` | 1 |
| 123 | `LNSSqrt` | `operators` | 1 |
| 124 | `LNSAddSub` | `operators` | 3 |
| 125 | `Cotran` | `operators` | 2 |
| 126 | `CotranHybrid` | `operators` | 2 |
| 127 | `LNSAtanPow` | `operators` | varies |
| 128 | `LNSLogSinCos` | `operators` | varies |
| | **Norms** | | |
| 129 | `FixNorm` | `operators` | 6 |
| 130 | `FixNormNaive` | `operators` | 4 |
| 131 | `Fix2DNorm` | `operators` | 6 |
| 132 | `Fix3DNorm` | `operators` | 6 |
| 133 | `Fix2DNormCORDIC` | `operators` | width+2 |
| 134 | `Fix3DNormCORDIC` | `operators` | width+2 |
| 135 | `FixSumOfSquares` | `operators` | 3 |
| | **Sorting** | | |
| 136 | `SortingNetwork` | `operators` | depth |
| 137 | `BitonicSort` | `sorting` | log²(n) |
| 138 | `OptimalDepthSort` | `sorting` | optimal |
| 139 | `TaoSort` | `operators` | varies |
| 140 | `SortWrapper` | `operators` | varies |
| | **Primitive Components — Generic** | | |
| 141 | `Primitive` | `primitives` | — |
| 142 | `GenericLUT` | `primitives` | — |
| 143 | `GenericMult` | `primitives` | — |
| 144 | `GenericMux` | `primitives` | — |
| 145 | `BooleanEquation` | `primitives` | — |
| 146 | `RowAdder` | `primitives` | — |
| | **Primitive Components — Intel** | | |
| 147 | `IntelLCELL` | `primitives.intel` | — |
| 148 | `IntelRCCM` | `primitives.intel` | — |
| 149 | `IntelTernaryAdder` | `primitives.intel` | — |
| | **Primitive Components — Xilinx** | | |
| 150 | `XilinxCarry4` | `primitives.xilinx` | — |
| 151 | `XilinxLUT5` | `primitives.xilinx` | — |
| 152 | `XilinxLUT6` | `primitives.xilinx` | — |
| 153 | `XilinxMUXF7` | `primitives.xilinx` | — |
| 154 | `XilinxMUXF8` | `primitives.xilinx` | — |
| 155 | `XilinxFDCE` | `primitives.xilinx` | — |
| 156 | `XilinxCFGLUT5` | `primitives.xilinx` | — |
| 157 | `XilinxLookahead8` | `primitives.xilinx` | — |
| 158 | `XilinxGenericMux` | `primitives.xilinx` | — |
| 159 | `XilinxN2MDecoder` | `primitives.xilinx` | — |
| 160 | `XilinxFourToTwoCompressor` | `primitives.xilinx` | — |
| 161 | `XilinxTernaryAddSub` | `primitives.xilinx` | — |
| 162 | `XilinxGPC` | `primitives.xilinx` | — |
| | **Building Blocks** | | |
| 163 | `Shifter` | `building_blocks` | 0 (comb) |
| 164 | `LeadingZeroCounter` | `building_blocks` | 0 (comb) |
| 165 | `LZOC3` | `building_blocks` | 0 (comb) |
| 166 | `Normalizer` | `building_blocks` | 0 (comb) |
| 167 | `RoundingUnit` | `building_blocks` | 0 (comb) |
| 168 | `OneHotDecoder` | `building_blocks` | 0 (comb) |
| 169 | `ThermometerDecoder` | `building_blocks` | 0 (comb) |
| 170 | `ShiftReg` | `operators` | n |
| 171 | `FixResize` | `operators` | 0 |
| | **Format Conversions** | | |
| 172 | `InputIEEE` | `conversions` | 1 |
| 173 | `OutputIEEE` | `conversions` | 1 |
| 174 | `Fix2FP` | `conversions` | 3 |
| 175 | `FP2Fix` | `conversions` | 3 |
| 176 | `FPResize` | `conversions` | 2 |
| 177 | `PIF2Posit` | `conversions` | 2 |
| 178 | `Posit2PIF` | `conversions` | 2 |
| | **DAG** | | |
| 179 | `DAGOperator` | `dag` | — |
| | **Testing** | | |
| 180 | `FPNumber` | `testing` | — |
| 181 | `IEEENumber` | `testing` | — |
| 182 | `PositNumber` | `testing` | — |
| 183 | `RegisterSandwich` | `testing` | — |
| 184 | `TestBench` | `testing` | — |
| | **Core Infrastructure** | | |
| 185 | `FPFormat` | `amaranth_fp` | — |
| 186 | `FPOperator` | `amaranth_fp` | — |
| 187 | `PipelineHelper` | `amaranth_fp` | — |
| 188 | `PipelinedComponent` | `amaranth_fp` | — |
| 189 | `FPGATarget` | `amaranth_fp` | — |
| 190 | `ieee_layout` | `amaranth_fp` | — |
| 191 | `internal_layout` | `amaranth_fp` | — |

**Total: 191 components across 14 modules.**
