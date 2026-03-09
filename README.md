# amaranth-fp

**Floating-point operator generator for [Amaranth HDL](https://amaranth-lang.org/), inspired by [FloPoCo](http://flopoco.org/).**

Generate pipelined, parameterized floating-point hardware operators at any precision — from half (16-bit) to double (64-bit) and beyond.

## Features

- **215+ pipelined components** covering arithmetic, transcendentals, ML activations, FFT, filters, and more
- **Configurable precision** — `FPFormat.half()`, `FPFormat.single()`, `FPFormat.double()`, or `FPFormat.custom(we, wf)`
- **Automatic pipelining** via `PipelinedComponent` with latency tracking
- **Sollya integration** for certified polynomial approximation coefficients
- **561 tests** verified against Sollya/mpmath golden reference
- **FPGA target models** ported from FloPoCo (Kintex-7, Virtex UltraScale+, StratixV, etc.)

## Installation

```bash
# Clone
git clone https://github.com/key2/amaranth-fp.git
cd amaranth-fp

# Install with PDM
pdm install

# Run tests
pdm run pytest tests/ -v
```

### Sollya Support (recommended)

For certified polynomial coefficients and provably correct test reference values, install [PythonSollya](https://github.com/key2/pythonsollya) (our modified fork with Python 3.13 support):

```bash
# Install Sollya C library first
brew install sollya  # macOS
# or: apt install libsollya-dev  # Debian/Ubuntu

# Install our modified PythonSollya
pip install git+https://github.com/key2/pythonsollya.git

# Or install amaranth-fp with Sollya support
pdm install -G sollya
```

> **Note:** The standard PythonSollya from `gitlab.com/metalibm-dev/pythonsollya` does not support Python 3.13. Our fork at `github.com/key2/pythonsollya` includes fixes for modern Python compatibility.

## Quick Start

```python
from amaranth import *
from amaranth_fp.format import FPFormat
from amaranth_fp.operators import FPAdd, FPMul
from amaranth_fp.conversions import InputIEEE, OutputIEEE

# Create a half-precision adder
fmt = FPFormat.half()  # 5-bit exponent, 10-bit mantissa
adder = FPAdd(fmt)
print(f"FPAdd latency: {adder.latency} cycles")  # 7

# Full pipeline: IEEE input → Add → IEEE output
class FPAddPipeline(Elaboratable):
    def elaborate(self, platform):
        m = Module()
        m.submodules.in_a = in_a = InputIEEE(fmt)
        m.submodules.in_b = in_b = InputIEEE(fmt)
        m.submodules.add = add = FPAdd(fmt)
        m.submodules.out = out = OutputIEEE(fmt)
        m.d.comb += [
            add.a.eq(in_a.fp_out),
            add.b.eq(in_b.fp_out),
            out.fp_in.eq(add.o),
        ]
        return m
```

## Component Catalog

| Category | Count | Examples |
|----------|-------|---------|
| **Core Arithmetic** | 12 | FPAdd, FPSub, FPMul, FPDiv, FPSqrt, FPFMA |
| **Transcendentals** | 5 | FPExp, FPLog, FPPow, FixSinCos, FixAtan2 |
| **Math Functions** | 18 | FPExp2, FPLog2, FPAsin, FPAtan, FPTanh, FPErf, FPCbrt, FPReciprocal, FPRsqrt |
| **ML Activations** | 6 | FPSigmoid, FPGELU, FPSoftplus, FPSwish, FPMish, FPSinc |
| **Constant Mult** | 6 | FPConstMult, FixRealKCM, IntConstMult |
| **Integer** | 14 | IntAdder, IntMultiplier, IntComparator, DSPBlock |
| **Complex/FFT** | 13 | FixComplexMult, FixFFT, R2Butterfly |
| **Filters** | 7 | FixFIR, FixIIR, FixSOPC |
| **Function Approx** | 11 | Table, FixHorner, PiecewisePoly, Multipartite |
| **BitHeap** | 11 | BitHeap, Compressor, DiffCompressedTable |
| **Conversions** | 7 | InputIEEE, OutputIEEE, Fix2FP, FP2Fix, FPResize |
| **Posit** | 9 | PositFormat, Posit2FP, PositAdd |
| **Primitives** | 22 | GenericLut, Xilinx CARRY4/LUT6, Intel LCELL |
| **Building Blocks** | 9 | Shifter, LZC, Normalizer, RoundingUnit, BranchMux |

See [`docs/README.md`](docs/README.md) for the complete API reference with 215+ components.

## Documentation

- [`docs/README.md`](docs/README.md) — Full API reference
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Design architecture
- [`docs/FLOPOCO_ANALYSIS.md`](docs/FLOPOCO_ANALYSIS.md) — FloPoCo codebase analysis
- [`docs/SOLLYA_ANALYSIS.md`](docs/SOLLYA_ANALYSIS.md) — Sollya/PythonSollya analysis
- [`docs/SOLLYA_TRIG_FUNCTIONS.md`](docs/SOLLYA_TRIG_FUNCTIONS.md) — Sollya-generated function design

## Testing

```bash
pdm run pytest tests/ -v          # Run all 561 tests
pdm run pytest tests/ -k "hardware"  # Hardware verification only
pdm run pytest tests/ -k "sollya"    # Sollya reference tests only
```

## License

BSD-2-Clause

## Acknowledgments

- [FloPoCo](http://flopoco.org/) — the C++ FP operator generator this project is inspired by
- [Amaranth HDL](https://amaranth-lang.org/) — the Python HDL framework
- [Sollya](https://www.sollya.org/) — certified polynomial approximation tool
- [PythonSollya](https://github.com/key2/pythonsollya) — Python bindings for Sollya (modified fork)
