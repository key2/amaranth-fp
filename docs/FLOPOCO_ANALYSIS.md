# FloPoCo Deep Analysis for Python/Amaranth Reimplementation

## 1. Executive Summary

**FloPoCo** (Floating-Point Cores for FPGAs) is a C++ framework that generates VHDL arithmetic operators optimized for FPGAs. Created at ENS-Lyon/INRIA, it produces pipelined, correctly-rounded floating-point and fixed-point operators with parametric precision.

### Core Design Philosophy

1. **Parametric precision**: All operators are parameterized by `(wE, wF)` — exponent and fraction widths are not limited to IEEE standard sizes.
2. **Automatic pipelining**: Operators declare signal delays; the framework automatically inserts pipeline registers to meet a target frequency.
3. **VHDL generation via C++ constructors**: Each operator's constructor writes VHDL into a `FlopocoStream` using `vhdl << ...` statements. The framework parses this VHDL to build a signal dependency graph.
4. **Composition via subcomponents**: Operators instantiate other operators as subcomponents using [`newInstance()`](flopoco/code/VHDLOperators/include/flopoco/Operator.hpp:440) or [`newSharedInstance()`](flopoco/code/VHDLOperators/include/flopoco/Operator.hpp:458).
5. **Internal FP format**: FloPoCo uses a custom format (2-bit exception + sign + exponent + fraction) that simplifies hardware, with IEEE conversion at boundaries.

---

## 2. Framework Architecture

### 2.1 The [`Operator`](flopoco/code/VHDLOperators/include/flopoco/Operator.hpp) Base Class

Every FloPoCo hardware block inherits from `Operator`. This class manages:

#### Signal Management
- **I/O declaration**: [`addInput()`](flopoco/code/VHDLOperators/include/flopoco/Operator.hpp:108), [`addOutput()`](flopoco/code/VHDLOperators/include/flopoco/Operator.hpp:127), [`addFPInput()`](flopoco/code/VHDLOperators/include/flopoco/Operator.hpp:161), [`addFPOutput()`](flopoco/code/VHDLOperators/include/flopoco/Operator.hpp:174), [`addIEEEInput()`](flopoco/code/VHDLOperators/include/flopoco/Operator.hpp:185), [`addIEEEOutput()`](flopoco/code/VHDLOperators/include/flopoco/Operator.hpp:196)
- **Internal signal declaration**: [`declare()`](flopoco/code/VHDLOperators/include/flopoco/Operator.hpp:247) — declares wires with optional delay annotations
- **Fixed-point signals**: [`declareFixPoint()`](flopoco/code/VHDLOperators/include/flopoco/Operator.hpp:294)
- **Floating-point signals**: [`declareFloatingPoint()`](flopoco/code/VHDLOperators/include/flopoco/Operator.hpp:316)

The key pattern for `declare()`:
```cpp
// Declare a signal with its critical path contribution
declare(getTarget()->adderDelay(wE+2), "expSum", wE+2)
// Declare a wire without delay
declare("fracX", wF+1)
```

#### Subcomponent Instantiation
Two approaches:

1. **`newInstance()`** — creates a new operator from its registered name + parameter string:
```cpp
newInstance("IntMultiplier", "SignificandMultiplication",
    "wX=" + to_string(wFX_+1) + " wY=" + to_string(wFY_+1) + " wOut=" + to_string(sigProdSize),
    "X=>sigX,Y=>sigY",   // input port maps
    "R=>sigProd");        // output port maps
```

2. **`newSharedInstance()`** — reuses an already-constructed operator:
```cpp
newSharedInstance(selfunctiontable, "SelFunctionTable1",
    "X=>sel1", "Y=>q1");
```

#### VHDL Generation
- The `vhdl` member ([`FlopocoStream`](flopoco/code/VHDLOperators/include/flopoco/FlopocoStream.hpp)) captures all VHDL written by the constructor
- On each `;` character, the stream triggers lexing + dependency extraction
- Signal names on LHS are annotated `??name??`, RHS with `$$name$$`
- After construction, [`applySchedule()`](flopoco/code/VHDLOperators/include/flopoco/Operator.hpp:80) resolves timing and inserts pipeline registers

#### Key Data Structures
```cpp
vector<Operator*> subComponentList_;  // instantiated sub-components
vector<Signal*>   signalList_;        // internal signals
vector<Signal*>   ioList_;            // I/O signals
FlopocoStream     vhdl;              // the VHDL code stream
map<string, Signal*> signalMap_;     // signal lookup by name
Target* target_;                     // FPGA target model
Operator* parentOp_;                 // parent operator
```

### 2.2 The [`Signal`](flopoco/code/VHDLOperators/include/flopoco/Signal.hpp) Class

Represents a VHDL signal with:
- **Type**: `in`, `out`, `wire`, `constant`
- **Width**: bit width
- **Timing**: `cycle_` (pipeline stage) and `criticalPath_` (delay within cycle)
- **Critical path contribution**: `criticalPathContribution_` — the delay this signal adds
- **Dependency graph**: `predecessors_` and `successors_` lists with optional cycle delays
- **Format metadata**: `isFP_`, `isFix_`, `isIEEE_`, `wE_`, `wF_`, `MSB_`, `LSB_`, `isSigned_`

### 2.3 The [`FlopocoStream`](flopoco/code/VHDLOperators/include/flopoco/FlopocoStream.hpp)

A specialized `ostringstream` that:
1. Buffers VHDL code in `vhdlCodeBuffer`
2. On each `;`, calls [`flushAndParseAndBuildDependencyTable()`](flopoco/code/VHDLOperators/include/flopoco/FlopocoStream.hpp:148)
3. Uses a VHDL lexer to identify signal names, annotate them, and build a dependency table
4. The dependency table maps `(lhsSignal, rhsSignal, delay)` triplets
5. A second-level pass (during `applySchedule()`) replaces annotated names with cycle-qualified versions

### 2.4 The [`Target`](flopoco/code/HWTargets/include/flopoco/Target.hpp) Class

Abstract class modeling FPGA timing. Key delay methods (all return seconds):

| Method | Purpose |
|--------|---------|
| `lutDelay()` | Bare LUT delay |
| `logicDelay(n)` | Logic function of n inputs (LUT + local routing) |
| `adderDelay(n)` | n-bit addition |
| `adder3Delay(n)` | n-bit ternary addition |
| `eqComparatorDelay(n)` | n-bit equality comparison |
| `ltComparatorDelay(n)` | n-bit less-than comparison |
| `eqConstComparatorDelay(n)` | n-bit comparison with constant |
| `fanoutDelay(n)` | Delay for fanout of n |
| `DSPMultiplierDelay()` | DSP block multiplier delay |
| `tableDelay(wIn, wOut, isLogic)` | Lookup table delay |

Key target attributes:
- `frequency_`: desired clock frequency in Hz
- `lutInputs_`: LUT input count (4, 5, or 6)
- `possibleDSPConfig_`: available DSP multiplier sizes (e.g., 25×18 for Virtex-6)
- `useHardMultipliers_`: whether to use DSP blocks

### 2.5 The [`DAGOperator`](flopoco/code/VHDLOperators/include/flopoco/DAGOperator.hpp)

A special operator that parses a `.dag` file to compose existing operators into a dataflow graph. It:
- Parses component declarations, instances, and signal connections
- Performs type inference on signal widths
- Builds FloPoCo operators for each node and connects them

---

## 3. FP Number Formats

### 3.1 FloPoCo Internal Format

Total width = `wE + wF + 3` bits:

```
[exn(1:0)] [sign] [exponent(wE-1:0)] [fraction(wF-1:0)]
```

| Field | Bits | Description |
|-------|------|-------------|
| `exn` | 2 | Exception: `00`=zero, `01`=normal, `10`=infinity, `11`=NaN |
| `sign` | 1 | Sign bit |
| `exponent` | wE | Biased exponent (bias = 2^(wE-1) - 1) |
| `fraction` | wF | Fractional part (implicit leading 1 for normal numbers) |

Key advantage over IEEE: The 2-bit exception field eliminates the need to detect special exponent patterns (all-zeros for subnormals, all-ones for inf/NaN), simplifying datapath logic.

FloPoCo has **one more exponent value** than IEEE: the exponent field `000...0` encodes `emin - 1`, allowing some IEEE subnormals to be represented as normal numbers.

### 3.2 IEEE 754 Format

Total width = `wE + wF + 1` bits:

```
[sign] [exponent(wE-1:0)] [fraction(wF-1:0)]
```

Standard sizes defined in [`IEEEFloatFormat`](flopoco/code/VHDLOperators/src/IEEEFP/IEEEFloatFormat.cpp):
- binary16: wE=5, wF=10
- binary32: wE=8, wF=23
- binary64: wE=11, wF=52
- binary128: wE=15, wF=112
- binary256: wE=19, wF=236

### 3.3 [`InputIEEE`](flopoco/code/VHDLOperators/src/Conversions/InputIEEE.cpp): IEEE → FloPoCo

Algorithm (pseudocode for `wEI == wEO` case):
```
expX = X[wEI+wFI-1 : wFI]
fracX = X[wFI-1 : 0]
sX = X[wEI+wFI]

expZero = (expX == 0)
expInfty = (expX == all-ones)
fracZero = (fracX == 0)

# FloPoCo can represent subnormals whose mantissa starts with 1
reprSubNormal = fracX[wFI-1]
sfracX = (fracX << 1) if (expZero and reprSubNormal) else fracX

# Round fraction if wFO < wFI (round-to-nearest-even)
if wFO < wFI:
    round = roundBit & (sticky | resultLSB)
    expfracR = (expX & sfracX[top bits]) + round
    
# Build exception field
exnR = "00" if zero
        "10" if infinity  
        "11" if NaN
        "01" otherwise (normal)

R = exnR & sX & expR & fracR
```

For `wEI > wEO` (range downgrading): detects overflow/underflow by comparing exponent against thresholds.

For `wEI < wEO` (range widening): zero-extends exponent with bias correction. Subnormals are flushed to zero (TODO in FloPoCo).

### 3.4 [`OutputIEEE`](flopoco/code/VHDLOperators/src/Conversions/OutputIEEE.cpp): FloPoCo → IEEE

Reverses the conversion:
- Extracts `exnX` (exception bits), `expX`, `fracX`, `sX`
- Maps FloPoCo's exponent field 0...0 (subnormal representation) back to IEEE subnormal format by prepending implicit 1 and right-shifting
- Handles rounding when `wFO < wFI`
- Forces exponent to all-zeros for zero, all-ones for inf/NaN
- For `onlyPositiveZeroes` mode: normalizes -0 to +0

---

## 4. Pipelining System

### 4.1 Philosophy (from [`pipelining.org`](flopoco/Attic/pipelining.org))

FloPoCo uses **on-the-fly ASAP scheduling**:

1. Each `declare()` call annotates a signal with its **critical path contribution** (delay in seconds)
2. After each VHDL statement (triggered by `;`), the lexer/parser extracts dependencies
3. The scheduler computes timing for new signals based on predecessor timing + delay
4. When accumulated delay exceeds `1/frequency`, a new pipeline stage is created
5. During `applySchedule()`, signal references are renamed to their cycle-qualified versions (e.g., `sigName` → `sigName_d3`)

### 4.2 Signal Timing Model

Each signal has:
- **`cycle_`**: the pipeline stage (0 = combinatorial with inputs)
- **`criticalPath_`**: accumulated delay within the current cycle
- **`criticalPathContribution_`**: the delay this signal adds

Scheduling pseudocode:
```
for each signal s with all predecessors scheduled:
    maxCycle = max(pred.cycle for pred in s.predecessors)
    maxCP = max(pred.criticalPath for preds at maxCycle)
    
    if maxCP + s.criticalPathContribution > 1/frequency:
        s.cycle = maxCycle + 1
        s.criticalPath = s.criticalPathContribution
    else:
        s.cycle = maxCycle
        s.criticalPath = maxCP + s.criticalPathContribution
```

### 4.3 Pipeline Register Generation

[`buildVHDLRegisters()`](flopoco/code/VHDLOperators/include/flopoco/Operator.hpp:495) generates register chains. Each signal with `lifeSpan > 0` gets delayed copies:
```vhdl
-- For signal "foo" active at cycle 2, used at cycle 5:
signal foo_d1, foo_d2, foo_d3 : std_logic_vector(...);
process(clk) begin
    if rising_edge(clk) then
        foo_d1 <= foo;
        foo_d2 <= foo_d1;
        foo_d3 <= foo_d2;
    end if;
end process;
```

### 4.4 Functional Delays

[`addRegisteredSignalCopy()`](flopoco/code/VHDLOperators/include/flopoco/Operator.hpp:380) creates a register that is a **functional delay** (e.g., z^(-1) for filters), not a pipeline register. These are preserved regardless of pipelining decisions.

### 4.5 Loop Handling

For feedback loops (e.g., accumulators):
- User calls `stopPipelining()` at loop start, `restartPipelining()` at loop end
- The scheduler detects the maximum achievable frequency from the largest loop
- A second pass pipelines the rest of the circuit at that frequency

---

## 5. Core Algorithms

### 5.1 FP Addition — [`FPAddSinglePath`](flopoco/code/VHDLOperators/src/FPAddSub/FPAddSinglePath.cpp)

Single-path architecture (simpler, good for moderate sizes):

```
INPUTS: X, Y in FloPoCo format (wE, wF)

1. COMPARE & SWAP
   - Compare |X| vs |Y| using exception+exponent+fraction
   - swap = 1 if |X| < |Y|
   - newX = larger operand, newY = smaller

2. EXPONENT DIFFERENCE
   - expDiff = expX - expY (after swap)

3. ALIGNMENT
   - Right-shift fracY by expDiff positions (using Shifter subcomponent)
   - Compute sticky bit from shifted-out bits
   - Pad: fracYpad = "0" & shiftedFracY

4. EFFECTIVE OPERATION
   - EffSub = signX XOR signY
   - If subtraction: XOR fracYpad with all-ones, set carry-in = EffSub AND NOT sticky

5. SIGNIFICAND ADDITION
   - fracAddResult = fracXpad + fracYpadXorOp + cInSigAdd  (via IntAdder)
   - fracXpad = "01" & fracX & "00" (with guard/round bits)

6. NORMALIZATION
   - Normalizer (LZC + left shift) on fracAddResult & sticky
   - Outputs: nZerosNew (shift count) and shiftedFrac

7. EXPONENT UPDATE
   - extendedExpInc = expX + 1
   - updatedExp = extendedExpInc - nZerosNew

8. ROUNDING (round-to-nearest-even)
   - stk = shiftedFrac[2] | shiftedFrac[1] | shiftedFrac[0]
   - rnd = shiftedFrac[3]
   - lsb = shiftedFrac[4]
   - needToRound = (rnd & stk) | (rnd & ~stk & lsb)
   - RoundedExpFrac = expFrac + needToRound  (via IntAdder)

9. EXCEPTION HANDLING
   - Combines input exceptions (sXsYExnXY) via truth table
   - Updates based on post-normalization overflow
   - Special case: exact zero → positive zero in RN mode

OUTPUT: R = excR & signR & expR & fracR
```

FloPoCo also has a **dual-path** variant ([`FPAddDualPath`](flopoco/code/VHDLOperators/src/FPAddSub/FPAddDualPath.cpp)) that separates close and far paths for better performance at large precisions.

### 5.2 FP Multiplication — [`FPMult`](flopoco/code/VHDLOperators/src/FPMultSquare/FPMult.cpp)

```
INPUTS: X, Y in FloPoCo format

1. SIGN
   sign = X.sign XOR Y.sign

2. EXPONENT
   expSumPreSub = ("00" & expX) + ("00" & expY)
   expSum = expSumPreSub - bias    // bias = 2^(wE-1) - 1

3. SIGNIFICAND MULTIPLICATION
   sigX = "1" & X.fraction    // prepend implicit 1
   sigY = "1" & Y.fraction
   sigProd = IntMultiplier(sigX, sigY)  // subcomponent
   
   // For faithful rounding: truncated product of size wFR + 3 guard bits
   // For correct rounding: full product of size (wFX+1)+(wFY+1)

4. NORMALIZATION
   norm = sigProd[MSB]    // 0 or 1
   expPostNorm = expSum + norm
   
   // Shift significand based on norm bit
   sigProdExt = (sigProd << 1) if norm=1 else (sigProd << 2)

5. ROUNDING (for correctly rounded)
   sticky = sigProdExt[guard_pos]
   guard = OR of lower bits
   round = sticky & ((guard & ~lsb) | lsb)
   
   expSigPostRound = (expPostNorm & sigProdExt[top]) + round  // via IntAdder

6. EXCEPTION HANDLING
   excSel = X.exn & Y.exn  (4 bits)
   Truth table:
     00×anything = 0×anything = zero
     01×01 = normal×normal = normal
     01×10 or 10×01 = normal×inf = inf
     anything involving 11 = NaN
   
   Post-normalization overflow check on expPostNorm top bits

OUTPUT: R = finalExc & sign & expR & fracR
```

### 5.3 FP Division — [`FPDiv`](flopoco/code/VHDLOperators/src/FPDivSqrt/FPDiv.cpp)

Uses **SRT digit recurrence** (radix-4 or radix-8).

```
INPUTS: X, Y in FloPoCo format

1. PREPROCESSING
   fX = "1" & X.fraction
   fY = "1" & Y.fraction
   expR0 = expX - expY
   sR = X.sign XOR Y.sign
   Exception handling via truth table on exnXY

2. SRT ITERATION (for radix-4, alpha=2, nDigit iterations)
   D = fY                           // divisor
   w[nDigit-1] = "00" & "0" & fX   // initial partial remainder
   
   FOR i = nDigit-1 DOWNTO 1:
     // Selection function: truncate w and D, look up in table
     sel[i] = w[i][top bits] & D[top bits]
     q[i] = SelFunctionTable(sel[i])    // digit in {-alpha..alpha}
     
     // Compute q[i] × D
     // Update: w[i-1] = radix * (w[i] - q[i] × D)
     //   implemented as: left-shift + add/subtract based on q[i] value
   
   // Final digit
   q[0] = 0 if w[0]==0, else sign-based

3. QUOTIENT RECONSTRUCTION  
   // Accumulate positive and negative digit contributions
   qP = concatenate all positive parts
   qM = concatenate all negative parts  
   quotient = qP - qM

4. NORMALIZATION & ROUNDING
   // quotient may need 1-bit left shift if < 1
   norm = quotient[MSB]
   expR = expR0 + bias + norm - 1
   // Round-to-nearest-even on remaining bits

OUTPUT: R = exnR & sR & expR & fracR
```

The selection function table is precomputed by [`selFunctionTable()`](flopoco/code/VHDLOperators/src/FPDivSqrt/FPDiv.cpp:59) which, for each `(w, d)` input, finds the digit `k ∈ {-alpha..alpha}` satisfying the SRT convergence bounds: `L_k(d) ≤ w < U_k(d)`.

### 5.4 FP Square Root — [`FPSqrt`](flopoco/code/VHDLOperators/src/FPDivSqrt/FPSqrt.cpp)

Two methods implemented:

#### Method 0: Binary Restoring Algorithm
```
Based on Parhami's textbook (2nd ed., p.438):
  Recurrence: R_i = 2*R_{i-1} - 2*d_i*S_{i-1} - 2^(-i)*d_i^2

1. PREPROCESSING
   eRn1 = (expX >> 1) + half_bias + expX[0]  // halve exponent
   R[0] = pre-normalized fraction (depends on exponent parity)
   S[0] = "1", d[0] = 1

2. ITERATION (i = 1 to wF+1)
   TwoR = R[i-1] & "0"                    // shift left
   T[i] = TwoR[high] - ("0" & S[i-1] & "01")  // tentative subtraction
   d[i] = NOT T[i][sign]                  // d=1 if T≥0, d=0 if T<0
   S[i] = S[i-1] & d[i]                   // append digit
   R[i] = T[i] if d[i]=1, else TwoR       // restoring step

3. OUTPUT
   fR = S[wF+1][wF:1]  (remove leading 1)
   round = d[wF+1]
```

#### Method 1: Binary Non-Restoring Algorithm
Similar but uses signed digits `s_i ∈ {-1, +1}`, avoiding the conditional restoration step.

### 5.5 FP Exponential — [`FPExp`](flopoco/code/VHDLOperators/src/ExpLog/FPExp.cpp)

Uses table-based range reduction + polynomial approximation:

```
INPUT: X in FloPoCo format

1. RANGE REDUCTION
   Convert to fixed-point: x = (-1)^s × 2^E × 1.F
   Split x = A + Y + Z where:
   - A is the k MSBs (tabulated)
   - Y is the next few bits  
   - Z is the remaining bits (small, for polynomial)

2. COMPUTATION (using exp(x) = exp(A) × exp(Y) × exp(Z))
   - exp(A): looked up in table (size 2^k × sizeExpA bits)
   - exp(Y): looked up in table OR computed
   - exp(Z) ≈ 1 + Z + Z²/2 + ...  (polynomial degree d)
     OR  exp(Z)-Z-1 tabulated (for small Z, using table of size 2^sizeZtrunc)
   
   Product via truncated multiplier (IntMultiplier subcomponent)

3. POST-PROCESSING
   Reconstruct full result, handle sign, overflow, underflow
   Round to FloPoCo format

Architecture parameters chosen by ExpArchitecture class based on:
- blockRAM size
- wE, wF
- User-specified k, d (or auto-computed)
```

### 5.6 FP Logarithm — [`FPLogIterative`](flopoco/code/VHDLOperators/src/ExpLog/FPLogIterative.cpp)

Uses iterative range reduction similar to the classical algorithm by Detrey and de Dinechin, based on the identity `log(1+x) = log(1+a_i) + log((1+x)/(1+a_i))` with table lookups for `log(1+a_i)`.

---

## 6. Building Blocks

### 6.1 [`Shifter`](flopoco/code/VHDLOperators/include/flopoco/ShiftersEtc/Shifters.hpp)

A barrel shifter supporting left or right shift:
- **Parameters**: `wX` (input width), `maxShift`, direction, `wR` (output width), `computeSticky`
- **Implementation**: Log-depth multiplexer tree (one stage per bit of shift amount)
- **Sticky computation**: Optional OR of all shifted-out bits (crucial for rounding)
- Used in FP addition alignment and normalization

### 6.2 [`LZOC`](flopoco/code/VHDLOperators/include/flopoco/ShiftersEtc/LZOC.hpp) — Leading Zero/One Counter

- Recursive structure with `ceil(log2(wIn))` stages
- `countType`: 0=count zeros, 1=count ones, -1=dynamic (input `OZb` selects)
- Output width: `floor(log2(wIn)) + 1` bits

### 6.3 [`Normalizer`](flopoco/code/VHDLOperators/include/flopoco/ShiftersEtc/Normalizer.hpp)

Combines LZOC + left shifter:
- Counts leading zeros/ones, then shifts to normalize
- Optional sticky bit computation from bits shifted past `wR`
- Parameters: `wX`, `wR`, `maxShift`, `computeSticky`, `countType`
- Used in FP addition post-addition normalization

### 6.4 [`IntMultiplier`](flopoco/code/VHDLOperators/src/IntMult/IntMultiplier.cpp)

Large integer multiplier using **tiling strategies**:
- Decomposes large multiplications into tiles that fit DSP blocks
- Multiple strategies: basic tiling, beam search, CSV-imported
- Partial products accumulated via [`BitHeap`](flopoco/code/VHDLOperators/include/flopoco/BitHeap/BitHeap.hpp)
- Supports truncated multiplication (faithful rounding)
- Key parameters: `wX`, `wY`, `wOut`, `dspThreshold`

### 6.5 [`BitHeap`](flopoco/code/VHDLOperators/include/flopoco/BitHeap/BitHeap.hpp)

A bit-column compression infrastructure:

```
                Column weights
    ...  8  4  2  1  (position)
         •  •  •  •  
         •  •  •  •  
            •  •     
            •        
    ─────────────────
    Compress to 2 rows, then final add
```

- **`addBit(rhs, position)`**: add a bit at a given weight
- **`addConstantOneBit(position)`**: add constant 1
- **`subtractConstantOneBit(position)`**: subtract constant 1 (via 2's complement trick)
- **Compression strategies**: First-fitting, Parandez-Afshar, Max-efficiency, Optimal (ILP-based)
- The compression reduces the bit heap to at most 2 rows, then a final adder produces the sum

### 6.6 Integer Adders

- [`IntAdder`](flopoco/code/VHDLOperators/src/IntAddSubCmp/IntAdder.cpp): Basic n-bit adder
- [`IntAddSub`](flopoco/code/VHDLOperators/src/IntAddSubCmp/IntAddSub.cpp): Combined adder/subtractor
- [`IntDualAddSub`](flopoco/code/VHDLOperators/src/IntAddSubCmp/IntDualAddSub.cpp): Computes both X+Y and X-Y simultaneously

---

## 7. Table Infrastructure

### 7.1 [`Table`](flopoco/code/HighLevelArithmetic/include/flopoco/Tables/Table.hpp) (Pure Data)

A simple vector of `mpz_class` values indexed by input:
- `wIn`: input address width
- `wOut`: output data width  
- `minIn`, `maxIn`: input range
- `values[]`: the table contents

### 7.2 [`DiffCompressedTable`](flopoco/code/VHDLOperators/include/flopoco/Tables/DiffCompressedTable.hpp)

VHDL-generating table operator with optional **differential compression** (Hsiao method):
- Splits table into a base value + small difference
- Reduces memory by trading LUTs for a small adder
- Controlled by `Target::tableCompression()` flag

### 7.3 [`DifferentialCompression`](flopoco/code/HighLevelArithmetic/include/flopoco/Tables/DifferentialCompression.hpp)

Implements the compression algorithm:
- Groups consecutive table entries
- Stores a shared base + per-entry differences
- Optimizes group sizes to minimize total cost (LUTs for table + LUTs for adder)

### 7.4 Table Cost Model ([`TableCostModel`](flopoco/code/HighLevelArithmetic/include/flopoco/Tables/TableCostModel.hpp))

Estimates the cost (in LUTs or BRAMs) of implementing a table on a specific target.

### 7.5 Function Approximation Tables

- [`BasicPolyApprox`](flopoco/code/HighLevelArithmetic/include/flopoco/FixFunctions/BasicPolyApprox.hpp): Polynomial approximation using Sollya
- [`UniformPiecewisePolyApprox`](flopoco/code/HighLevelArithmetic/include/flopoco/FixFunctions/UniformPiecewisePolyApprox.hpp): Uniform piecewise polynomial
- [`Multipartite`](flopoco/code/HighLevelArithmetic/include/flopoco/FixFunctions/MultipartiteTable/Multipartite.hpp): Multipartite table method (multiple small tables summed)

---

## 8. Key Design Patterns

### 8.1 Constructor-as-Generator Pattern
The operator's constructor **is** the VHDL generator. All `vhdl << ...` statements execute at construction time, building the VHDL string.

**Amaranth equivalent**: The `elaborate()` method of an Amaranth `Elaboratable`.

### 8.2 Declare-Use-Schedule Pattern
```cpp
declare(delay, "signalName", width)  // declare with timing
vhdl << tab << "signalName <= ...;"  // use in VHDL
// Framework automatically schedules and pipelines
```

**Amaranth equivalent**: Create `Signal()`, use in `m.d.comb +=` or `m.d.sync +=`. Pipelining must be done explicitly.

### 8.3 Subcomponent Composition
```cpp
newInstance("OpType", "instanceName", "params", "inPorts", "outPorts");
```

**Amaranth equivalent**: `m.submodules.instanceName = op = OpType(params); m.d.comb += [op.input.eq(signal), ...]`

### 8.4 Target-Parameterized Delays
Every `declare()` uses `getTarget()->someDelay(n)` to annotate timing. This allows the same operator code to generate different pipeline depths for different FPGAs.

**Amaranth equivalent**: No direct equivalent — Amaranth relies on synthesis tools for timing closure. Pipeline depth would need to be explicitly parameterized.

### 8.5 Shared Operators
When the same operator configuration is used multiple times, FloPoCo generates the VHDL entity once and instantiates it multiple times. Controlled by `setShared()`.

**Amaranth equivalent**: Amaranth naturally deduplicates identical elaboratables.

---

## 9. Mapping to Amaranth

| FloPoCo Concept | Amaranth Equivalent |
|----------------|---------------------|
| `Operator` base class | `Elaboratable` / `Component` |
| `vhdl << ...` stream | `m.d.comb +=` / `m.d.sync +=` statements in `elaborate()` |
| `declare(delay, name, width)` | `Signal(width, name=name)` (no auto-pipeline) |
| `addInput(name, width)` | Add to component's signature / interface |
| `addOutput(name, width)` | Add to component's signature / interface |
| `addFPInput(name, wE, wF)` | Custom `FPLayout` / `StructLayout` with `exn + sign + exp + frac` |
| `newInstance(...)` | `m.submodules.name = SubOp(...)` |
| `FlopocoStream` parsing | Not needed — Amaranth tracks dependencies natively |
| `Signal.cycle_` / `criticalPath_` | No equivalent — pipeline stages are manual |
| `Target.adderDelay(n)` | No direct equivalent — use platform timing reports |
| `BitHeap` | Can build custom — or use Amaranth's `Cat()` + adder tree |
| `Shifter` | `Signal >> amount` / barrel shifter module |
| `LZOC` / `Normalizer` | Custom modules (priority encoder + shifter) |
| `Table` | Amaranth `Memory` or `Switch` statement |
| `TestCase` / `emulate()` | Python unittest with `Simulator` |
| `getTarget()->lutDelay()` | No equivalent — Amaranth is target-agnostic |
| FloPoCo internal FP format | Custom Python `FPNumber` class + `StructLayout` |
| `setShared()` / shared operators | Amaranth auto-deduplication |
| Pipeline registers | Manual `m.d.sync +=` stages or pipeline wrapper utility |

### Key Architectural Decisions for Reimplementation

1. **FP Format**: Keep FloPoCo's 2-bit exception format — it greatly simplifies hardware.

2. **Pipelining**: Build a pipeline insertion utility that takes a combinatorial design and a latency target, then inserts registers. Alternatively, support manual pipeline stage annotation.

3. **BitHeap**: Implement as a Python class that collects weighted bits and generates an efficient compressor tree.

4. **Tables**: Use Amaranth `Memory` for BRAM-based tables, `Switch` for LUT-based tables.

5. **Testing**: Use FloPoCo's MPFR-based `emulate()` approach — Python has `mpmath` as equivalent.

6. **Target Independence**: Unlike FloPoCo which generates target-specific VHDL, Amaranth generates target-independent RTLIL. Let synthesis tools handle timing. Expose pipeline depth as a parameter.
