"""Floating-point 2^x (pipelined, 6-cycle latency).

Algorithm:
  1. Unpack, handle exceptions
  2. Range reduce: x = k + f, where k = floor(x), f in [0, 1)
  3. Compute 2^f via table + polynomial on [0, 1)
  4. Reconstruct: 2^x = 2^k * 2^f (shift exponent by k)
  5. Pack result
"""
from __future__ import annotations

import math

from amaranth import *

from ...format import FPFormat
from ...pipelined import PipelinedComponent
from ..table import Table

__all__ = ["FPExp2"]


class FPExp2(PipelinedComponent):
    """Pipelined floating-point 2^x.

    Parameters
    ----------
    fmt : FPFormat
        Floating-point format.
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")
        self.latency = 6

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we, wf, bias = fmt.we, fmt.wf, fmt.bias

        # Table for 2^f, f in [0, 1) with table_bits address bits
        table_bits = min(wf, 8)
        table_size = 1 << table_bits
        frac_bits = wf + 4
        exp2_table = []
        for i in range(table_size):
            f = i / table_size
            val = int(round((2.0 ** f) * (1 << frac_bits)))
            exp2_table.append(val & ((1 << (frac_bits + 2)) - 1))

        tbl = Table(table_bits, frac_bits + 2, exp2_table)
        m.submodules.exp2_tbl = tbl

        # Stage 0: Unpack
        a_mant = Signal(wf); a_exp = Signal(we); a_sign = Signal(); a_exc = Signal(2)
        m.d.comb += [
            a_mant.eq(self.a[:wf]), a_exp.eq(self.a[wf:wf + we]),
            a_sign.eq(self.a[wf + we]), a_exc.eq(self.a[wf + we + 1:]),
        ]

        # Stage 0 → 1
        s1_mant = Signal(wf); s1_exp = Signal(we); s1_sign = Signal(); s1_exc = Signal(2)
        m.d.sync += [s1_mant.eq(a_mant), s1_exp.eq(a_exp), s1_sign.eq(a_sign), s1_exc.eq(a_exc)]

        # Stage 1: Exception handling + compute k = floor(x) and frac(x)
        s1_is_special = Signal()
        s1_exc_out = Signal(2)
        with m.If(s1_exc == 0b11):
            m.d.comb += [s1_exc_out.eq(0b11), s1_is_special.eq(1)]
        with m.Elif(s1_exc == 0b10):
            with m.If(s1_sign):
                m.d.comb += [s1_exc_out.eq(0b00), s1_is_special.eq(1)]
            with m.Else():
                m.d.comb += [s1_exc_out.eq(0b10), s1_is_special.eq(1)]
        with m.Elif(s1_exc == 0b00):
            m.d.comb += [s1_exc_out.eq(0b01), s1_is_special.eq(1)]
        with m.Else():
            m.d.comb += [s1_exc_out.eq(0b01), s1_is_special.eq(0)]

        # Convert x to fixed-point to extract floor(x) and frac(x)
        # x = (1.mantissa) * 2^e where e = biased_exp - bias
        # We need enough integer bits for floor(x). For half precision, max useful
        # exponent is ~15, so we need ~5 integer bits. Use we+1 integer bits.
        int_bits = we + 1
        total_fp_bits = int_bits + wf  # fixed-point: int_bits.wf
        e_val = Signal(signed(we + 1))
        m.d.comb += e_val.eq(s1_exp - bias)

        # Full significand with implicit 1: (1 << wf) | mantissa
        significand = Signal(wf + 1)
        m.d.comb += significand.eq(Cat(s1_mant, Const(1, 1)))

        # Shift significand left by e to get fixed-point representation
        # x_fp = significand << e (if e >= 0) or significand >> (-e) (if e < 0)
        # Fixed point has int_bits integer bits and wf fractional bits
        # significand is (wf+1) bits representing 1.mant, which is a value in [1,2)
        # with wf fractional bits. When e=0, the integer part is 1 and frac = mant.
        # We represent x in fixed-point as a (total_fp_bits)-bit unsigned number
        # where the top int_bits bits are integer and bottom wf bits are fraction.
        x_fixed = Signal(total_fp_bits)
        # Position of significand in the fixed-point: when e=0, the significand
        # (wf+1 bits) occupies bits [0:wf+1], with bit wf being the integer "1" bit.
        # For general e, we shift by e: bits [0+e : wf+1+e] (if e>=0)
        # We compute: x_fixed = significand << e (within the fixed-point field)
        # But we need to handle negative e (x < 1) where floor(x)=0 and frac(x)=x.

        # Use a wide shifted value
        shifted_wide = Signal(total_fp_bits + wf + 1)
        # Start with significand at position [0:wf+1], then shift left by (e + 0)
        # Actually, let's think in terms of the fixed-point format:
        # x_fixed has wf fractional bits. significand has wf fractional bits.
        # So when e=0, x_fixed = significand (the low wf+1 bits).
        # When e>0, x_fixed = significand << e.
        # When e<0, x_fixed = significand >> (-e).

        # For the shift, we use a barrel shifter approach with clipping
        # Max shift left = int_bits-1 (since we have int_bits integer bits and significand already has 1 int bit)
        # Max shift right = wf (losing all bits)

        # Simple approach: compute for each possible e
        with m.If(e_val < 0):
            # x < 1: floor(x) = 0, frac = x
            # shift right by -e
            shift_r = Signal(we + 1)
            m.d.comb += shift_r.eq(-e_val)
            m.d.comb += x_fixed.eq(significand >> shift_r)
        with m.Else():
            # shift left by e
            m.d.comb += x_fixed.eq(significand << e_val[:we])

        # Extract integer part (top int_bits of x_fixed) and fractional part (bottom wf)
        k_val = Signal(signed(int_bits + 1))
        frac_val = Signal(wf)
        m.d.comb += [
            k_val.eq(x_fixed[wf:]),  # integer bits
            frac_val.eq(x_fixed[:wf]),  # fractional bits
        ]

        # For negative x: 2^(-|x|) = 2^(-k-1) * 2^(1-f) if f > 0
        # Simpler: for negative x, 2^x = 2^(-|x|) = 1/2^|x|
        # |x| = k + f where k = floor(|x|), f = frac(|x|)
        # 2^(-k-f) = 2^(-k) * 2^(-f) = 2^(-k-1) * 2^(1-f)
        # So for negative x: result_exp = bias - k - 1, table lookup 2^(1-f)
        # Or: result_exp = bias - k, table lookup 2^(-f) ... but 2^(-f) is in (0.5, 1]
        # Easiest: for negative x, compute 2^|x| and invert conceptually
        # Actually: 2^x where x<0: let |x| = k + f
        # 2^x = 2^(-k-f) = 2^(-k) * 2^(-f)
        # 2^(-f) = 1/2^f, f in [0,1), so 2^(-f) in (0.5, 1]
        # If f=0: 2^(-f)=1, result = 2^(-k), exp = bias-k, mant=0
        # If f>0: 2^(-f) = 2^(1-f)/2, and 2^(1-f) in (1,2], so result = 2^(-k-1) * 2^(1-f)
        #   exp = bias-k-1, mant from 2^(1-f) table

        # For positive x: 2^x = 2^k * 2^f, exp = bias+k, mant from 2^f table
        # For negative x: if f=0, exp = bias-k, mant=0
        #                  if f>0, let f' = 1-f, exp = bias-k-1, mant from 2^f' table

        neg_k = Signal(signed(int_bits + 1))
        neg_frac = Signal(wf)
        neg_has_frac = Signal()
        m.d.comb += [
            neg_k.eq(k_val),
            neg_frac.eq(((1 << wf) - frac_val) & ((1 << wf) - 1)),  # 1) - f in fixed point
            neg_has_frac.eq(frac_val != 0),
        ]

        # Table address from fractional part
        tbl_addr = Signal(table_bits)
        tbl_frac = Signal(wf)
        actual_k = Signal(signed(int_bits + 1))
        with m.If(s1_sign):
            # Negative x: use 1-f for table (if f>0), adjust k
            with m.If(neg_has_frac):
                m.d.comb += [
                    tbl_frac.eq(neg_frac),
                    actual_k.eq(-k_val - 1),
                ]
            with m.Else():
                m.d.comb += [
                    tbl_frac.eq(0),
                    actual_k.eq(-k_val),
                ]
        with m.Else():
            m.d.comb += [
                tbl_frac.eq(frac_val),
                actual_k.eq(k_val),
            ]

        m.d.comb += tbl_addr.eq(tbl_frac >> (wf - table_bits) if wf >= table_bits else tbl_frac)
        m.d.comb += tbl.addr.eq(tbl_addr)

        # Stage 1 → 2
        s2_exc = Signal(2); s2_special = Signal(); s2_k = Signal(signed(int_bits + 1)); s2_sign = Signal()
        m.d.sync += [s2_exc.eq(s1_exc_out), s2_special.eq(s1_is_special), s2_k.eq(actual_k), s2_sign.eq(s1_sign)]

        # Stage 2: Read table (1-cycle memory latency)
        s3_exc = Signal(2); s3_special = Signal(); s3_k = Signal(signed(int_bits + 1)); s3_sign = Signal()
        m.d.sync += [s3_exc.eq(s2_exc), s3_special.eq(s2_special), s3_k.eq(s2_k), s3_sign.eq(s2_sign)]

        # Stage 3: Table value available, compute result exponent
        tbl_val = Signal(frac_bits + 2)
        m.d.comb += tbl_val.eq(tbl.data)

        result_exp = Signal(signed(we + 4))
        m.d.comb += result_exp.eq(bias + s3_k)

        result_mant = Signal(wf)
        m.d.comb += result_mant.eq(tbl_val[frac_bits - wf:frac_bits])

        # Stage 3 → 4
        s4_exc = Signal(2); s4_special = Signal(); s4_rexp = Signal(signed(we + 4)); s4_rmant = Signal(wf)
        m.d.sync += [s4_exc.eq(s3_exc), s4_special.eq(s3_special), s4_rexp.eq(result_exp), s4_rmant.eq(result_mant)]

        # Stage 4: Overflow/underflow check, pack
        final_exc = Signal(2); final_sign = Signal(); final_mant = Signal(wf); final_exp = Signal(we)
        max_exp = (1 << we) - 1

        with m.If(s4_special):
            m.d.comb += [final_exc.eq(s4_exc), final_sign.eq(0), final_mant.eq(0),
                         final_exp.eq(Mux(s4_exc == 0b01, bias, 0))]
        with m.Elif(s4_rexp >= max_exp):
            m.d.comb += [final_exc.eq(0b10), final_sign.eq(0), final_mant.eq(0), final_exp.eq(0)]
        with m.Elif(s4_rexp <= 0):
            m.d.comb += [final_exc.eq(0b00), final_sign.eq(0), final_mant.eq(0), final_exp.eq(0)]
        with m.Else():
            m.d.comb += [final_exc.eq(0b01), final_sign.eq(0), final_mant.eq(s4_rmant), final_exp.eq(s4_rexp[:we])]

        # Stage 4 → 5
        s5 = Signal(fmt.width)
        m.d.sync += s5.eq(Cat(final_mant, final_exp, final_sign, final_exc))

        # Stage 5 → 6 (output)
        s6 = Signal(fmt.width)
        m.d.sync += s6.eq(s5)
        m.d.comb += self.o.eq(s6)

        return m
