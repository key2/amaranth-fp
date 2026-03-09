"""Floating-point divider (pipelined, 6 stages).

Based on non-restoring division algorithm, translated to Amaranth HDL.
Operates on the internal FloPoCo format:
    [exception(2) | sign(1) | exponent(we) | mantissa(wf)]
Exception encoding: 00=zero, 01=normal, 10=inf, 11=NaN.
"""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from ..building_blocks import RoundingUnit

__all__ = ["FPDiv"]


class FPDiv(PipelinedComponent):
    """Pipelined floating-point divider (6-cycle latency).

    Parameters
    ----------
    fmt : FPFormat
        Floating-point format (defines we, wf).

    Attributes
    ----------
    a : Signal(fmt.width), in
    b : Signal(fmt.width), in
    o : Signal(fmt.width), out
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.b = Signal(fmt.width, name="b")
        self.o = Signal(fmt.width, name="o")
        self.latency = 6

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we = fmt.we
        wf = fmt.wf

        # ==================================================================
        # Stage 0: unpack inputs, compute result sign
        # ==================================================================
        a_mant = Signal(wf, name="a_mant")
        a_exp = Signal(we, name="a_exp")
        a_sign = Signal(name="a_sign")
        a_exc = Signal(2, name="a_exc")

        b_mant = Signal(wf, name="b_mant")
        b_exp = Signal(we, name="b_exp")
        b_sign = Signal(name="b_sign")
        b_exc = Signal(2, name="b_exc")

        m.d.comb += [
            a_mant.eq(self.a[:wf]),
            a_exp.eq(self.a[wf:wf + we]),
            a_sign.eq(self.a[wf + we]),
            a_exc.eq(self.a[wf + we + 1:]),

            b_mant.eq(self.b[:wf]),
            b_exp.eq(self.b[wf:wf + we]),
            b_sign.eq(self.b[wf + we]),
            b_exc.eq(self.b[wf + we + 1:]),
        ]

        result_sign = Signal(name="result_sign")
        m.d.comb += result_sign.eq(a_sign ^ b_sign)

        # ── Stage 0 → 1 pipeline register ──
        a_mant_r1 = Signal(wf, name="a_mant_r1")
        a_exp_r1 = Signal(we, name="a_exp_r1")
        a_exc_r1 = Signal(2, name="a_exc_r1")
        b_mant_r1 = Signal(wf, name="b_mant_r1")
        b_exp_r1 = Signal(we, name="b_exp_r1")
        b_exc_r1 = Signal(2, name="b_exc_r1")
        result_sign_r1 = Signal(name="result_sign_r1")
        m.d.sync += [
            a_mant_r1.eq(a_mant), a_exp_r1.eq(a_exp), a_exc_r1.eq(a_exc),
            b_mant_r1.eq(b_mant), b_exp_r1.eq(b_exp), b_exc_r1.eq(b_exc),
            result_sign_r1.eq(result_sign),
        ]
        for s in [a_mant_r1, a_exp_r1, a_exc_r1, b_mant_r1, b_exp_r1, b_exc_r1, result_sign_r1]:
            self.add_latency(s, 1)

        # ==================================================================
        # Stage 1: exception decode, exponent subtraction
        # ==================================================================
        exc_sel = Signal(4, name="exc_sel")
        m.d.comb += exc_sel.eq(Cat(b_exc_r1, a_exc_r1))

        exc_result = Signal(2, name="exc_result")
        with m.Switch(exc_sel):
            with m.Case(0b0000): m.d.comb += exc_result.eq(0b11)  # 0/0
            with m.Case(0b0001): m.d.comb += exc_result.eq(0b00)  # 0/n
            with m.Case(0b0010): m.d.comb += exc_result.eq(0b00)  # 0/inf
            with m.Case(0b0100): m.d.comb += exc_result.eq(0b10)  # n/0
            with m.Case(0b0101): m.d.comb += exc_result.eq(0b01)  # n/n
            with m.Case(0b0110): m.d.comb += exc_result.eq(0b00)  # n/inf
            with m.Case(0b1000): m.d.comb += exc_result.eq(0b10)  # inf/0
            with m.Case(0b1001): m.d.comb += exc_result.eq(0b10)  # inf/n
            with m.Case(0b1010): m.d.comb += exc_result.eq(0b11)  # inf/inf
            with m.Default():    m.d.comb += exc_result.eq(0b11)

        bias = fmt.bias
        exp_diff = Signal(we + 2, name="exp_diff")
        m.d.comb += exp_diff.eq(
            Cat(a_exp_r1, Const(0, 2)).as_signed()
            - Cat(b_exp_r1, Const(0, 2)).as_signed()
            + bias
        )

        sig_a = Signal(wf + 1, name="sig_a")
        sig_b = Signal(wf + 1, name="sig_b")
        m.d.comb += [
            sig_a.eq(Cat(a_mant_r1, Const(1, 1))),
            sig_b.eq(Cat(b_mant_r1, Const(1, 1))),
        ]

        # ── Stage 1 → 2 pipeline register ──
        exc_result_r2 = Signal(2, name="exc_result_r2")
        exp_diff_r2 = Signal(we + 2, name="exp_diff_r2")
        sig_a_r2 = Signal(wf + 1, name="sig_a_r2")
        sig_b_r2 = Signal(wf + 1, name="sig_b_r2")
        result_sign_r2 = Signal(name="result_sign_r2")
        m.d.sync += [
            exc_result_r2.eq(exc_result),
            exp_diff_r2.eq(exp_diff),
            sig_a_r2.eq(sig_a),
            sig_b_r2.eq(sig_b),
            result_sign_r2.eq(result_sign_r1),
        ]
        for s in [exc_result_r2, exp_diff_r2, sig_a_r2, sig_b_r2, result_sign_r2]:
            self.add_latency(s, 2)

        # ==================================================================
        # Stage 2: first half of division iterations
        # ==================================================================
        n_bits = wf + 3
        rem_width = wf + 2

        # First quotient bit
        diff_0 = Signal(rem_width, name="diff_0")
        m.d.comb += diff_0.eq(sig_a_r2 - sig_b_r2)
        q_bit_0 = Signal(name="q_bit_0")
        m.d.comb += q_bit_0.eq(~diff_0[rem_width - 1])

        rem_after_0 = Signal(rem_width, name="rem_after_0")
        with m.If(q_bit_0):
            m.d.comb += rem_after_0.eq(diff_0)
        with m.Else():
            m.d.comb += rem_after_0.eq(sig_a_r2)

        remainders = [rem_after_0]
        quotient_bits = [q_bit_0]

        half = n_bits // 2

        for i in range(1, half):
            rem_cur = remainders[i - 1]
            rem_shifted = Signal(rem_width, name=f"rem_shifted_{i}")
            m.d.comb += rem_shifted.eq(rem_cur << 1)

            diff = Signal(rem_width, name=f"diff_{i}")
            m.d.comb += diff.eq(rem_shifted - sig_b_r2)

            q_bit = Signal(name=f"q_bit_{i}")
            m.d.comb += q_bit.eq(~diff[rem_width - 1])
            quotient_bits.append(q_bit)

            rem_next = Signal(rem_width, name=f"rem_next_{i}")
            with m.If(q_bit):
                m.d.comb += rem_next.eq(diff)
            with m.Else():
                m.d.comb += rem_next.eq(rem_shifted)
            remainders.append(rem_next)

        # Partial quotient (first half bits)
        partial_q = Signal(half, name="partial_q")
        for i in range(half):
            m.d.comb += partial_q[half - 1 - i].eq(quotient_bits[i])

        # ── Stage 2 → 3 pipeline register ──
        partial_q_r3 = Signal(half, name="partial_q_r3")
        rem_r3 = Signal(rem_width, name="rem_r3")
        sig_b_r3 = Signal(wf + 1, name="sig_b_r3")
        exp_diff_r3 = Signal(we + 2, name="exp_diff_r3")
        exc_result_r3 = Signal(2, name="exc_result_r3")
        result_sign_r3 = Signal(name="result_sign_r3")
        m.d.sync += [
            partial_q_r3.eq(partial_q),
            rem_r3.eq(remainders[-1]),
            sig_b_r3.eq(sig_b_r2),
            exp_diff_r3.eq(exp_diff_r2),
            exc_result_r3.eq(exc_result_r2),
            result_sign_r3.eq(result_sign_r2),
        ]
        for s in [partial_q_r3, rem_r3, sig_b_r3, exp_diff_r3, exc_result_r3, result_sign_r3]:
            self.add_latency(s, 3)

        # ==================================================================
        # Stage 3: second half of division iterations
        # ==================================================================
        remainders2 = [rem_r3]
        quotient_bits2 = []

        for i in range(half, n_bits):
            rem_cur = remainders2[i - half]
            rem_shifted = Signal(rem_width, name=f"rem_shifted2_{i}")
            m.d.comb += rem_shifted.eq(rem_cur << 1)

            diff = Signal(rem_width, name=f"diff2_{i}")
            m.d.comb += diff.eq(rem_shifted - sig_b_r3)

            q_bit = Signal(name=f"q_bit2_{i}")
            m.d.comb += q_bit.eq(~diff[rem_width - 1])
            quotient_bits2.append(q_bit)

            rem_next = Signal(rem_width, name=f"rem_next2_{i}")
            with m.If(q_bit):
                m.d.comb += rem_next.eq(diff)
            with m.Else():
                m.d.comb += rem_next.eq(rem_shifted)
            remainders2.append(rem_next)

        # Assemble full quotient
        second_half_bits = n_bits - half
        second_q = Signal(second_half_bits, name="second_q")
        for i in range(second_half_bits):
            m.d.comb += second_q[second_half_bits - 1 - i].eq(quotient_bits2[i])

        quotient = Signal(n_bits, name="quotient")
        m.d.comb += quotient.eq(Cat(second_q, partial_q_r3))

        final_rem_nonzero = Signal(name="final_rem_nonzero")
        m.d.comb += final_rem_nonzero.eq(remainders2[-1].any())

        # ── Stage 3 → 4 pipeline register ──
        quotient_r4 = Signal(n_bits, name="quotient_r4")
        final_rem_nz_r4 = Signal(name="final_rem_nz_r4")
        exp_diff_r4 = Signal(we + 2, name="exp_diff_r4")
        exc_result_r4 = Signal(2, name="exc_result_r4")
        result_sign_r4 = Signal(name="result_sign_r4")
        m.d.sync += [
            quotient_r4.eq(quotient),
            final_rem_nz_r4.eq(final_rem_nonzero),
            exp_diff_r4.eq(exp_diff_r3),
            exc_result_r4.eq(exc_result_r3),
            result_sign_r4.eq(result_sign_r3),
        ]
        for s in [quotient_r4, final_rem_nz_r4, exp_diff_r4, exc_result_r4, result_sign_r4]:
            self.add_latency(s, 4)

        # ==================================================================
        # Stage 4: normalization, rounding
        # ==================================================================
        norm = Signal(name="norm")
        m.d.comb += norm.eq(quotient_r4[n_bits - 1])

        exp_post_norm = Signal(we + 2, name="exp_post_norm")
        with m.If(norm):
            m.d.comb += exp_post_norm.eq(exp_diff_r4)
        with m.Else():
            m.d.comb += exp_post_norm.eq(exp_diff_r4 - 1)

        norm_quot = Signal(n_bits, name="norm_quot")
        with m.If(norm):
            m.d.comb += norm_quot.eq(quotient_r4)
        with m.Else():
            m.d.comb += norm_quot.eq(quotient_r4 << 1)

        round_mant = Signal(wf, name="round_mant")
        guard_bit = Signal(name="guard_bit")
        round_bit = Signal(name="round_bit")
        sticky_val = Signal(name="sticky_val")

        m.d.comb += round_mant.eq(norm_quot[n_bits - 1 - wf:n_bits - 1])
        m.d.comb += guard_bit.eq(norm_quot[n_bits - 1 - wf - 1])

        if n_bits - 1 - wf - 2 >= 0:
            m.d.comb += round_bit.eq(norm_quot[n_bits - 1 - wf - 2])
        else:
            m.d.comb += round_bit.eq(0)

        sticky_start = n_bits - 1 - wf - 3
        if sticky_start >= 0:
            m.d.comb += sticky_val.eq(
                norm_quot[:sticky_start + 1].any() | final_rem_nz_r4
            )
        else:
            m.d.comb += sticky_val.eq(final_rem_nz_r4)

        round_in = Signal(wf + 3, name="round_in")
        m.d.comb += round_in.eq(Cat(sticky_val, round_bit, guard_bit, round_mant))

        rounder = RoundingUnit(wf)
        m.submodules.rounder = rounder
        m.d.comb += rounder.mantissa_in.eq(round_in)

        rounded_mant = rounder.mantissa_out
        round_overflow = rounder.overflow

        final_exp = Signal(we + 2, name="final_exp")
        m.d.comb += final_exp.eq(exp_post_norm + round_overflow)

        # ── Stage 4 → 5 pipeline register ──
        rounded_mant_r5 = Signal(wf, name="rounded_mant_r5")
        final_exp_r5 = Signal(we + 2, name="final_exp_r5")
        exc_result_r5 = Signal(2, name="exc_result_r5")
        result_sign_r5 = Signal(name="result_sign_r5")
        m.d.sync += [
            rounded_mant_r5.eq(rounded_mant),
            final_exp_r5.eq(final_exp),
            exc_result_r5.eq(exc_result_r4),
            result_sign_r5.eq(result_sign_r4),
        ]
        for s in [rounded_mant_r5, final_exp_r5, exc_result_r5, result_sign_r5]:
            self.add_latency(s, 5)

        # ==================================================================
        # Stage 5: overflow/underflow, final mux, pack
        # ==================================================================
        exp_top = Signal(2, name="exp_top")
        m.d.comb += exp_top.eq(final_exp_r5[we:we + 2])

        exc_post_norm = Signal(2, name="exc_post_norm")
        with m.Switch(exp_top):
            with m.Case(0b00): m.d.comb += exc_post_norm.eq(0b01)
            with m.Case(0b01): m.d.comb += exc_post_norm.eq(0b10)
            with m.Case(0b10): m.d.comb += exc_post_norm.eq(0b00)
            with m.Case(0b11): m.d.comb += exc_post_norm.eq(0b00)

        final_exc = Signal(2, name="final_exc")
        with m.If((exc_result_r5 == 0b11) | (exc_result_r5 == 0b10) | (exc_result_r5 == 0b00)):
            m.d.comb += final_exc.eq(exc_result_r5)
        with m.Else():
            m.d.comb += final_exc.eq(exc_post_norm)

        final_mant = Signal(wf, name="final_mant")
        final_e = Signal(we, name="final_e")

        with m.If(final_exc == 0b01):
            m.d.comb += [
                final_mant.eq(rounded_mant_r5),
                final_e.eq(final_exp_r5[:we]),
            ]
        with m.Else():
            m.d.comb += [
                final_mant.eq(0),
                final_e.eq(0),
            ]

        # ── Stage 5 → 6 pipeline register (output) ──
        o_r6 = Signal(fmt.width, name="o_r6")
        m.d.sync += o_r6.eq(Cat(final_mant, final_e, result_sign_r5, final_exc))
        m.d.comb += self.o.eq(o_r6)

        return m
