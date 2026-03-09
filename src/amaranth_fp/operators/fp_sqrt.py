"""Floating-point square root (pipelined, 5 stages).

Based on binary restoring square root algorithm, translated to Amaranth HDL.
Operates on the internal FloPoCo format:
    [exception(2) | sign(1) | exponent(we) | mantissa(wf)]
Exception encoding: 00=zero, 01=normal, 10=inf, 11=NaN.
"""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from ..building_blocks import RoundingUnit

__all__ = ["FPSqrt"]


class FPSqrt(PipelinedComponent):
    """Pipelined floating-point square root (5-cycle latency).

    Parameters
    ----------
    fmt : FPFormat
        Floating-point format (defines we, wf).

    Attributes
    ----------
    a : Signal(fmt.width), in
    o : Signal(fmt.width), out
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")
        self.latency = 5

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we = fmt.we
        wf = fmt.wf

        # ==================================================================
        # Stage 0: unpack, exception decode, odd exponent handling
        # ==================================================================
        a_mant = Signal(wf, name="a_mant")
        a_exp = Signal(we, name="a_exp")
        a_sign = Signal(name="a_sign")
        a_exc = Signal(2, name="a_exc")

        m.d.comb += [
            a_mant.eq(self.a[:wf]),
            a_exp.eq(self.a[wf:wf + we]),
            a_sign.eq(self.a[wf + we]),
            a_exc.eq(self.a[wf + we + 1:]),
        ]

        exc_result = Signal(2, name="exc_result")
        with m.Switch(a_exc):
            with m.Case(0b00):
                m.d.comb += exc_result.eq(0b00)
            with m.Case(0b01):
                with m.If(a_sign):
                    m.d.comb += exc_result.eq(0b11)
                with m.Else():
                    m.d.comb += exc_result.eq(0b01)
            with m.Case(0b10):
                with m.If(a_sign):
                    m.d.comb += exc_result.eq(0b11)
                with m.Else():
                    m.d.comb += exc_result.eq(0b10)
            with m.Case(0b11):
                m.d.comb += exc_result.eq(0b11)

        result_sign = Signal(name="result_sign")
        m.d.comb += result_sign.eq(0)

        exp_odd = Signal(name="exp_odd")
        m.d.comb += exp_odd.eq(~a_exp[0])

        sig_a = Signal(wf + 1, name="sig_a")
        m.d.comb += sig_a.eq(Cat(a_mant, Const(1, 1)))

        radicand_width = wf + 2
        radicand = Signal(radicand_width, name="radicand")
        with m.If(exp_odd):
            m.d.comb += radicand.eq(sig_a << 1)
        with m.Else():
            m.d.comb += radicand.eq(sig_a)

        exp_biased_sum = Signal(we + 1, name="exp_biased_sum")
        m.d.comb += exp_biased_sum.eq(a_exp + fmt.bias)
        result_exp = Signal(we + 2, name="result_exp")
        m.d.comb += result_exp.eq(exp_biased_sum >> 1)

        n_bits = wf + 3
        rad_total = 2 * n_bits
        radicand_ext = Signal(rad_total, name="radicand_ext")
        m.d.comb += radicand_ext.eq(radicand << (rad_total - radicand_width))

        # ── Stage 0 → 1 pipeline register ──
        exc_result_r1 = Signal(2, name="exc_result_r1")
        result_exp_r1 = Signal(we + 2, name="result_exp_r1")
        radicand_ext_r1 = Signal(rad_total, name="radicand_ext_r1")
        m.d.sync += [
            exc_result_r1.eq(exc_result),
            result_exp_r1.eq(result_exp),
            radicand_ext_r1.eq(radicand_ext),
        ]
        for s in [exc_result_r1, result_exp_r1, radicand_ext_r1]:
            self.add_latency(s, 1)

        # ==================================================================
        # Stage 1: first half of sqrt iterations
        # ==================================================================
        rem_width = rad_total + 2
        half = n_bits // 2

        # Use running Q accumulator for correct trial computation
        rem_cur = Signal(rem_width, name="rem_init")
        m.d.comb += rem_cur.eq(0)
        q_cur = Signal(rem_width, name="q_init")
        m.d.comb += q_cur.eq(0)

        for i in range(half):
            bit_hi_pos = rad_total - 1 - 2 * i
            bit_lo_pos = rad_total - 2 - 2 * i

            rem_shifted = Signal(rem_width, name=f"rem_shifted_1_{i}")
            if bit_lo_pos >= 0:
                m.d.comb += rem_shifted.eq(
                    (rem_cur << 2) | (radicand_ext_r1[bit_lo_pos:bit_hi_pos + 1])
                )
            else:
                m.d.comb += rem_shifted.eq(rem_cur << 2)

            trial = Signal(rem_width, name=f"trial_1_{i}")
            m.d.comb += trial.eq((q_cur << 2) | 1)

            diff = Signal(rem_width, name=f"sqrt_diff_1_{i}")
            m.d.comb += diff.eq(rem_shifted - trial)

            q_bit = Signal(name=f"sqrt_bit_1_{i}")
            m.d.comb += q_bit.eq(~diff[rem_width - 1])

            rem_next = Signal(rem_width, name=f"sqrt_rem_1_{i}")
            with m.If(q_bit):
                m.d.comb += rem_next.eq(diff)
            with m.Else():
                m.d.comb += rem_next.eq(rem_shifted)

            q_next = Signal(rem_width, name=f"q_1_{i}")
            with m.If(q_bit):
                m.d.comb += q_next.eq((q_cur << 1) | 1)
            with m.Else():
                m.d.comb += q_next.eq(q_cur << 1)

            rem_cur = rem_next
            q_cur = q_next

        # Pack Q accumulator for pipeline
        partial_q = Signal(rem_width, name="partial_q")
        m.d.comb += partial_q.eq(q_cur)

        # ── Stage 1 → 2 pipeline register ──
        q_r2 = Signal(rem_width, name="q_r2")
        rem_r2 = Signal(rem_width, name="rem_r2")
        radicand_ext_r2 = Signal(rad_total, name="radicand_ext_r2")
        exc_result_r2 = Signal(2, name="exc_result_r2")
        result_exp_r2 = Signal(we + 2, name="result_exp_r2")
        m.d.sync += [
            q_r2.eq(partial_q),
            rem_r2.eq(rem_cur),
            radicand_ext_r2.eq(radicand_ext_r1),
            exc_result_r2.eq(exc_result_r1),
            result_exp_r2.eq(result_exp_r1),
        ]
        for s in [q_r2, rem_r2, radicand_ext_r2, exc_result_r2, result_exp_r2]:
            self.add_latency(s, 2)

        # ==================================================================
        # Stage 2: second half of sqrt iterations
        # ==================================================================
        # Stage 2: continue iteration with Q accumulator from stage 1
        rem_s2 = rem_r2
        q_s2 = q_r2

        for i in range(half, n_bits):
            bit_hi_pos = rad_total - 1 - 2 * i
            bit_lo_pos = rad_total - 2 - 2 * i

            rem_shifted = Signal(rem_width, name=f"rem_shifted_2_{i}")
            if bit_lo_pos >= 0:
                m.d.comb += rem_shifted.eq(
                    (rem_s2 << 2) | (radicand_ext_r2[bit_lo_pos:bit_hi_pos + 1])
                )
            else:
                m.d.comb += rem_shifted.eq(rem_s2 << 2)

            trial = Signal(rem_width, name=f"trial_2_{i}")
            m.d.comb += trial.eq((q_s2 << 2) | 1)

            diff = Signal(rem_width, name=f"sqrt_diff_2_{i}")
            m.d.comb += diff.eq(rem_shifted - trial)

            q_bit = Signal(name=f"sqrt_bit_2_{i}")
            m.d.comb += q_bit.eq(~diff[rem_width - 1])

            rem_next = Signal(rem_width, name=f"sqrt_rem_2_{i}")
            with m.If(q_bit):
                m.d.comb += rem_next.eq(diff)
            with m.Else():
                m.d.comb += rem_next.eq(rem_shifted)

            q_next = Signal(rem_width, name=f"q_2_{i}")
            with m.If(q_bit):
                m.d.comb += q_next.eq((q_s2 << 1) | 1)
            with m.Else():
                m.d.comb += q_next.eq(q_s2 << 1)

            rem_s2 = rem_next
            q_s2 = q_next

        # Final Q is the sqrt result
        sqrt_result = Signal(n_bits, name="sqrt_result")
        m.d.comb += sqrt_result.eq(q_s2[:n_bits])

        final_rem_nonzero = Signal(name="final_rem_nonzero")
        m.d.comb += final_rem_nonzero.eq(rem_s2.any())

        # ── Stage 2 → 3 pipeline register ──
        sqrt_result_r3 = Signal(n_bits, name="sqrt_result_r3")
        final_rem_nz_r3 = Signal(name="final_rem_nz_r3")
        exc_result_r3 = Signal(2, name="exc_result_r3")
        result_exp_r3 = Signal(we + 2, name="result_exp_r3")
        m.d.sync += [
            sqrt_result_r3.eq(sqrt_result),
            final_rem_nz_r3.eq(final_rem_nonzero),
            exc_result_r3.eq(exc_result_r2),
            result_exp_r3.eq(result_exp_r2),
        ]
        for s in [sqrt_result_r3, final_rem_nz_r3, exc_result_r3, result_exp_r3]:
            self.add_latency(s, 3)

        # ==================================================================
        # Stage 3: normalization, rounding
        # ==================================================================
        round_mant = Signal(wf, name="round_mant")
        guard_bit = Signal(name="guard_bit")
        round_bit = Signal(name="round_bit")
        sticky_val = Signal(name="sticky_val")

        m.d.comb += [
            round_mant.eq(sqrt_result_r3[n_bits - 1 - wf:n_bits - 1]),
            guard_bit.eq(sqrt_result_r3[n_bits - 1 - wf - 1]),
        ]

        if n_bits - 1 - wf - 2 >= 0:
            m.d.comb += round_bit.eq(sqrt_result_r3[n_bits - 1 - wf - 2])
        else:
            m.d.comb += round_bit.eq(0)

        sticky_start = n_bits - 1 - wf - 3
        if sticky_start >= 0:
            m.d.comb += sticky_val.eq(
                sqrt_result_r3[:sticky_start + 1].any() | final_rem_nz_r3
            )
        else:
            m.d.comb += sticky_val.eq(final_rem_nz_r3)

        round_in = Signal(wf + 3, name="round_in")
        m.d.comb += round_in.eq(Cat(sticky_val, round_bit, guard_bit, round_mant))

        rounder = RoundingUnit(wf)
        m.submodules.rounder = rounder
        m.d.comb += rounder.mantissa_in.eq(round_in)

        rounded_mant = rounder.mantissa_out
        round_overflow = rounder.overflow

        final_exp = Signal(we + 2, name="final_exp")
        m.d.comb += final_exp.eq(result_exp_r3 + round_overflow)

        # ── Stage 3 → 4 pipeline register ──
        rounded_mant_r4 = Signal(wf, name="rounded_mant_r4")
        final_exp_r4 = Signal(we + 2, name="final_exp_r4")
        exc_result_r4 = Signal(2, name="exc_result_r4")
        m.d.sync += [
            rounded_mant_r4.eq(rounded_mant),
            final_exp_r4.eq(final_exp),
            exc_result_r4.eq(exc_result_r3),
        ]
        for s in [rounded_mant_r4, final_exp_r4, exc_result_r4]:
            self.add_latency(s, 4)

        # ==================================================================
        # Stage 4: final mux, pack
        # ==================================================================
        exp_top = Signal(2, name="exp_top")
        m.d.comb += exp_top.eq(final_exp_r4[we:we + 2])

        exc_post_norm = Signal(2, name="exc_post_norm")
        with m.Switch(exp_top):
            with m.Case(0b00): m.d.comb += exc_post_norm.eq(0b01)
            with m.Case(0b01): m.d.comb += exc_post_norm.eq(0b10)
            with m.Case(0b10): m.d.comb += exc_post_norm.eq(0b00)
            with m.Case(0b11): m.d.comb += exc_post_norm.eq(0b00)

        final_exc = Signal(2, name="final_exc")
        with m.If((exc_result_r4 == 0b11) | (exc_result_r4 == 0b10) | (exc_result_r4 == 0b00)):
            m.d.comb += final_exc.eq(exc_result_r4)
        with m.Else():
            m.d.comb += final_exc.eq(exc_post_norm)

        final_mant = Signal(wf, name="final_mant")
        final_e = Signal(we, name="final_e")

        with m.If(final_exc == 0b01):
            m.d.comb += [
                final_mant.eq(rounded_mant_r4),
                final_e.eq(final_exp_r4[:we]),
            ]
        with m.Else():
            m.d.comb += [
                final_mant.eq(0),
                final_e.eq(0),
            ]

        result_sign_val = Signal(name="result_sign_val")
        m.d.comb += result_sign_val.eq(0)

        # ── Stage 4 → 5 pipeline register (output) ──
        o_r5 = Signal(fmt.width, name="o_r5")
        m.d.sync += o_r5.eq(Cat(final_mant, final_e, result_sign_val, final_exc))
        m.d.comb += self.o.eq(o_r5)

        return m
