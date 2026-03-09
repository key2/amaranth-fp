"""Convert IEEE 754 floating-point to FloPoCo internal format (pipelined, 1 stage)."""

from __future__ import annotations

from amaranth import Elaboratable, Module, Signal, Cat
from amaranth.hdl import Mux

from amaranth_fp.format import FPFormat, ieee_layout, internal_layout
from amaranth_fp.pipelined import PipelinedComponent


class InputIEEE(PipelinedComponent):
    """Convert an IEEE 754 encoded value to FloPoCo internal representation (1-cycle latency).

    Parameters
    ----------
    fmt : FPFormat
        Floating-point format descriptor.

    Attributes
    ----------
    ieee_in : Signal(fmt.ieee_width)
    fp_out : Signal(fmt.width)
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.ieee_in = Signal(fmt.ieee_width, name="ieee_in")
        self.fp_out = Signal(fmt.width, name="fp_out")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt

        # ==================================================================
        # Stage 0: unpack IEEE, detect specials, pack internal
        # ==================================================================
        ieee = Signal(ieee_layout(fmt))
        m.d.comb += ieee.eq(self.ieee_in)

        sign = Signal(1, name="sign")
        exponent = Signal(fmt.we, name="exponent")
        mantissa = Signal(fmt.wf, name="mantissa")

        m.d.comb += [
            sign.eq(ieee.sign),
            exponent.eq(ieee.exponent),
            mantissa.eq(ieee.mantissa),
        ]

        exp_all_zeros = Signal(name="exp_all_zeros")
        exp_all_ones = Signal(name="exp_all_ones")
        mant_all_zeros = Signal(name="mant_all_zeros")

        m.d.comb += [
            exp_all_zeros.eq(exponent == 0),
            exp_all_ones.eq(exponent == (1 << fmt.we) - 1),
            mant_all_zeros.eq(mantissa == 0),
        ]

        exc = Signal(2, name="exception")
        out_sign = Signal(1, name="out_sign")
        out_exp = Signal(fmt.we, name="out_exp")
        out_mant = Signal(fmt.wf, name="out_mant")

        with m.If(exp_all_zeros & mant_all_zeros):
            m.d.comb += [
                exc.eq(0b00), out_sign.eq(sign), out_exp.eq(0), out_mant.eq(0),
            ]
        with m.Elif(exp_all_zeros & ~mant_all_zeros):
            m.d.comb += [
                exc.eq(0b00), out_sign.eq(sign), out_exp.eq(0), out_mant.eq(0),
            ]
        with m.Elif(exp_all_ones & mant_all_zeros):
            m.d.comb += [
                exc.eq(0b10), out_sign.eq(sign), out_exp.eq(0), out_mant.eq(0),
            ]
        with m.Elif(exp_all_ones & ~mant_all_zeros):
            m.d.comb += [
                exc.eq(0b11), out_sign.eq(0), out_exp.eq(0), out_mant.eq(0),
            ]
        with m.Else():
            m.d.comb += [
                exc.eq(0b01), out_sign.eq(sign), out_exp.eq(exponent), out_mant.eq(mantissa),
            ]

        out = Signal(internal_layout(self.fmt))
        m.d.comb += [
            out.mantissa.eq(out_mant),
            out.exponent.eq(out_exp),
            out.sign.eq(out_sign),
            out.exception.eq(exc),
        ]

        # ── Stage 0 → 1 pipeline register (output) ──
        fp_out_r1 = Signal(fmt.width, name="fp_out_r1")
        m.d.sync += fp_out_r1.eq(out)
        m.d.comb += self.fp_out.eq(fp_out_r1)

        return m
