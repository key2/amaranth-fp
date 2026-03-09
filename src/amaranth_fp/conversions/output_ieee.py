"""Convert FloPoCo internal format to IEEE 754 floating-point (pipelined, 1 stage)."""

from __future__ import annotations

from amaranth import Elaboratable, Module, Signal

from amaranth_fp.format import FPFormat, ieee_layout, internal_layout
from amaranth_fp.pipelined import PipelinedComponent


class OutputIEEE(PipelinedComponent):
    """Convert a FloPoCo internal representation to IEEE 754 encoding (1-cycle latency).

    Parameters
    ----------
    fmt : FPFormat
        Floating-point format descriptor.

    Attributes
    ----------
    fp_in : Signal(fmt.width)
    ieee_out : Signal(fmt.ieee_width)
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.fp_in = Signal(fmt.width, name="fp_in")
        self.ieee_out = Signal(fmt.ieee_width, name="ieee_out")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt

        # ==================================================================
        # Stage 0: unpack internal, switch on exception, pack IEEE
        # ==================================================================
        fp = Signal(internal_layout(fmt))
        m.d.comb += fp.eq(self.fp_in)

        exc = Signal(2, name="exception")
        sign = Signal(1, name="sign")
        exponent = Signal(fmt.we, name="exponent")
        mantissa = Signal(fmt.wf, name="mantissa")

        m.d.comb += [
            exc.eq(fp.exception),
            sign.eq(fp.sign),
            exponent.eq(fp.exponent),
            mantissa.eq(fp.mantissa),
        ]

        out_sign = Signal(1, name="out_sign")
        out_exp = Signal(fmt.we, name="out_exp")
        out_mant = Signal(fmt.wf, name="out_mant")

        all_ones_exp = (1 << fmt.we) - 1

        with m.Switch(exc):
            with m.Case(0b00):
                m.d.comb += [out_sign.eq(sign), out_exp.eq(0), out_mant.eq(0)]
            with m.Case(0b01):
                m.d.comb += [out_sign.eq(sign), out_exp.eq(exponent), out_mant.eq(mantissa)]
            with m.Case(0b10):
                m.d.comb += [out_sign.eq(sign), out_exp.eq(all_ones_exp), out_mant.eq(0)]
            with m.Case(0b11):
                m.d.comb += [out_sign.eq(0), out_exp.eq(all_ones_exp), out_mant.eq(1 << (fmt.wf - 1))]

        out = Signal(ieee_layout(fmt))
        m.d.comb += [
            out.mantissa.eq(out_mant),
            out.exponent.eq(out_exp),
            out.sign.eq(out_sign),
        ]

        # ── Stage 0 → 1 pipeline register (output) ──
        ieee_out_r1 = Signal(fmt.ieee_width, name="ieee_out_r1")
        m.d.sync += ieee_out_r1.eq(out)
        m.d.comb += self.ieee_out.eq(ieee_out_r1)

        return m
