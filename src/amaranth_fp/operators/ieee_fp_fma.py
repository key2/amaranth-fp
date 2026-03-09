"""Direct IEEE format FMA (pipelined, 11 stages).

Wraps InputIEEE → FPFMA → OutputIEEE.
"""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from ..conversions.input_ieee import InputIEEE
from ..conversions.output_ieee import OutputIEEE
from .fp_fma import FPFMA

__all__ = ["IEEEFPFMA"]


class IEEEFPFMA(PipelinedComponent):
    """IEEE-format FMA (11-cycle latency = 1 + 9 + 1).

    Parameters
    ----------
    fmt : FPFormat

    Attributes
    ----------
    a, b, c : Signal(fmt.ieee_width), in — IEEE format
    o : Signal(fmt.ieee_width), out — IEEE format
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.ieee_width, name="a")
        self.b = Signal(fmt.ieee_width, name="b")
        self.c = Signal(fmt.ieee_width, name="c")
        self.o = Signal(fmt.ieee_width, name="o")
        self.latency = 11

    def elaborate(self, platform) -> Module:
        m = Module()

        in_a = InputIEEE(self.fmt)
        in_b = InputIEEE(self.fmt)
        in_c = InputIEEE(self.fmt)
        fma = FPFMA(self.fmt)
        out_conv = OutputIEEE(self.fmt)

        m.submodules.in_a = in_a
        m.submodules.in_b = in_b
        m.submodules.in_c = in_c
        m.submodules.fma = fma
        m.submodules.out_conv = out_conv

        m.d.comb += [
            in_a.ieee_in.eq(self.a),
            in_b.ieee_in.eq(self.b),
            in_c.ieee_in.eq(self.c),
            fma.a.eq(in_a.fp_out),
            fma.b.eq(in_b.fp_out),
            fma.c.eq(in_c.fp_out),
            out_conv.fp_in.eq(fma.o),
            self.o.eq(out_conv.ieee_out),
        ]
        return m
