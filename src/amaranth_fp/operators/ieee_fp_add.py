"""Direct IEEE format FP adder (pipelined, 9 stages).

Wraps InputIEEE → FPAdd → OutputIEEE.
"""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from ..conversions.input_ieee import InputIEEE
from ..conversions.output_ieee import OutputIEEE
from .fp_add import FPAdd

__all__ = ["IEEEFPAdd"]


class IEEEFPAdd(PipelinedComponent):
    """IEEE-format FP adder (9-cycle latency = 1 + 7 + 1).

    Parameters
    ----------
    fmt : FPFormat

    Attributes
    ----------
    a, b : Signal(fmt.ieee_width), in — IEEE format
    o : Signal(fmt.ieee_width), out — IEEE format
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.ieee_width, name="a")
        self.b = Signal(fmt.ieee_width, name="b")
        self.o = Signal(fmt.ieee_width, name="o")
        self.latency = 9

    def elaborate(self, platform) -> Module:
        m = Module()

        in_a = InputIEEE(self.fmt)
        in_b = InputIEEE(self.fmt)
        adder = FPAdd(self.fmt)
        out_conv = OutputIEEE(self.fmt)

        m.submodules.in_a = in_a
        m.submodules.in_b = in_b
        m.submodules.adder = adder
        m.submodules.out_conv = out_conv

        m.d.comb += [
            in_a.ieee_in.eq(self.a),
            in_b.ieee_in.eq(self.b),
            adder.a.eq(in_a.fp_out),
            adder.b.eq(in_b.fp_out),
            out_conv.fp_in.eq(adder.o),
            self.o.eq(out_conv.ieee_out),
        ]
        return m
