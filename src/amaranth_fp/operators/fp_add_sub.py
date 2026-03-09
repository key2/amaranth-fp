"""Combined FP add/subtract with operation select (pipelined, 7 stages)."""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from .fp_add import FPAdd

__all__ = ["FPAddSub"]


class FPAddSub(PipelinedComponent):
    """FP add/subtract: flips b sign if sub, delegates to FPAdd (7-cycle latency).

    Parameters
    ----------
    fmt : FPFormat

    Attributes
    ----------
    a, b : Signal(fmt.width), in
    op : Signal(1), in — 0=add, 1=sub
    o : Signal(fmt.width), out
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.b = Signal(fmt.width, name="b")
        self.op = Signal(1, name="op")
        self.o = Signal(fmt.width, name="o")
        self.latency = 7

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we = fmt.we
        wf = fmt.wf

        adder = FPAdd(fmt)
        m.submodules.adder = adder

        sign_pos = wf + we
        b_mod = Signal(fmt.width, name="b_mod")
        m.d.comb += [
            b_mod[:sign_pos].eq(self.b[:sign_pos]),
            b_mod[sign_pos].eq(self.b[sign_pos] ^ self.op),
            b_mod[sign_pos + 1:].eq(self.b[sign_pos + 1:]),
        ]

        m.d.comb += [
            adder.a.eq(self.a),
            adder.b.eq(b_mod),
            self.o.eq(adder.o),
        ]
        return m
