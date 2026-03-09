"""3-input FP adder (pipelined). Uses two FPAdd instances: (a+b)+c."""
from __future__ import annotations

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from .fp_add import FPAdd

__all__ = ["FPAdd3Input"]


class FPAdd3Input(PipelinedComponent):
    """Pipelined 3-input floating-point adder: o = (a + b) + c.

    Parameters
    ----------
    fmt : FPFormat
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.b = Signal(fmt.width, name="b")
        self.c = Signal(fmt.width, name="c")
        self.o = Signal(fmt.width, name="o")

        self._add1 = FPAdd(fmt)
        self._add2 = FPAdd(fmt)
        self.latency = 2 * self._add1.latency

    def elaborate(self, platform) -> Module:
        m = Module()
        add1 = self._add1
        add2 = self._add2
        m.submodules.add1 = add1
        m.submodules.add2 = add2

        # First adder: a + b
        m.d.comb += [add1.a.eq(self.a), add1.b.eq(self.b)]

        # Delay c to align with add1 output
        c_delayed = self.c
        for i in range(add1.latency):
            c_next = Signal(self.fmt.width, name=f"c_d{i}")
            m.d.sync += c_next.eq(c_delayed)
            c_delayed = c_next

        # Second adder: (a+b) + c
        m.d.comb += [add2.a.eq(add1.o), add2.b.eq(c_delayed)]
        m.d.comb += self.o.eq(add2.o)

        return m
