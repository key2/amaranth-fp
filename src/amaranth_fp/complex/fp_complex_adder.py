"""Floating-point complex addition (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent
from ..format import FPFormat
from ..operators.fp_add import FPAdd

__all__ = ["FPComplexAdder"]


class FPComplexAdder(PipelinedComponent):
    """FP complex addition using two FPAdd instances.

    Parameters
    ----------
    fmt : FPFormat
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        w = fmt.width
        self.a_re = Signal(w, name="a_re")
        self.a_im = Signal(w, name="a_im")
        self.b_re = Signal(w, name="b_re")
        self.b_im = Signal(w, name="b_im")
        self.o_re = Signal(w, name="o_re")
        self.o_im = Signal(w, name="o_im")
        self._add_re = FPAdd(fmt)
        self._add_im = FPAdd(fmt)
        self.latency = self._add_re.latency

    def elaborate(self, platform) -> Module:
        m = Module()
        m.submodules.add_re = add_re = self._add_re
        m.submodules.add_im = add_im = self._add_im

        m.d.comb += [
            add_re.a.eq(self.a_re), add_re.b.eq(self.b_re),
            add_im.a.eq(self.a_im), add_im.b.eq(self.b_im),
            self.o_re.eq(add_re.o), self.o_im.eq(add_im.o),
        ]

        return m
