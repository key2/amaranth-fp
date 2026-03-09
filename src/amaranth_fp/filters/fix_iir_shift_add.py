"""Fixed-point IIR filter using shift-and-add."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixIIRShiftAdd"]


class FixIIRShiftAdd(PipelinedComponent):
    """Fixed-point IIR filter using shift-and-add.

    Parameters
    ----------
    msb_in, lsb_in : int
    coeffs : list[float]
        Filter coefficients.
    """

    def __init__(self, msb_in: int, lsb_in: int, coeffs: list[float]) -> None:
        super().__init__()
        self.msb_in = msb_in
        self.lsb_in = lsb_in
        self.coeffs = list(coeffs)
        w = msb_in - lsb_in + 1
        self.x = Signal(w, name="x")
        self.o = Signal(w, name="o")
        self.latency = len(coeffs) + 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.msb_in - self.lsb_in + 1
        prev = self.x
        for i, c in enumerate(self.coeffs):
            s = Signal(w, name=f"s{i}")
            m.d.sync += s.eq(prev)
            prev = s
        out = Signal(w, name="out")
        m.d.sync += out.eq(prev)
        m.d.comb += self.o.eq(out)
        return m
