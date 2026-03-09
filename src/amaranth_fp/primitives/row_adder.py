"""Row adder primitive for bitheap compression (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["RowAdder"]


class RowAdder(PipelinedComponent):
    """Multi-operand row adder (carry-save style).

    Adds *n_inputs* values of *width* bits, producing a sum.

    Parameters
    ----------
    width : int
    n_inputs : int
    """

    def __init__(self, width: int, n_inputs: int = 3) -> None:
        super().__init__()
        self.width = width
        self.n_inputs = n_inputs
        self.inputs = [Signal(width, name=f"in{i}") for i in range(n_inputs)]
        out_bits = width + (n_inputs - 1).bit_length()
        self.o = Signal(out_bits, name="o")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        accum = Signal.like(self.o, name="accum")
        total = self.inputs[0]
        for inp in self.inputs[1:]:
            s = Signal.like(self.o, name="partial")
            m.d.comb += s.eq(total + inp)
            total = s
        m.d.comb += accum.eq(total)
        o_r = Signal.like(self.o, name="o_r")
        m.d.sync += o_r.eq(accum)
        m.d.comb += self.o.eq(o_r)
        return m
