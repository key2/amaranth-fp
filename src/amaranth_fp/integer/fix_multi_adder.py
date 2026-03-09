"""Fixed-point multi-operand adder."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixMultiAdder"]


class FixMultiAdder(PipelinedComponent):
    """Fixed-point multi-operand adder.

    Parameters
    ----------
    width : int
    n_inputs : int
    """

    def __init__(self, width: int, n_inputs: int = 3) -> None:
        super().__init__()
        self.width = width
        self.n_inputs = n_inputs
        self.inputs = [Signal(width, name=f"x{i}") for i in range(n_inputs)]
        self.o = Signal(width + (n_inputs - 1).bit_length(), name="o")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        # Stage 0: sum all inputs
        total = Signal(w + (self.n_inputs - 1).bit_length(), name="total")
        expr = self.inputs[0]
        for inp in self.inputs[1:]:
            expr = expr + inp
        m.d.comb += total.eq(expr)
        s1 = Signal.like(total, name="s1")
        m.d.sync += s1.eq(total)
        out = Signal.like(total, name="out")
        m.d.sync += out.eq(s1)
        m.d.comb += self.o.eq(out)
        return m
