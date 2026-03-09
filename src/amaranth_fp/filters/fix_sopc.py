"""Sum of products with constants (pipelined, 2 stages)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixSOPC"]


class FixSOPC(PipelinedComponent):
    """Sum of products with constants: sum(c[i] * x[i]) (2-cycle latency).

    Parameters
    ----------
    input_width : int
    n_inputs : int
    constants : list[int]
    output_width : int
    """

    def __init__(self, input_width: int, n_inputs: int,
                 constants: list[int], output_width: int) -> None:
        super().__init__()
        self.input_width = input_width
        self.n_inputs = n_inputs
        self.constants = list(constants)
        self.output_width = output_width
        self.x = [Signal(input_width, name=f"x_{i}") for i in range(n_inputs)]
        self.y = Signal(output_width, name="y")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        iw = self.input_width
        ow = self.output_width
        n = self.n_inputs

        # Stage 1: compute products
        prods = []
        for i in range(n):
            p = Signal(iw + 32, name=f"prod_{i}")  # generous width
            m.d.comb += p.eq(self.x[i] * self.constants[i])
            p_r = Signal(iw + 32, name=f"prod_r_{i}")
            m.d.sync += p_r.eq(p)
            prods.append(p_r)

        # Stage 2: sum
        if prods:
            s = prods[0]
            for p in prods[1:]:
                s2 = Signal(iw + 32 + n, name=f"sum_{id(p)}")
                m.d.comb += s2.eq(s + p)
                s = s2
            y_r = Signal(ow, name="y_r")
            m.d.sync += y_r.eq(s[:ow])
            m.d.comb += self.y.eq(y_r)
        else:
            m.d.comb += self.y.eq(0)

        return m
