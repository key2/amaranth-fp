"""Parameterized integer multiplier (pipelined, 2 stages)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["IntMultiplier"]


class IntMultiplier(PipelinedComponent):
    """Pipelined integer multiplier (2-cycle latency).

    Parameters
    ----------
    a_width, b_width : int
    signed_a, signed_b : bool
    """

    def __init__(self, a_width: int, b_width: int, *, signed_a: bool = False, signed_b: bool = False) -> None:
        super().__init__()
        self.a_width = a_width
        self.b_width = b_width
        self.signed_a = signed_a
        self.signed_b = signed_b
        self.a = Signal(Shape(a_width, signed_a), name="a")
        self.b = Signal(Shape(b_width, signed_b), name="b")
        p_width = a_width + b_width
        self.p = Signal(Shape(p_width, signed_a or signed_b), name="p")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()

        prod = Signal(self.p.shape(), name="prod")
        m.d.comb += prod.eq(self.a * self.b)

        prod_r1 = Signal(self.p.shape(), name="prod_r1")
        m.d.sync += prod_r1.eq(prod)

        prod_r2 = Signal(self.p.shape(), name="prod_r2")
        m.d.sync += prod_r2.eq(prod_r1)
        m.d.comb += self.p.eq(prod_r2)

        return m
