"""Fixed-point × fixed-point constant multiplier (pipelined, 2 stages)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixFixConstMult"]


class FixFixConstMult(PipelinedComponent):
    """Fixed-point constant multiplier (2-cycle latency).

    Parameters
    ----------
    input_width : int
    constant_width : int
    constant : int — the fixed-point constant value
    """

    def __init__(self, input_width: int, constant_width: int, constant: int) -> None:
        super().__init__()
        self.input_width = input_width
        self.constant_width = constant_width
        self.constant = constant
        self.x = Signal(input_width, name="x")
        self.p = Signal(input_width + constant_width, name="p")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        pw = self.input_width + self.constant_width

        prod = Signal(pw, name="prod")
        m.d.comb += prod.eq(self.x * self.constant)

        prod_r1 = Signal(pw, name="prod_r1")
        m.d.sync += prod_r1.eq(prod)

        prod_r2 = Signal(pw, name="prod_r2")
        m.d.sync += prod_r2.eq(prod_r1)
        m.d.comb += self.p.eq(prod_r2)

        return m
