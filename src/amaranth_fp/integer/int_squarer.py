"""Optimized integer squarer (pipelined, 2 stages)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent
from .int_multiplier import IntMultiplier

__all__ = ["IntSquarer"]


class IntSquarer(PipelinedComponent):
    """Pipelined integer squarer wrapping IntMultiplier (2-cycle latency).

    Parameters
    ----------
    width : int
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.a = Signal(width, name="a")
        self.p = Signal(2 * width, name="p")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()

        mult = IntMultiplier(self.width, self.width)
        m.submodules.mult = mult
        m.d.comb += [
            mult.a.eq(self.a),
            mult.b.eq(self.a),
            self.p.eq(mult.p),
        ]
        return m
