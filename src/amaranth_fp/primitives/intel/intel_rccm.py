"""Intel Reconfigurable Constant Coefficient Multiplier."""
from __future__ import annotations
from amaranth import *
from ...pipelined import PipelinedComponent

__all__ = ["IntelRCCM"]


class IntelRCCM(PipelinedComponent):
    """Intel RCCM primitive.

    Parameters
    ----------
    width : int
    constant : int
    """

    def __init__(self, width: int, constant: int = 1):
        super().__init__()
        self.width = width
        self.constant = constant
        self.x = Signal(width, name="x")
        self.o = Signal(2 * width, name="o")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        out = Signal(2 * self.width, name="out")
        m.d.sync += out.eq(self.x * self.constant)
        m.d.comb += self.o.eq(out)
        return m
