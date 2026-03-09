"""Integer constant comparator (pipelined, 1 stage)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["IntConstantComparator"]


class IntConstantComparator(PipelinedComponent):
    """Compare against a compile-time constant (1-cycle latency).

    Parameters
    ----------
    width : int
    constant : int
    signed : bool
    """

    def __init__(self, width: int, constant: int, *, signed: bool = False) -> None:
        super().__init__()
        self.width = width
        self.constant = constant
        self._signed = signed
        self.a = Signal(Shape(width, signed), name="a")
        self.lt = Signal(1, name="lt")
        self.eq = Signal(1, name="eq")
        self.gt = Signal(1, name="gt")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        c = Const(self.constant, Shape(self.width, self._signed))

        lt_c = Signal(name="lt_c")
        eq_c = Signal(name="eq_c")
        gt_c = Signal(name="gt_c")
        m.d.comb += [
            lt_c.eq(self.a < c),
            eq_c.eq(self.a == c),
            gt_c.eq(self.a > c),
        ]

        lt_r = Signal(name="lt_r")
        eq_r = Signal(name="eq_r")
        gt_r = Signal(name="gt_r")
        m.d.sync += [lt_r.eq(lt_c), eq_r.eq(eq_c), gt_r.eq(gt_c)]
        m.d.comb += [self.lt.eq(lt_r), self.eq.eq(eq_r), self.gt.eq(gt_r)]
        return m
