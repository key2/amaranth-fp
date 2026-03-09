"""Evaluate arbitrary function via direct table lookup."""
from __future__ import annotations

from typing import Callable

from amaranth import *

from ..pipelined import PipelinedComponent
from .table import Table

__all__ = ["FixFunctionByTable"]


class FixFunctionByTable(PipelinedComponent):
    """Evaluate a function by exhaustive table lookup.

    At construction time, evaluates *func* at all 2^input_width points
    uniformly spaced over *input_range* and stores the results in a ROM.

    Parameters
    ----------
    input_width : int
    output_width : int
    func : Callable[[float], float]
    input_range : tuple[float, float]
    """

    def __init__(
        self,
        input_width: int,
        output_width: int,
        func: Callable[[float], float],
        input_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.func = func
        self.input_range = input_range

        n = 1 << input_width
        lo, hi = input_range
        step = (hi - lo) / n
        out_max = (1 << output_width) - 1
        values: list[int] = []
        for i in range(n):
            x = lo + (i + 0.5) * step
            y = func(x)
            v = int(round(y)) & out_max
            values.append(v)

        self._values = values
        self.addr = Signal(input_width, name="addr")
        self.data = Signal(output_width, name="data")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()

        tbl = Table(self.input_width, self.output_width, self._values)
        m.submodules.tbl = tbl

        m.d.comb += [
            tbl.addr.eq(self.addr),
            self.data.eq(tbl.data),
        ]

        return m
