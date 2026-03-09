"""FP dot product of two vectors (pipelined)."""
from __future__ import annotations

from math import ceil, log2

from amaranth import *

from ..format import FPFormat
from ..pipelined import PipelinedComponent
from .fp_mul import FPMul
from .fp_add import FPAdd

__all__ = ["FPDotProduct"]


class FPDotProduct(PipelinedComponent):
    """Pipelined dot product of two n-element vectors.

    Parameters
    ----------
    fmt : FPFormat
    n : int
        Vector length.

    Attributes
    ----------
    a : list[Signal], in — first vector elements
    b : list[Signal], in — second vector elements
    o : Signal, out — dot product result
    """

    def __init__(self, fmt: FPFormat, n: int) -> None:
        super().__init__()
        self.fmt = fmt
        self.n = n
        self.a = [Signal(fmt.width, name=f"a_{i}") for i in range(n)]
        self.b = [Signal(fmt.width, name=f"b_{i}") for i in range(n)]
        self.o = Signal(fmt.width, name="o")

        mul_lat = FPMul(fmt).latency
        add_lat = FPAdd(fmt).latency
        tree_depth = int(ceil(log2(max(n, 2))))
        self.latency = mul_lat + tree_depth * add_lat

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        n = self.n

        # n multipliers
        products = []
        for i in range(n):
            mul = FPMul(fmt)
            setattr(m.submodules, f"mul_{i}", mul)
            m.d.comb += [mul.a.eq(self.a[i]), mul.b.eq(self.b[i])]
            products.append(mul.o)

        # Reduction tree of adders
        level = products
        level_idx = 0
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    add = FPAdd(fmt)
                    setattr(m.submodules, f"add_{level_idx}_{i}", add)
                    m.d.comb += [add.a.eq(level[i]), add.b.eq(level[i + 1])]
                    next_level.append(add.o)
                else:
                    # Odd element: delay to match adder latency
                    delayed = level[i]
                    add_lat = FPAdd(fmt).latency
                    for d in range(add_lat):
                        nxt = Signal(fmt.width, name=f"dp_d{level_idx}_{i}_{d}")
                        m.d.sync += nxt.eq(delayed)
                        delayed = nxt
                    next_level.append(delayed)
            level = next_level
            level_idx += 1

        m.d.comb += self.o.eq(level[0])

        return m
