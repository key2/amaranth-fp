"""Floating-point acosh(x) = log(x + sqrt(x^2 - 1)).

Composed from FPMul + FPSub + FPSqrt + FPAdd + FPLog.
"""
from __future__ import annotations

from amaranth import *

from ...format import FPFormat
from ...pipelined import PipelinedComponent
from ...operators.fp_mul import FPMul
from ...operators.fp_sub import FPSub
from ...operators.fp_add import FPAdd
from ...operators.fp_sqrt import FPSqrt
from ...operators.fp_log import FPLog

__all__ = ["FPAcosh"]


class FPAcosh(PipelinedComponent):
    """Pipelined floating-point acosh(x) = log(x + sqrt(x^2 - 1)).

    Parameters
    ----------
    fmt : FPFormat
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")

        self._square = FPMul(fmt)
        self._sub = FPSub(fmt)
        self._sqrt = FPSqrt(fmt)
        self._add = FPAdd(fmt)
        self._log = FPLog(fmt)

        # FPSub.latency is set during elaborate(); use FPAdd latency as proxy
        sub_latency = getattr(self._sub, 'latency', self._add.latency)
        self.latency = (self._square.latency + sub_latency +
                        self._sqrt.latency + self._add.latency + self._log.latency)

    def _encode_one(self) -> int:
        fmt = self.fmt
        return (0b01 << (1 + fmt.we + fmt.wf)) | (fmt.bias << fmt.wf)

    def elaborate(self, platform) -> Module:
        m = Module()
        m.submodules.square = sq = self._square
        m.submodules.sub = sub = self._sub
        m.submodules.sqrt = sqrt = self._sqrt
        m.submodules.add = add = self._add
        m.submodules.log = log = self._log

        one_enc = self._encode_one()

        m.d.comb += [sq.a.eq(self.a), sq.b.eq(self.a)]
        m.d.comb += [sub.a.eq(sq.o), sub.b.eq(one_enc)]
        m.d.comb += sqrt.a.eq(sub.o)

        x_delayed = self.a
        for i in range(sq.latency + sub.latency + sqrt.latency):
            d = Signal(self.fmt.width, name=f"x_delay_{i}")
            m.d.sync += d.eq(x_delayed)
            x_delayed = d
        m.d.comb += [add.a.eq(x_delayed), add.b.eq(sqrt.o)]
        m.d.comb += [log.a.eq(add.o), self.o.eq(log.o)]

        return m
