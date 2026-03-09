"""Floating-point atanh(x) = 0.5 * log((1+x)/(1-x)).

Composed from FPAdd + FPSub + FPDiv + FPLog + FPConstMult.
"""
from __future__ import annotations

from amaranth import *

from ...format import FPFormat
from ...pipelined import PipelinedComponent
from ...operators.fp_add import FPAdd
from ...operators.fp_sub import FPSub
from ...operators.fp_div import FPDiv
from ...operators.fp_log import FPLog
from ...operators.fp_const_mult import FPConstMult

__all__ = ["FPAtanh"]


class FPAtanh(PipelinedComponent):
    """Pipelined floating-point atanh(x) = 0.5 * log((1+x)/(1-x)).

    Parameters
    ----------
    fmt : FPFormat
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")

        self._add = FPAdd(fmt)      # 1 + x
        self._sub = FPSub(fmt)      # 1 - x
        self._div = FPDiv(fmt)      # (1+x)/(1-x)
        self._log = FPLog(fmt)      # log((1+x)/(1-x))
        self._half = FPConstMult(fmt, 0.5)

        # FPSub.latency is set during elaborate(); use FPAdd latency as proxy
        sub_latency = getattr(self._sub, 'latency', self._add.latency)
        self.latency = (max(self._add.latency, sub_latency) +
                        self._div.latency + self._log.latency + self._half.latency)

    def _encode_one(self) -> int:
        fmt = self.fmt
        return (0b01 << (1 + fmt.we + fmt.wf)) | (fmt.bias << fmt.wf)

    def elaborate(self, platform) -> Module:
        m = Module()
        m.submodules.add = add = self._add
        m.submodules.sub = sub = self._sub
        m.submodules.div = div = self._div
        m.submodules.log = log = self._log
        m.submodules.half = half = self._half

        one_enc = self._encode_one()

        # 1 + x and 1 - x in parallel
        m.d.comb += [add.a.eq(one_enc), add.b.eq(self.a)]
        m.d.comb += [sub.a.eq(one_enc), sub.b.eq(self.a)]

        # Equalize add/sub latency (they should be the same but just in case)
        add_out = add.o
        sub_out = sub.o
        if add.latency < sub.latency:
            for i in range(sub.latency - add.latency):
                d = Signal(self.fmt.width, name=f"add_delay_{i}")
                m.d.sync += d.eq(add_out)
                add_out = d
        elif sub.latency < add.latency:
            for i in range(add.latency - sub.latency):
                d = Signal(self.fmt.width, name=f"sub_delay_{i}")
                m.d.sync += d.eq(sub_out)
                sub_out = d

        m.d.comb += [div.a.eq(add_out), div.b.eq(sub_out)]
        m.d.comb += log.a.eq(div.o)
        m.d.comb += [half.a.eq(log.o), self.o.eq(half.o)]

        return m
