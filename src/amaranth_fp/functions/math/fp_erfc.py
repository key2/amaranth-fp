"""Floating-point erfc(x) = 1 - erf(x).

Wraps FPErf + FPSub. Latency: FPErf.latency + FPSub.latency.
"""
from __future__ import annotations

from amaranth import *

from ...format import FPFormat, float_to_flopoco
from ...pipelined import PipelinedComponent
from ...operators.fp_sub import FPSub
from .fp_erf import FPErf

__all__ = ["FPErfc"]


class FPErfc(PipelinedComponent):
    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")

        self._erf = FPErf(fmt)
        self._sub = FPSub(fmt)
        self.latency = self._erf.latency + self._sub.latency

    def _encode_one(self) -> int:
        fmt = self.fmt
        return float_to_flopoco(1.0, fmt.we, fmt.wf, fmt.bias)

    def elaborate(self, platform) -> Module:
        m = Module()
        m.submodules.erf = erf = self._erf
        m.submodules.sub = sub = self._sub

        one_enc = self._encode_one()

        # Delay the constant "1" by erf.latency cycles so it arrives at sub
        # at the same time as the erf output
        prev = Signal(self.fmt.width, name="one_d_init")
        m.d.comb += prev.eq(one_enc)
        for i in range(erf.latency):
            d = Signal(self.fmt.width, name=f"one_d{i}")
            m.d.sync += d.eq(prev)
            prev = d

        m.d.comb += [
            erf.a.eq(self.a),
            sub.a.eq(prev),
            sub.b.eq(erf.o),
            self.o.eq(sub.o),
        ]
        return m
