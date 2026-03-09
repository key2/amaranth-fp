"""Floating-point acos(x) = pi/2 - asin(x).

Wraps FPAsin + subtraction from pi/2 constant. Latency: FPAsin.latency + 7.
"""
from __future__ import annotations

import math

from amaranth import *

from ...format import FPFormat
from ...pipelined import PipelinedComponent
from ...operators.fp_sub import FPSub
from .fp_asin import FPAsin

__all__ = ["FPAcos"]


class FPAcos(PipelinedComponent):
    """Pipelined floating-point acos(x) = pi/2 - asin(x).

    Parameters
    ----------
    fmt : FPFormat
    """

    def __init__(self, fmt: FPFormat) -> None:
        super().__init__()
        self.fmt = fmt
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")

        self._asin = FPAsin(fmt)
        self._sub = FPSub(fmt)
        self.latency = self._asin.latency + 7

    def _encode_pi_half(self) -> int:
        """Encode pi/2 in FloPoCo internal format."""
        fmt = self.fmt
        we, wf, bias = fmt.we, fmt.wf, fmt.bias
        val = math.pi / 2
        exp_unbiased = int(math.floor(math.log2(val)))
        sig = val / (2.0 ** exp_unbiased)
        frac = int(round((sig - 1.0) * (1 << wf))) & ((1 << wf) - 1)
        exp_biased = exp_unbiased + bias
        exc = 0b01
        sign = 0
        return (exc << (1 + we + wf)) | (sign << (we + wf)) | (exp_biased << wf) | frac

    def elaborate(self, platform) -> Module:
        m = Module()
        m.submodules.asin = asin = self._asin
        m.submodules.sub = sub = self._sub

        pi_half_enc = self._encode_pi_half()

        m.d.comb += [
            asin.a.eq(self.a),
            sub.a.eq(pi_half_enc),
            sub.b.eq(asin.o),
            self.o.eq(sub.o),
        ]
        return m
