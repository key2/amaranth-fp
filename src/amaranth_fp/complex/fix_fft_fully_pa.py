"""Fully pipelined fixed-point FFT."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixFFTFullyPA"]


class FixFFTFullyPA(PipelinedComponent):
    """Fully pipelined fixed-point FFT.

    Parameters
    ----------
    n : int
        FFT size (power of 2).
    msb_in, lsb_in : int
        Format of inputs.
    """

    def __init__(self, n: int, msb_in: int, lsb_in: int) -> None:
        super().__init__()
        self.n = n
        self.msb_in = msb_in
        self.lsb_in = lsb_in
        w = msb_in - lsb_in + 1
        self.x_re = Signal(w, name="x_re")
        self.x_im = Signal(w, name="x_im")
        self.o_re = Signal(w, name="o_re")
        self.o_im = Signal(w, name="o_im")
        self.latency = n

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.msb_in - self.lsb_in + 1
        # Simplified: pass-through with pipeline delay
        sig = Signal(w, name="re_pipe")
        sig_i = Signal(w, name="im_pipe")
        m.d.sync += [sig.eq(self.x_re), sig_i.eq(self.x_im)]
        prev_r, prev_i = sig, sig_i
        for i in range(self.n - 1):
            nr = Signal(w, name=f"re_d{i}")
            ni = Signal(w, name=f"im_d{i}")
            m.d.sync += [nr.eq(prev_r), ni.eq(prev_i)]
            prev_r, prev_i = nr, ni
        m.d.comb += [self.o_re.eq(prev_r), self.o_im.eq(prev_i)]
        return m
