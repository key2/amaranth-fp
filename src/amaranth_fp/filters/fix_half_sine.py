"""Half-sine window generator using precomputed ROM (pipelined, 1 stage)."""
from __future__ import annotations

import math

from amaranth import *
from amaranth.lib.memory import Memory

from ..pipelined import PipelinedComponent

__all__ = ["FixHalfSine"]


class FixHalfSine(PipelinedComponent):
    """Half-sine window generator (1-cycle latency).

    Precomputes sin(pi * i / n_samples) values in ROM.

    Parameters
    ----------
    width : int — output width (fixed-point)
    n_samples : int
    """

    def __init__(self, width: int, n_samples: int) -> None:
        super().__init__()
        self.width = width
        self.n_samples = n_samples
        self.addr = Signal(range(n_samples), name="addr")
        self.y = Signal(width, name="y")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        n = self.n_samples

        # Precompute half-sine values
        scale = (1 << w) - 1
        init = [int(round(math.sin(math.pi * i / n) * scale)) for i in range(n)]

        rom = Memory(shape=unsigned(w), depth=n, init=init)
        m.submodules.rom = rom
        rd = rom.read_port()
        m.d.comb += [rd.addr.eq(self.addr), rd.en.eq(1)]
        m.d.comb += self.y.eq(rd.data)

        return m
