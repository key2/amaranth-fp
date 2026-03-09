"""Single sin or cos output (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent
from .fix_sincos import FixSinCos

__all__ = ["FixSinOrCos"]


class FixSinOrCos(PipelinedComponent):
    """Compute either sin or cos (saves area vs both).

    Parameters
    ----------
    width : int
    compute_sin : bool
        If True compute sin, else cos.
    """

    def __init__(self, width: int, compute_sin: bool = True) -> None:
        super().__init__()
        self.width = width
        self.compute_sin = compute_sin
        self._sincos = FixSinCos(width)
        self.angle = Signal(width, name="angle")
        self.output = Signal(width, name="output")
        self.latency = self._sincos.latency

    def elaborate(self, platform) -> Module:
        m = Module()
        m.submodules.sincos = sc = self._sincos
        m.d.comb += sc.angle.eq(self.angle)
        if self.compute_sin:
            m.d.comb += self.output.eq(sc.sin_out)
        else:
            m.d.comb += self.output.eq(sc.cos_out)
        return m
