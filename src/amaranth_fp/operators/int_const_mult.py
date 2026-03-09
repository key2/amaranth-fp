"""Integer constant multiplication via shift-add (pipelined, 1 stage).

Decomposes constant into CSD (canonical signed digit) representation
for efficient shift-add implementation.
"""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["IntConstMult"]


def _to_csd(value: int) -> list[tuple[int, int]]:
    """Convert integer to CSD representation: list of (bit_position, sign).

    sign is +1 or -1.
    """
    if value == 0:
        return []

    terms = []
    n = abs(value)
    sign_mult = 1 if value > 0 else -1
    pos = 0
    while n > 0:
        if n & 1:
            if n & 2:
                # Next bit also set → use -1 here and carry
                terms.append((pos, -1 * sign_mult))
                n += 1
            else:
                terms.append((pos, 1 * sign_mult))
            n >>= 1
        else:
            n >>= 1
        pos += 1
    return terms


class IntConstMult(PipelinedComponent):
    """Integer constant multiplication via shift-add network.

    Parameters
    ----------
    width : int
        Input width.
    constant : int
        Compile-time integer constant.

    Attributes
    ----------
    x : Signal(width), in
    result : Signal(width + constant.bit_length()), out
    """

    def __init__(self, width: int, constant: int) -> None:
        super().__init__()
        self.width = width
        self.constant = constant
        out_w = width + max(constant.bit_length(), 1) + 1
        self.x = Signal(width, name="x")
        self.result = Signal(out_w, name="result")
        self.latency = 1
        self._csd = _to_csd(constant)
        self._out_w = out_w

    def elaborate(self, platform) -> Module:
        m = Module()
        ow = self._out_w

        if not self._csd:
            # constant == 0
            r = Signal(ow, name="icm_r")
            m.d.sync += r.eq(0)
            m.d.comb += self.result.eq(r)
            return m

        # Compute sum of shifted x values using CSD
        acc = Signal(signed(ow + 1), name="icm_acc")
        total = None
        for pos, sign in self._csd:
            term = Signal(signed(ow + 1), name=f"icm_t{pos}")
            if sign > 0:
                m.d.comb += term.eq(self.x << pos)
            else:
                m.d.comb += term.eq(-(self.x << pos))
            if total is None:
                total = term
            else:
                new_total = Signal(signed(ow + 1), name=f"icm_s{pos}")
                m.d.comb += new_total.eq(total + term)
                total = new_total

        m.d.comb += acc.eq(total)

        r = Signal(ow, name="icm_r")
        m.d.sync += r.eq(acc[:ow])
        m.d.comb += self.result.eq(r)

        return m
