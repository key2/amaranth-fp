"""Shift-add constant multiplier using CSD decomposition (pipelined, 2 stages)."""
from __future__ import annotations

import math

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixRealShiftAdd"]


def _to_csd(value: int, width: int) -> list[int]:
    """Convert integer to CSD (canonical signed digit) representation.

    Returns list of digits in {-1, 0, 1} from LSB to MSB.
    """
    digits = []
    carry = 0
    for i in range(width + 1):
        bit = ((value >> i) & 1) + carry
        carry = 0
        if bit >= 2:
            bit -= 2
            carry = 1
        if bit == 1 and i < width and ((value >> (i + 1)) & 1):
            digits.append(-1)
            carry = 1
        else:
            digits.append(bit)
    return digits


class FixRealShiftAdd(PipelinedComponent):
    """Shift-add constant multiplier (2-cycle latency).

    Decomposes constant into CSD representation, generates shift-add tree.

    Parameters
    ----------
    input_width : int
    constant : float
    output_width : int
    """

    def __init__(self, input_width: int, constant: float, output_width: int) -> None:
        super().__init__()
        self.input_width = input_width
        self.constant = constant
        self.output_width = output_width
        self.x = Signal(input_width, name="x")
        self.p = Signal(output_width, name="p")
        self.latency = 2

    def elaborate(self, platform) -> Module:
        m = Module()
        iw = self.input_width
        ow = self.output_width
        c = self.constant

        # Convert to fixed-point integer constant
        c_int = int(round(abs(c) * (1 << ow)))
        negate = c < 0

        # Get CSD digits
        csd = _to_csd(c_int, ow + iw)

        # Stage 0→1: compute shift-add tree combinationally
        acc_width = iw + ow + 2
        acc = Signal(acc_width, name="acc")

        terms = []
        for i, d in enumerate(csd):
            if d == 0:
                continue
            term = Signal(acc_width, name=f"term_{i}")
            if d == 1:
                m.d.comb += term.eq(self.x << i)
            else:  # d == -1
                m.d.comb += term.eq(-(self.x << i))
            terms.append(term)

        if terms:
            s = terms[0]
            for t in terms[1:]:
                s2 = Signal(acc_width, name=f"sum_{id(t)}")
                m.d.comb += s2.eq(s + t)
                s = s2
            if negate:
                m.d.comb += acc.eq(-s)
            else:
                m.d.comb += acc.eq(s)
        else:
            m.d.comb += acc.eq(0)

        # Pipeline stage 1
        acc_r1 = Signal(acc_width, name="acc_r1")
        m.d.sync += acc_r1.eq(acc)

        # Extract output bits and pipeline stage 2
        result = Signal(ow, name="result")
        m.d.comb += result.eq(acc_r1[ow:ow + ow] if ow + ow <= acc_width else acc_r1[:ow])

        p_r2 = Signal(ow, name="p_r2")
        m.d.sync += p_r2.eq(acc_r1[:ow])
        m.d.comb += self.p.eq(p_r2)

        return m
