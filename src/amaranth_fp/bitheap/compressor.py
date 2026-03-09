"""Generalized parallel counter / compressor (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["Compressor"]


class Compressor(PipelinedComponent):
    """Generalized parallel counter.

    Standard compressors: (3,2) full adder, (2,2) half adder, (4,2).

    Parameters
    ----------
    input_counts : list[int]
        Number of input bits per column weight.
    output_width : int
        Width of compressed output.
    """

    def __init__(self, input_counts: list[int], output_width: int) -> None:
        super().__init__()
        self.input_counts = input_counts
        self.output_width = output_width
        self.n_inputs = sum(input_counts)
        self.inputs = [Signal(name=f"cin_{i}") for i in range(self.n_inputs)]
        self.output = Signal(output_width, name="output")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        ow = self.output_width

        # Sum all weighted input bits
        total = Signal(ow, name="total")
        terms = []
        bit_idx = 0
        for weight, count in enumerate(self.input_counts):
            for _ in range(count):
                terms.append((weight, self.inputs[bit_idx]))
                bit_idx += 1

        # Build sum combinationally
        if terms:
            parts = []
            for weight, sig in terms:
                part = Signal(ow, name=f"part_{weight}_{sig.name}")
                m.d.comb += part.eq(sig << weight)
                parts.append(part)

            acc = Signal(ow, name="acc")
            m.d.comb += acc.eq(sum(parts))
            m.d.comb += total.eq(acc)

        out_r = Signal(ow, name="out_r")
        m.d.sync += out_r.eq(total)
        m.d.comb += self.output.eq(out_r)

        return m
