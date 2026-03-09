"""Normalizer: counts leading zeros then left-shifts to normalize.

Based on FloPoCo's Normalizer.cpp: combines an LZC with a left shifter
to produce a normalized output where the leading 1 is in the MSB position.
"""

from amaranth import *
from math import ceil, log2

from .lzc import LeadingZeroCounter
from .shifter import Shifter

__all__ = ["Normalizer"]


class Normalizer(Elaboratable):
    """Normalizes an input by counting leading zeros and left-shifting.

    The output is the input shifted left so that the most significant '1'
    is at the MSB of the output, with the shift count provided as an output.

    Parameters
    ----------
    input_width : int
        Bit width of the input.
    output_width : int
        Bit width of the normalized output.
    count_width : int or None
        Bit width of the count output. If None, computed as
        ``ceil(log2(input_width + 1))``.

    Attributes
    ----------
    i : Signal(input_width), in
        Input value to normalize.
    count : Signal(count_width), out
        Number of leading zeros detected.
    o : Signal(output_width), out
        Normalized (left-shifted) result.
    """

    def __init__(self, input_width, output_width, count_width=None):
        self.input_width = input_width
        self.output_width = output_width
        self.count_width = count_width or max(1, ceil(log2(input_width + 1)))

        self.i = Signal(input_width)
        self.count = Signal(self.count_width)
        self.o = Signal(output_width)

    def elaborate(self, platform):
        m = Module()

        # Instantiate LZC
        lzc = LeadingZeroCounter(self.input_width)
        m.submodules.lzc = lzc
        m.d.comb += lzc.i.eq(self.i)
        m.d.comb += self.count.eq(lzc.count)

        # Instantiate left shifter
        # Work at max(input_width, output_width) to avoid truncation issues
        work_width = max(self.input_width, self.output_width)
        shifter = Shifter(
            width=work_width,
            shift_width=self.count_width,
            direction="left",
        )
        m.submodules.shifter = shifter

        # Zero-extend input to working width, shift, then take output_width MSBs
        m.d.comb += shifter.i.eq(self.i)
        m.d.comb += shifter.shift.eq(lzc.count)

        # Take the most significant output_width bits from the shifted result
        if self.output_width <= work_width:
            m.d.comb += self.o.eq(shifter.o[:self.output_width])
        else:
            m.d.comb += self.o.eq(shifter.o)

        return m
