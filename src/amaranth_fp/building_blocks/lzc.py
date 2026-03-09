"""Leading Zero Counter (LZC) using a recursive tree-based approach.

Based on FloPoCo's LZOC.cpp: recursively splits the input in half,
counts leading zeros in the upper half, and if the upper half is all
zeros, adds the count from the lower half. Base case handles 2-bit inputs.
"""

from amaranth import *
from math import ceil, log2

__all__ = ["LeadingZeroCounter"]


class LeadingZeroCounter(Elaboratable):
    """Counts leading zeros using a recursive divide-and-conquer tree.

    Parameters
    ----------
    width : int
        Bit width of the input. Must be >= 1.

    Attributes
    ----------
    i : Signal(width), in
        Input value.
    count : Signal(ceil(log2(width))), out
        Number of leading zeros (from MSB).
    all_zeros : Signal(1), out
        High when the entire input is zero.
    """

    def __init__(self, width):
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")

        self.width = width
        self.count_width = max(1, ceil(log2(width + 1)))

        self.i = Signal(width)
        self.count = Signal(self.count_width)
        self.all_zeros = Signal()

    def elaborate(self, platform):
        m = Module()

        count, az = self._build_tree(m, self.i, self.width, prefix="lzc")
        m.d.comb += [
            self.count.eq(count),
            self.all_zeros.eq(az),
        ]

        return m

    def _build_tree(self, m, data, width, prefix):
        """Recursively build the LZC tree.

        Returns (count_signal, all_zeros_signal) for the given data slice.
        """
        if width == 1:
            # Base case: 1 bit — count is 1 if bit is 0, all_zeros if bit is 0
            az = Signal(name=f"{prefix}_az")
            cnt = Signal(1, name=f"{prefix}_cnt")
            m.d.comb += [
                az.eq(~data[0]),
                cnt.eq(~data[0]),
            ]
            return cnt, az

        if width == 2:
            # Base case: 2 bits
            az = Signal(name=f"{prefix}_az")
            cnt = Signal(1, name=f"{prefix}_cnt")
            m.d.comb += [
                az.eq(~data[0] & ~data[1]),
                # Leading zeros from MSB: if MSB=0 and LSB=0 => 2 (handled by az),
                # if MSB=0, LSB=1 => 1; if MSB=1 => 0
                cnt.eq(~data[1]),
            ]
            return cnt, az

        # Recursive case: split in half
        half = width >> 1
        upper_width = width - half  # upper half (MSB side)

        upper = data[half:]  # MSB half
        lower = data[:half]  # LSB half

        hi_cnt, hi_az = self._build_tree(m, upper, upper_width, prefix=f"{prefix}_hi")
        lo_cnt, lo_az = self._build_tree(m, lower, half, prefix=f"{prefix}_lo")

        count_width = max(1, ceil(log2(width + 1)))
        cnt = Signal(count_width, name=f"{prefix}_cnt")
        az = Signal(name=f"{prefix}_az")

        m.d.comb += az.eq(hi_az & lo_az)

        # If upper half is all zeros, total count = upper_width + lo_cnt
        # Otherwise, total count = hi_cnt
        with m.If(hi_az):
            m.d.comb += cnt.eq(upper_width + lo_cnt)
        with m.Else():
            m.d.comb += cnt.eq(hi_cnt)

        return cnt, az
