"""IEEE 754 round-to-nearest-even (RNE) rounding unit.

Extracts guard, round, and sticky bits from the extended mantissa and
applies the RNE rounding rule: round up when G=1 and (R|S|LSB)=1.
"""

from amaranth import *

__all__ = ["RoundingUnit"]


class RoundingUnit(Elaboratable):
    """Rounds an extended mantissa using IEEE 754 round-to-nearest-even.

    The input mantissa has ``width + 3`` bits laid out as::

        [mantissa (width bits)] [guard] [round] [sticky]
         MSB ...                 bit 2   bit 1   bit 0

    Parameters
    ----------
    width : int
        Target mantissa width (after rounding).

    Attributes
    ----------
    mantissa_in : Signal(width + 3), in
        Extended mantissa: ``width`` mantissa bits + guard + round + sticky.
    mantissa_out : Signal(width), out
        Rounded mantissa.
    overflow : Signal(1), out
        Set if rounding caused a carry out of the mantissa MSB.
    """

    def __init__(self, width):
        self.width = width

        self.mantissa_in = Signal(width + 3)
        self.mantissa_out = Signal(width)
        self.overflow = Signal()

    def elaborate(self, platform):
        m = Module()

        width = self.width

        # Extract fields: [mantissa_bits(width) | G | R | S]
        sticky = self.mantissa_in[0]
        round_bit = self.mantissa_in[1]
        guard = self.mantissa_in[2]
        mantissa = self.mantissa_in[3:]  # upper width bits

        # LSB of the unrounded mantissa
        lsb = mantissa[0]

        # RNE: round up = G & (R | S | LSB)
        round_up = guard & (round_bit | sticky | lsb)

        # Add rounding increment
        rounded = Signal(width + 1, name="rounded")
        m.d.comb += rounded.eq(mantissa + round_up)

        m.d.comb += [
            self.mantissa_out.eq(rounded[:width]),
            self.overflow.eq(rounded[width]),
        ]

        return m
