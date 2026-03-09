"""Barrel shifter using log2 stages of muxes.

Based on FloPoCo's Shifters.cpp approach: each stage k conditionally shifts
by 2^k based on bit k of the shift amount, producing an efficient O(n log n)
barrel shifter.
"""

from amaranth import *
from math import ceil, log2

__all__ = ["Shifter"]


class Shifter(Elaboratable):
    """Barrel shifter with configurable direction and arithmetic mode.

    Parameters
    ----------
    width : int
        Bit width of the input and output data.
    shift_width : int
        Bit width of the shift amount signal.
    direction : str
        Shift direction: ``"left"`` or ``"right"``.
    arithmetic : bool
        If True and direction is ``"right"``, performs sign-extension
        (arithmetic right shift). Ignored for left shifts.

    Attributes
    ----------
    i : Signal(width), in
        Data input.
    shift : Signal(shift_width), in
        Shift amount.
    o : Signal(width), out
        Shifted output.
    """

    def __init__(self, width, shift_width, direction="left", arithmetic=False):
        if direction not in ("left", "right"):
            raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")

        self.width = width
        self.shift_width = shift_width
        self.direction = direction
        self.arithmetic = arithmetic

        self.i = Signal(width)
        self.shift = Signal(shift_width)
        self.o = Signal(width)

    def elaborate(self, platform):
        m = Module()

        width = self.width
        n_stages = self.shift_width

        # Chain of intermediate values through each stage
        stage = self.i

        for k in range(n_stages):
            shift_amount = 1 << k
            next_stage = Signal(width, name=f"stage_{k}")

            if self.direction == "left":
                # Left shift: fill with zeros from the right
                shifted = Signal(width, name=f"shifted_{k}")
                m.d.comb += shifted.eq(stage << shift_amount)
                m.d.comb += next_stage.eq(Mux(self.shift[k], shifted, stage))
            else:
                # Right shift: arithmetic (sign-extend) or logical (zero-fill)
                shifted = Signal(width, name=f"shifted_{k}")
                if self.arithmetic:
                    # Arithmetic: replicate the sign bit into vacated positions
                    sign_bit = stage[-1]
                    if shift_amount < width:
                        fill = Signal(shift_amount, name=f"fill_{k}")
                        m.d.comb += fill.eq(Mux(sign_bit, (1 << shift_amount) - 1, 0))
                        m.d.comb += shifted.eq(
                            Cat(stage[shift_amount:], fill)
                        )
                    else:
                        # Shift amount >= width: fill entirely with sign bit
                        m.d.comb += shifted.eq(Mux(sign_bit, (1 << width) - 1, 0))
                else:
                    # Logical: zero-fill
                    m.d.comb += shifted.eq(stage >> shift_amount)

                m.d.comb += next_stage.eq(Mux(self.shift[k], shifted, stage))

            stage = next_stage

        m.d.comb += self.o.eq(stage)

        return m
