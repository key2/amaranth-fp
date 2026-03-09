"""Generic multiplexer primitive (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["GenericMux"]


class GenericMux(PipelinedComponent):
    """Parameterised N-to-1 multiplexer.

    Parameters
    ----------
    width : int
        Data width of each input.
    n_inputs : int
        Number of inputs.
    """

    def __init__(self, width: int, n_inputs: int) -> None:
        super().__init__()
        self.width = width
        self.n_inputs = n_inputs
        self.inputs = [Signal(width, name=f"in{i}") for i in range(n_inputs)]
        sel_bits = (n_inputs - 1).bit_length() if n_inputs > 1 else 1
        self.sel = Signal(sel_bits, name="sel")
        self.o = Signal(width, name="o")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        selected = Signal(self.width, name="selected")
        with m.Switch(self.sel):
            for i, inp in enumerate(self.inputs):
                with m.Case(i):
                    m.d.comb += selected.eq(inp)
        # Register output for pipeline
        o_r = Signal(self.width, name="o_r")
        m.d.sync += o_r.eq(selected)
        m.d.comb += self.o.eq(o_r)
        return m
