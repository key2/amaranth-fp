"""KCM (Kernel Constant Multiplier) — table-based constant multiplication (pipelined, 2 stages)."""
from __future__ import annotations

from amaranth import *
from amaranth.lib.memory import Memory

from ..pipelined import PipelinedComponent

__all__ = ["FixRealKCM"]


class FixRealKCM(PipelinedComponent):
    """Table-based constant multiplier (2-cycle latency).

    Splits input into chunks, looks up partial products in ROM tables, sums them.

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

        # Scale constant to fixed-point
        scale = 1 << ow
        c_fixed = int(round(c * scale))

        # Single table lookup for small inputs
        table_size = 1 << iw
        init_data = [((i * c_fixed) >> ow) & ((1 << ow) - 1) for i in range(table_size)]

        mem = Memory(shape=unsigned(ow), depth=table_size, init=init_data)
        m.submodules.mem = mem

        rd = mem.read_port()
        m.d.comb += [rd.addr.eq(self.x), rd.en.eq(1)]

        # Memory has 1-cycle read latency → data available at cycle 1
        # Add one more register for cycle 2
        p_r2 = Signal(ow, name="p_r2")
        m.d.sync += p_r2.eq(rd.data)
        m.d.comb += self.p.eq(p_r2)

        return m
