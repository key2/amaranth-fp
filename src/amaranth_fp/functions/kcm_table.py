"""KCM (Kernel Constant Multiplier) table."""
from __future__ import annotations

from amaranth import *
from amaranth.lib.memory import Memory

from ..pipelined import PipelinedComponent

__all__ = ["KCMTable"]


class KCMTable(PipelinedComponent):
    """Lookup table for KCM constant multiplication.

    Parameters
    ----------
    input_width : int
    output_width : int
    constant : int
    """

    def __init__(self, input_width: int, output_width: int, constant: int):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.constant = constant
        contents = [(i * constant) & ((1 << output_width) - 1) for i in range(1 << input_width)]
        self.contents = contents
        self.addr = Signal(input_width, name="addr")
        self.data = Signal(output_width, name="data")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        mem = Memory(shape=unsigned(self.output_width),
                     depth=1 << self.input_width, init=self.contents)
        m.submodules.mem = mem
        rp = mem.read_port()
        m.d.comb += [rp.addr.eq(self.addr), rp.en.eq(1), self.data.eq(rp.data)]
        return m
