"""TestBench helper for simulation."""
from __future__ import annotations

from amaranth import *
from ..pipelined import PipelinedComponent

__all__ = ["TestBench"]


class TestBench(Elaboratable):
    """Wraps an operator with input/output registers for testing.

    Parameters
    ----------
    dut : PipelinedComponent
        The device under test.
    """

    def __init__(self, dut: PipelinedComponent):
        self.dut = dut

    def elaborate(self, platform) -> Module:
        m = Module()
        m.submodules.dut = self.dut
        return m
