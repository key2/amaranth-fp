"""Boolean equation evaluator primitive (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["BooleanEquation"]


class BooleanEquation(PipelinedComponent):
    """Evaluate a truth-table-defined Boolean function.

    Parameters
    ----------
    n_inputs : int
        Number of single-bit inputs.
    truth_table : int
        Truth table as an integer (bit *i* = output for input pattern *i*).
    """

    def __init__(self, n_inputs: int, truth_table: int) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.truth_table = truth_table
        self.inputs = Signal(n_inputs, name="inputs")
        self.output = Signal(name="output")
        self.latency = 1

    def elaborate(self, platform) -> Module:
        m = Module()
        tt = self.truth_table
        result = Signal(name="result")
        with m.Switch(self.inputs):
            for i in range(1 << self.n_inputs):
                with m.Case(i):
                    m.d.comb += result.eq((tt >> i) & 1)
        o_r = Signal(name="o_r")
        m.d.sync += o_r.eq(result)
        m.d.comb += self.output.eq(o_r)
        return m
