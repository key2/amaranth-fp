"""DAG-based operator composition."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["DAGOperator"]


class DAGOperator(PipelinedComponent):
    """Operator defined as a directed acyclic graph of sub-operators.

    Parameters
    ----------
    width : int
        Default signal width.
    """

    def __init__(self, width: int = 32):
        super().__init__()
        self.width = width
        self.nodes: list = []
        self.edges: list = []
        self.x = Signal(width, name="x")
        self.o = Signal(width, name="o")
        self.latency = 1

    def add_node(self, op):
        self.nodes.append(op)

    def add_edge(self, src, dst):
        self.edges.append((src, dst))

    def elaborate(self, platform) -> Module:
        m = Module()
        # Simple pass-through; real impl would wire sub-operators
        r = Signal(self.width, name="r")
        m.d.sync += r.eq(self.x)
        m.d.comb += self.o.eq(r)
        return m
