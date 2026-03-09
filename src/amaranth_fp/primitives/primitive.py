"""Base primitive component."""
from __future__ import annotations

from amaranth import *
from ..pipelined import PipelinedComponent

__all__ = ["Primitive"]


class Primitive(PipelinedComponent):
    """Base class for target-specific primitive components.

    Parameters
    ----------
    name : str
        Primitive name.
    """

    def __init__(self, name: str = "primitive"):
        super().__init__()
        self.prim_name = name
        self.latency = 0

    def elaborate(self, platform) -> Module:
        return Module()
