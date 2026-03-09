"""amaranth-fp — Floating-point operator generator for Amaranth HDL.

Inspired by FloPoCo, this library provides parameterized floating-point
operators targeting FPGA synthesis via the Amaranth HDL framework.
"""

from .format import FPFormat, ieee_layout, internal_layout
from .operator import FPOperator
from .pipeline import PipelineHelper
from .pipelined import PipelinedComponent
from .targets import FPGATarget

__all__ = [
    "FPFormat",
    "FPGATarget",
    "FPOperator",
    "PipelineHelper",
    "PipelinedComponent",
    "ieee_layout",
    "internal_layout",
]
