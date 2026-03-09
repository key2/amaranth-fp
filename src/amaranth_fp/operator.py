"""Base FP operator class for amaranth-fp.

Provides ``FPOperator``, the abstract base for all floating-point operator
components.  Extends Amaranth's ``Component`` with FP format metadata and
pipeline-register insertion helpers.
"""
from __future__ import annotations

from amaranth import *
from amaranth.lib.wiring import Component, In, Out

from .format import FPFormat


class FPOperator(Component):
    """Abstract base class for floating-point operator components.

    Subclasses should define their port signatures (via class-level annotations
    or by passing a dict to ``super().__init__``) and implement
    :meth:`elaborate`.

    Args:
        fmt: The floating-point format for this operator.
        pipeline_stages: Number of pipeline register stages to insert
            (0 means purely combinational).
    """

    def __init__(
        self,
        fmt: FPFormat,
        pipeline_stages: int = 0,
        signature_members: dict | None = None,
    ) -> None:
        self._fmt = fmt
        self._pipeline_stages = pipeline_stages
        if signature_members is not None:
            super().__init__(signature_members)
        else:
            super().__init__({})

    @property
    def fmt(self) -> FPFormat:
        """The floating-point format used by this operator."""
        return self._fmt

    @property
    def pipeline_stages(self) -> int:
        """Number of pipeline stages configured for this operator."""
        return self._pipeline_stages

    def pipeline_register(self, m: Module, signal: Signal, stage: int) -> Signal:
        """Conditionally insert a pipeline register for *signal* at *stage*.

        If ``stage < pipeline_stages``, a new registered copy of *signal* is
        created in the ``sync`` domain and returned.  Otherwise *signal* is
        returned unchanged (combinational pass-through).

        Args:
            m: The Amaranth ``Module`` being elaborated.
            signal: The signal to potentially register.
            stage: The current pipeline stage index (0-based).

        Returns:
            Either the original signal or a registered copy.
        """
        if stage < self._pipeline_stages:
            reg = Signal(signal.shape(), name=f"{signal.name}_s{stage}")
            m.d.sync += reg.eq(signal)
            return reg
        return signal
