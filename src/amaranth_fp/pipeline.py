"""Pipeline utilities for amaranth-fp.

Provides ``PipelineHelper`` for managing multi-stage pipeline register
insertion across a set of signals.
"""
from __future__ import annotations

from amaranth import *


class PipelineHelper:
    """Helper for inserting pipeline registers across multiple stages.

    Usage::

        def elaborate(self, platform):
            m = Module()
            pipe = PipelineHelper(m, stages=3)

            a = Signal(8, name="a")
            b = Signal(8, name="b")
            # ... combinational logic producing a, b ...

            a, b = pipe.stage(a, b)       # stage 0 -> 1
            # ... more combinational logic ...
            a, b = pipe.stage(a, b)       # stage 1 -> 2
            # ... more combinational logic ...
            a, b = pipe.stage(a, b)       # stage 2 -> 3
            return m

    Args:
        m: The Amaranth ``Module`` being elaborated.
        stages: Total number of pipeline register stages.
    """

    def __init__(self, m: Module, stages: int) -> None:
        self._m = m
        self._stages = stages
        self._current: int = 0

    @property
    def current_stage(self) -> int:
        """The current pipeline stage index (0-based)."""
        return self._current

    @property
    def stages(self) -> int:
        """Total number of pipeline stages."""
        return self._stages

    def stage(self, *signals: Signal) -> tuple[Signal, ...]:
        """Register signals for the current stage and advance the counter.

        If ``current_stage < stages``, each signal is registered (sync domain).
        Otherwise signals pass through unchanged.

        Returns:
            Tuple of (possibly registered) signals in the same order.
        """
        result: list[Signal] = []
        for sig in signals:
            if self._current < self._stages:
                reg = Signal(sig.shape(), name=f"{sig.name}_s{self._current}")
                self._m.d.sync += reg.eq(sig)
                result.append(reg)
            else:
                result.append(sig)
        self._current += 1
        return tuple(result)

    def delay(self, signal: Signal, n: int = 1) -> Signal:
        """Delay a signal by *n* clock cycles via a chain of registers.

        Args:
            signal: The signal to delay.
            n: Number of cycles to delay (each inserts one sync register).

        Returns:
            The delayed signal.
        """
        s = signal
        for i in range(n):
            reg = Signal(signal.shape(), name=f"{signal.name}_d{i}")
            self._m.d.sync += reg.eq(s)
            s = reg
        return s
