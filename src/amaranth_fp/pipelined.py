"""PipelinedComponent base class for registered pipeline stages."""
from __future__ import annotations

from amaranth.hdl import *

__all__ = ["PipelinedComponent"]


class PipelinedComponent(Elaboratable):
    """Base class for pipelined Amaranth components.

    Provides helpers to register signals across pipeline stages and
    align signals from different computation paths to the same stage.
    """

    def __init__(self):
        self.latency = 0
        self._sig_latency: dict[str, int] = {}

    def sync_to(self, m: Module, sig: Signal, target_cycle: int) -> Signal:
        """Delay *sig* with sync registers until it reaches *target_cycle*."""
        current = self._sig_latency.get(sig.name, 0)
        for i in range(target_cycle - current):
            delayed = Signal(sig.shape(), name=f"{sig.name}_d{current + i + 1}")
            m.d.sync += delayed.eq(sig)
            sig = delayed
        self._sig_latency[sig.name] = target_cycle
        return sig

    def add_latency(self, sig: Signal, cycles: int):
        """Record that *sig* is available at pipeline stage *cycles*."""
        self._sig_latency[sig.name] = cycles

    def get_latency(self, sig: Signal) -> int:
        """Return the pipeline stage at which *sig* is available."""
        return self._sig_latency.get(sig.name, 0)
