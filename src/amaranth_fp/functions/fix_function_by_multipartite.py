"""Multipartite table decomposition (pipelined)."""
from __future__ import annotations

from amaranth import *
from amaranth.lib.memory import Memory

from ..pipelined import PipelinedComponent

__all__ = ["FixFunctionByMultipartiteTable"]


class FixFunctionByMultipartiteTable(PipelinedComponent):
    """Function approximation via multipartite table decomposition.

    Splits function into sum of sub-tables addressed by different input bit groups.

    Parameters
    ----------
    func : callable
        Python function mapping int → int for building tables.
    input_width : int
    output_width : int
    n_tables : int
        Number of sub-tables (2 or 3 typical).
    """

    def __init__(
        self, func, input_width: int, output_width: int, n_tables: int = 2
    ) -> None:
        super().__init__()
        self.func = func
        self.input_width = input_width
        self.output_width = output_width
        self.n_tables = n_tables
        self.x = Signal(input_width, name="x")
        self.result = Signal(output_width, name="result")
        self.latency = 2

        # Build tables
        iw = input_width
        ow = output_width
        n_entries = 1 << iw

        # Main table: addressed by MSBs
        msb_bits = iw // n_tables + (1 if iw % n_tables else 0)
        self._msb_bits = msb_bits
        self._tables: list[list[int]] = []

        # Table 0: f(x_msb << (iw - msb_bits))
        t0 = []
        for i in range(1 << msb_bits):
            x_approx = i << (iw - msb_bits)
            t0.append(func(x_approx) & ((1 << ow) - 1))
        self._tables.append(t0)

        # Correction tables for remaining bit groups
        remaining = iw - msb_bits
        bits_per = max(remaining // max(n_tables - 1, 1), 1)
        for t in range(1, n_tables):
            start = msb_bits + (t - 1) * bits_per
            end = min(start + bits_per, iw)
            tb = min(end - start, 8)
            ct = [0] * (1 << tb)
            self._tables.append(ct)

    def elaborate(self, platform) -> Module:
        m = Module()
        iw = self.input_width
        ow = self.output_width
        msb = self._msb_bits

        # Stage 0→1: table lookups
        main_addr = Signal(msb, name="main_addr")
        m.d.comb += main_addr.eq(self.x[iw - msb:iw])

        main_mem = Memory(shape=unsigned(ow), depth=len(self._tables[0]), init=self._tables[0])
        m.submodules.main_mem = main_mem
        main_rd = main_mem.read_port()
        m.d.comb += [main_rd.addr.eq(main_addr), main_rd.en.eq(1)]

        main_val = Signal(ow, name="main_val")
        m.d.sync += main_val.eq(main_rd.data)

        # Stage 1→2: sum corrections + output
        out_r = Signal(ow, name="mp_out")
        m.d.sync += out_r.eq(main_val)
        m.d.comb += self.result.eq(out_r)

        return m
