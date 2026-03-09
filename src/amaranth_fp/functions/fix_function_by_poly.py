"""Piecewise polynomial function approximation."""
from __future__ import annotations

from math import ceil, log2

from amaranth import *

from ..pipelined import PipelinedComponent
from .table import Table
from .fix_horner import FixHornerEvaluator

__all__ = ["FixFunctionByPiecewisePoly"]


class FixFunctionByPiecewisePoly(PipelinedComponent):
    """Piecewise polynomial function approximation.

    Splits the input range into *num_segments* equal segments. The top bits
    of the input select the segment, and its coefficients are looked up in
    a ROM. The remaining bits are fed into a Horner evaluator.

    Parameters
    ----------
    input_width : int
    output_width : int
    num_segments : int
        Must be a power of two.
    degree : int
        Polynomial degree.
    coefficients : list[list[int]]
        coefficients[seg][coeff_index] in fixed-point. Length = num_segments,
        each inner list has degree+1 entries [c0, c1, ..., c_degree].
    coeff_width : int
        Bit width of each coefficient.
    """

    def __init__(
        self,
        input_width: int,
        output_width: int,
        num_segments: int,
        degree: int,
        coefficients: list[list[int]],
        coeff_width: int = 16,
    ) -> None:
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.num_segments = num_segments
        self.degree = degree
        self.coefficients = coefficients
        self.coeff_width = coeff_width

        self.x = Signal(input_width, name="x")
        self.result = Signal(output_width, name="result")
        # table lookup (1) + Horner stages (degree)  + output reg (1)
        self.latency = degree + 2

        seg_bits = int(ceil(log2(num_segments))) if num_segments > 1 else 1
        self._seg_bits = seg_bits
        self._local_bits = input_width - seg_bits

        # Build per-coefficient tables: for each coeff index, a table
        # mapping segment -> coefficient value
        self._coeff_tables: list[list[int]] = []
        for ci in range(degree + 1):
            tbl = []
            for seg in range(num_segments):
                tbl.append(coefficients[seg][ci] & ((1 << coeff_width) - 1))
            # Pad to power of 2
            while len(tbl) < (1 << seg_bits):
                tbl.append(0)
            self._coeff_tables.append(tbl)

    def elaborate(self, platform) -> Module:
        m = Module()
        seg_bits = self._seg_bits
        local_bits = self._local_bits
        cw = self.coeff_width
        degree = self.degree
        ow = self.output_width

        seg_addr = Signal(seg_bits, name="seg_addr")
        local_x = Signal(local_bits, name="local_x")
        m.d.comb += [
            seg_addr.eq(self.x[local_bits:]),
            local_x.eq(self.x[:local_bits]),
        ]

        # Coefficient lookup tables (latency 1 each)
        coeff_data = []
        for ci in range(degree + 1):
            tbl = Table(seg_bits, cw, self._coeff_tables[ci])
            setattr(m.submodules, f"coeff_tbl_{ci}", tbl)
            m.d.comb += tbl.addr.eq(seg_addr)
            coeff_data.append(tbl.data)

        # After table lookup (1 cycle), feed into Horner evaluator
        # We need to capture coefficients from table output and pass to Horner
        # Use a simple Horner with constant coefficients isn't possible here since
        # coefficients are dynamic. We implement a sequential Horner manually.

        # Pipeline: stage 0 = table lookup, stages 1..degree = Horner iterations
        # After table read (synchronous memory, 1 cycle), we have coefficients.

        # Delay local_x by 1 cycle to align with table output
        local_x_d = Signal(local_bits, name="local_x_d")
        m.d.sync += local_x_d.eq(local_x)

        # Horner iteration with dynamic coefficients
        acc_w = cw + local_bits
        # Start: acc = c[degree]
        acc = Signal(acc_w, name="horner_init")
        m.d.sync += acc.eq(coeff_data[degree])

        prev_acc = acc
        prev_x = local_x_d

        for stage in range(degree - 1, -1, -1):
            # Delay x for alignment
            if stage < degree - 1:
                next_x = Signal(local_bits, name=f"px_d{degree - 1 - stage}")
                m.d.sync += next_x.eq(prev_x)
                prev_x = next_x

            # Delay coeff_data[stage] to align: it comes out of table at cycle 1,
            # needs to arrive at the correct Horner stage
            c_delayed = coeff_data[stage]
            delays_needed = degree - stage - 1
            for d in range(delays_needed):
                c_next = Signal(cw, name=f"c{stage}_d{d}")
                m.d.sync += c_next.eq(c_delayed)
                c_delayed = c_next

            prod = Signal(2 * acc_w, name=f"pp_prod_{stage}")
            mac = Signal(acc_w, name=f"pp_mac_{stage}")
            m.d.comb += prod.eq(prev_acc * prev_x)
            m.d.comb += mac.eq(prod[:acc_w] + c_delayed)

            next_acc = Signal(acc_w, name=f"pp_acc_{stage}")
            m.d.sync += next_acc.eq(mac)
            prev_acc = next_acc

        # Output register
        out_r = Signal(ow, name="pp_out")
        m.d.sync += out_r.eq(prev_acc[:ow])
        m.d.comb += self.result.eq(out_r)

        return m
