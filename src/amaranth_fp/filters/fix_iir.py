"""IIR filter — direct form I (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixIIR"]


class FixIIR(PipelinedComponent):
    """Fixed-point IIR filter (direct form I).

    y[n] = sum(b[i]*x[n-i]) - sum(a[j]*y[n-j])

    Parameters
    ----------
    input_width : int
    output_width : int
    b_coeffs : list[int]
    a_coeffs : list[int] — a[0] is assumed to be 1 (not stored)
    coeff_width : int
    """

    def __init__(self, input_width: int, output_width: int,
                 b_coeffs: list[int], a_coeffs: list[int], coeff_width: int) -> None:
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.b_coeffs = list(b_coeffs)
        self.a_coeffs = list(a_coeffs)
        self.coeff_width = coeff_width
        self.x = Signal(input_width, name="x")
        self.y = Signal(output_width, name="y")
        self.latency = max(len(b_coeffs), len(a_coeffs))

    def elaborate(self, platform) -> Module:
        m = Module()
        iw = self.input_width
        ow = self.output_width
        cw = self.coeff_width
        nb = len(self.b_coeffs)
        na = len(self.a_coeffs)
        acc_w = max(iw, ow) + cw + max(nb, na) + 1

        # Delay lines for x and y
        x_delay = [Signal(iw, name=f"x_d{i}") for i in range(nb)]
        y_delay = [Signal(ow, name=f"y_d{i}") for i in range(na)]

        # Shift x delay line
        if nb > 0:
            m.d.sync += x_delay[0].eq(self.x)
            for i in range(1, nb):
                m.d.sync += x_delay[i].eq(x_delay[i - 1])

        # FIR part: sum b[i]*x[n-i]
        fir_sum = Signal(acc_w, name="fir_sum")
        fir_terms = []
        for i, b in enumerate(self.b_coeffs):
            if i == 0:
                t = Signal(acc_w, name=f"bt_{i}")
                m.d.comb += t.eq(self.x * b)
                fir_terms.append(t)
            else:
                t = Signal(acc_w, name=f"bt_{i}")
                m.d.comb += t.eq(x_delay[i - 1] * b)
                fir_terms.append(t)

        if fir_terms:
            s = fir_terms[0]
            for t in fir_terms[1:]:
                s2 = Signal(acc_w, name=f"fs_{id(t)}")
                m.d.comb += s2.eq(s + t)
                s = s2
            m.d.comb += fir_sum.eq(s)

        # IIR part: sum a[j]*y[n-j]
        iir_sum = Signal(acc_w, name="iir_sum")
        iir_terms = []
        for j, a in enumerate(self.a_coeffs):
            t = Signal(acc_w, name=f"at_{j}")
            m.d.comb += t.eq(y_delay[j] * a) if j < len(y_delay) else None
            iir_terms.append(t)

        if iir_terms:
            s = iir_terms[0]
            for t in iir_terms[1:]:
                s2 = Signal(acc_w, name=f"is_{id(t)}")
                m.d.comb += s2.eq(s + t)
                s = s2
            m.d.comb += iir_sum.eq(s)

        # Output = fir_sum - iir_sum
        y_new = Signal(acc_w, name="y_new")
        m.d.comb += y_new.eq(fir_sum - iir_sum)

        # Shift y delay line
        if na > 0:
            m.d.sync += y_delay[0].eq(y_new[:ow])
            for i in range(1, na):
                m.d.sync += y_delay[i].eq(y_delay[i - 1])

        m.d.comb += self.y.eq(y_new[:ow])

        return m
