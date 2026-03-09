"""Pipelined Horner polynomial evaluator (fixed-point)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixHornerEvaluator"]


class FixHornerEvaluator(PipelinedComponent):
    """Evaluate a polynomial using Horner's method.

    Horner: result = c[n]; for i in range(n-1, -1, -1): result = result * x + c[i]
    Each multiply-add is one pipeline stage.

    Parameters
    ----------
    coefficients : list[int]
        Fixed-point polynomial coefficients [c0, c1, ..., cn].
    input_width : int
    coeff_width : int
    output_width : int
    """

    def __init__(
        self,
        coefficients: list[int],
        input_width: int,
        coeff_width: int,
        output_width: int,
    ) -> None:
        super().__init__()
        self.coefficients = coefficients
        self.input_width = input_width
        self.coeff_width = coeff_width
        self.output_width = output_width

        self.x = Signal(input_width, name="x")
        self.result = Signal(output_width, name="result")
        self.latency = max(len(coefficients) - 1, 1)

    def elaborate(self, platform) -> Module:
        m = Module()
        coeffs = self.coefficients
        n = len(coeffs)
        iw = self.input_width
        cw = self.coeff_width
        ow = self.output_width
        # Internal accumulator width: coeff_width + input_width (to hold products)
        acc_w = cw + iw

        if n == 0:
            m.d.comb += self.result.eq(0)
            return m

        if n == 1:
            # result = c[0], pipelined one cycle
            r = Signal(ow, name="horner_r0")
            m.d.sync += r.eq(coeffs[0] & ((1 << ow) - 1))
            m.d.comb += self.result.eq(r)
            return m

        # Horner iteration: start with c[n-1], then acc = acc * x + c[i]
        # Pipeline: each iteration is one sync stage
        acc = Signal(acc_w, name="horner_acc_init")
        m.d.sync += acc.eq(coeffs[n - 1] & ((1 << acc_w) - 1))

        x_d = Signal(iw, name="x_d0")
        m.d.sync += x_d.eq(self.x)

        prev_acc = acc
        prev_x = x_d

        for stage in range(n - 2, -1, -1):
            # multiply-add: new_acc = prev_acc * x + c[stage]
            prod = Signal(2 * acc_w, name=f"prod_{stage}")
            mac = Signal(acc_w, name=f"mac_{stage}")
            m.d.comb += prod.eq(prev_acc * prev_x)
            m.d.comb += mac.eq(prod[:acc_w] + (coeffs[stage] & ((1 << acc_w) - 1)))

            next_acc = Signal(acc_w, name=f"horner_acc_{stage}")
            m.d.sync += next_acc.eq(mac)

            if stage > 0:
                next_x = Signal(iw, name=f"x_d{n - 1 - stage}")
                m.d.sync += next_x.eq(prev_x)
                prev_x = next_x

            prev_acc = next_acc

        m.d.comb += self.result.eq(prev_acc[:ow])

        return m
