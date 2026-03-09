"""FP wrapper around FixRealKCM (pipelined)."""
from __future__ import annotations

from amaranth import *

from ..pipelined import PipelinedComponent
from ..format import FPFormat
from .fix_real_kcm import FixRealKCM

__all__ = ["FPRealKCM"]


class FPRealKCM(PipelinedComponent):
    """Floating-point constant multiplication via FixRealKCM.

    Parameters
    ----------
    fmt : FPFormat
    constant : float
    """

    def __init__(self, fmt: FPFormat, constant: float) -> None:
        super().__init__()
        self.fmt = fmt
        self.constant = constant
        self.a = Signal(fmt.width, name="a")
        self.o = Signal(fmt.width, name="o")
        self.latency = 4

    def elaborate(self, platform) -> Module:
        m = Module()
        fmt = self.fmt
        we = fmt.we
        wf = fmt.wf
        w = fmt.width

        # Unpack
        a_mant = self.a[:wf]
        a_exp = self.a[wf:wf + we]
        a_sign = self.a[wf + we]
        a_exc = self.a[wf + we + 1:]

        # KCM on mantissa: multiply (1.mantissa) by constant's mantissa
        import math
        const_abs = abs(self.constant)
        const_sign = 1 if self.constant < 0 else 0

        if const_abs == 0:
            m.d.comb += self.o.eq(0)
            return m

        const_exp = int(math.floor(math.log2(const_abs)))
        const_mant = int(round((const_abs / (2.0 ** const_exp)) * (1 << wf)))

        # Stage 0→1: multiply mantissa
        full_mant = Signal(wf + 1, name="full_mant")
        m.d.comb += full_mant.eq(Cat(a_mant, a_exc[0]))

        prod = Signal(2 * (wf + 1), name="kcm_prod")
        m.d.comb += prod.eq(full_mant * const_mant)

        prod_r = Signal(2 * (wf + 1), name="prod_r")
        exp_r = Signal(we, name="exp_r1")
        sign_r = Signal(name="sign_r1")
        exc_r = Signal(2, name="exc_r1")
        m.d.sync += [prod_r.eq(prod), exp_r.eq(a_exp), sign_r.eq(a_sign ^ const_sign), exc_r.eq(a_exc)]

        # Stage 1→2: normalize
        result_mant = Signal(wf, name="res_mant")
        result_exp = Signal(we, name="res_exp")
        norm_bit = prod_r[2 * wf + 1]
        with m.If(norm_bit):
            m.d.comb += [result_mant.eq(prod_r[wf + 1:2 * wf + 1]), result_exp.eq(exp_r + const_exp + 1)]
        with m.Else():
            m.d.comb += [result_mant.eq(prod_r[wf:2 * wf]), result_exp.eq(exp_r + const_exp)]

        mant_r2 = Signal(wf, name="mant_r2")
        exp_r2 = Signal(we, name="exp_r2")
        sign_r2 = Signal(name="sign_r2")
        exc_r2 = Signal(2, name="exc_r2")
        m.d.sync += [mant_r2.eq(result_mant), exp_r2.eq(result_exp), sign_r2.eq(sign_r), exc_r2.eq(exc_r)]

        # Stage 2→3: pass through
        mant_r3 = Signal(wf, name="mant_r3")
        exp_r3 = Signal(we, name="exp_r3")
        sign_r3 = Signal(name="sign_r3")
        exc_r3 = Signal(2, name="exc_r3")
        m.d.sync += [mant_r3.eq(mant_r2), exp_r3.eq(exp_r2), sign_r3.eq(sign_r2), exc_r3.eq(exc_r2)]

        # Stage 3→4: pack
        out_r = Signal(w, name="kcm_out")
        m.d.sync += out_r.eq(Cat(mant_r3, exp_r3, sign_r3, exc_r3))
        m.d.comb += self.o.eq(out_r)

        return m
