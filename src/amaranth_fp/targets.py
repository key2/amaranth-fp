"""FPGA target timing models for amaranth-fp.

Ported from FloPoCo's HWTargets C++ source files. Each target captures key
timing parameters used for pipeline-stage estimation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FPGATarget:
    """FPGA target timing and resource model.

    Attributes:
        name: Human-readable target name.
        lut_delay_ns: LUT propagation delay in nanoseconds.
        carry_delay_ns: Carry-chain propagation delay per step in ns.
        dsp_delay_ns: DSP multiplier delay in nanoseconds.
        reg_to_reg_delay_ns: Flip-flop (setup + clk-to-Q) delay in ns.
        dsp_width_a: First operand width of DSP multiplier.
        dsp_width_b: Second operand width of DSP multiplier.
        has_dsp: Whether the target has DSP blocks.
        max_frequency_mhz: Maximum clock frequency in MHz.
    """

    name: str
    lut_delay_ns: float
    carry_delay_ns: float
    dsp_delay_ns: float
    reg_to_reg_delay_ns: float
    dsp_width_a: int
    dsp_width_b: int
    has_dsp: bool
    max_frequency_mhz: float

    def cycles_for_delay(self, delay_ns: float) -> int:
        """Compute how many pipeline stages are needed for a combinational delay.

        Args:
            delay_ns: Total combinational path delay in nanoseconds.

        Returns:
            Number of pipeline register stages required.
        """
        period_ns = 1000.0 / self.max_frequency_mhz
        if delay_ns <= period_ns:
            return 0
        return math.ceil(delay_ns / period_ns) - 1


# ---------------------------------------------------------------------------
# Predefined targets — values ported from FloPoCo C++ target files
# ---------------------------------------------------------------------------

def Kintex7() -> FPGATarget:
    """Xilinx Kintex-7 (xc7k70tfbv484-3).

    Values from ``flopoco/code/HWTargets/src/Targets/Kintex7.cpp`` and
    the corresponding header.
    """
    return FPGATarget(
        name="Kintex7",
        lut_delay_ns=0.043,           # lut5Delay_ = 0.043e-9
        carry_delay_ns=0.049,         # carry4Delay_ = 0.049e-9
        dsp_delay_ns=2.892,           # DSPMultiplierDelay_ = 2.392e-9 + 0.5e-9
        reg_to_reg_delay_ns=0.216,    # ffDelay_ = 0.216e-9
        dsp_width_a=25,
        dsp_width_b=18,
        has_dsp=True,
        max_frequency_mhz=741.0,
    )


def VirtexUltrascalePlus() -> FPGATarget:
    """Xilinx Virtex UltraScale+ (xcvu3p-2-FFVC1517).

    Values from ``flopoco/code/HWTargets/src/Targets/VirtexUltrascalePlus.cpp``
    and the corresponding header.
    """
    return FPGATarget(
        name="VirtexUltrascalePlus",
        lut_delay_ns=0.035,           # lut5Delay_ = 0.035e-9
        carry_delay_ns=0.015,         # carry8Delay_ = 0.015e-9
        dsp_delay_ns=0.0,             # DSPMultiplierDelay_ not yet characterized
        reg_to_reg_delay_ns=0.150,    # ffDelay_ = 0.150e-9
        dsp_width_a=25,
        dsp_width_b=18,
        has_dsp=True,
        max_frequency_mhz=775.0,
    )


def Zynq7000() -> FPGATarget:
    """Xilinx Zynq-7000 (xc7z020clg484-1, Zedboard).

    Values from ``flopoco/code/HWTargets/src/Targets/Zynq7000.cpp``
    and the corresponding header.
    """
    return FPGATarget(
        name="Zynq7000",
        lut_delay_ns=0.124,           # lutDelay_ = 0.124e-9
        carry_delay_ns=0.114,         # carry4Delay_ = 0.114e-9
        dsp_delay_ns=0.0,             # DSPMultiplierDelay_ not yet characterized
        reg_to_reg_delay_ns=0.518,    # ffDelay_ = 0.518e-9
        dsp_width_a=25,
        dsp_width_b=18,
        has_dsp=True,
        max_frequency_mhz=500.0,
    )


def StratixV() -> FPGATarget:
    """Intel/Altera Stratix V.

    Values from ``flopoco/code/HWTargets/src/Targets/StratixV.cpp``
    and the corresponding header.
    """
    return FPGATarget(
        name="StratixV",
        lut_delay_ns=0.433,           # lutDelay_ = 0.433e-9
        carry_delay_ns=0.022,         # fastcarryDelay_ = 0.022e-9
        dsp_delay_ns=1.875,           # DSPMultiplierDelay_ = 1.875e-9
        reg_to_reg_delay_ns=0.156,    # ffDelay_ = 0.156e-9
        dsp_width_a=27,
        dsp_width_b=27,
        has_dsp=True,
        max_frequency_mhz=717.0,
    )


def Versal() -> FPGATarget:
    """Xilinx Versal.

    Values from ``flopoco/code/HWTargets/src/Targets/Versal.cpp``
    and the corresponding header. Note: timing data is currently copied
    from Zynq7000 in FloPoCo and has not been independently validated.
    """
    return FPGATarget(
        name="Versal",
        lut_delay_ns=0.124,           # lutDelay_ = 0.124e-9
        carry_delay_ns=0.114,         # carry4Delay_ = 0.114e-9
        dsp_delay_ns=0.0,             # DSPMultiplierDelay_ not yet characterized
        reg_to_reg_delay_ns=0.518,    # ffDelay_ = 0.518e-9
        dsp_width_a=25,
        dsp_width_b=18,
        has_dsp=True,
        max_frequency_mhz=500.0,
    )


def GenericTarget() -> FPGATarget:
    """Conservative generic FPGA target with safe default timings."""
    return FPGATarget(
        name="Generic",
        lut_delay_ns=0.5,
        carry_delay_ns=0.15,
        dsp_delay_ns=3.0,
        reg_to_reg_delay_ns=0.5,
        dsp_width_a=18,
        dsp_width_b=18,
        has_dsp=True,
        max_frequency_mhz=200.0,
    )
