"""Tests for FPGATarget."""

import math

from amaranth_fp.targets import (
    FPGATarget, Kintex7, VirtexUltrascalePlus, Zynq7000,
    StratixV, Versal, GenericTarget,
)


class TestPredefinedTargets:
    def test_kintex7_exists(self):
        t = Kintex7()
        assert t.name == "Kintex7"
        assert t.has_dsp is True
        assert t.max_frequency_mhz > 0

    def test_virtex_ultrascale_plus(self):
        t = VirtexUltrascalePlus()
        assert t.name == "VirtexUltrascalePlus"
        assert t.dsp_width_a == 25
        assert t.dsp_width_b == 18

    def test_zynq7000(self):
        t = Zynq7000()
        assert t.max_frequency_mhz == 500.0

    def test_stratixv(self):
        t = StratixV()
        assert t.dsp_width_a == 27
        assert t.dsp_width_b == 27

    def test_versal(self):
        t = Versal()
        assert t.name == "Versal"

    def test_generic(self):
        t = GenericTarget()
        assert t.name == "Generic"
        assert t.lut_delay_ns > 0


class TestCyclesForDelay:
    def test_zero_delay(self):
        t = GenericTarget()
        assert t.cycles_for_delay(0.0) == 0

    def test_within_one_period(self):
        t = GenericTarget()  # 200 MHz -> 5 ns period
        assert t.cycles_for_delay(4.0) == 0

    def test_exactly_one_period(self):
        t = GenericTarget()  # 5 ns period
        assert t.cycles_for_delay(5.0) == 0

    def test_needs_one_stage(self):
        t = GenericTarget()  # 5 ns period
        # 6 ns > 5 ns, ceil(6/5)-1 = 2-1 = 1
        assert t.cycles_for_delay(6.0) == 1

    def test_needs_two_stages(self):
        t = GenericTarget()  # 5 ns period
        # 12 ns: ceil(12/5)-1 = 3-1 = 2
        assert t.cycles_for_delay(12.0) == 2
