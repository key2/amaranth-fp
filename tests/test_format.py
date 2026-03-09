"""Tests for FPFormat and layout helpers."""

import pytest
from amaranth.lib.data import StructLayout

from amaranth_fp.format import FPFormat, ieee_layout, internal_layout


class TestFPFormatPredefined:
    def test_half(self):
        fmt = FPFormat.half()
        assert fmt.we == 5
        assert fmt.wf == 10

    def test_single(self):
        fmt = FPFormat.single()
        assert fmt.we == 8
        assert fmt.wf == 23

    def test_double(self):
        fmt = FPFormat.double()
        assert fmt.we == 11
        assert fmt.wf == 52


class TestFPFormatProperties:
    def test_width_half(self):
        fmt = FPFormat.half()
        # 2 (exc) + 1 (sign) + 5 (exp) + 10 (mant) = 18
        assert fmt.width == 18

    def test_ieee_width_half(self):
        fmt = FPFormat.half()
        # 1 (sign) + 5 (exp) + 10 (mant) = 16
        assert fmt.ieee_width == 16

    def test_bias_half(self):
        fmt = FPFormat.half()
        assert fmt.bias == 15  # 2^(5-1) - 1

    def test_bias_single(self):
        fmt = FPFormat.single()
        assert fmt.bias == 127

    def test_bias_double(self):
        fmt = FPFormat.double()
        assert fmt.bias == 1023

    def test_emin(self):
        fmt = FPFormat.half()
        assert fmt.emin == -14  # 1 - 15

    def test_emax(self):
        fmt = FPFormat.half()
        assert fmt.emax == 15  # bias


class TestFPFormatValidation:
    def test_we_too_small(self):
        with pytest.raises(ValueError):
            FPFormat(we=1, wf=10)

    def test_wf_too_small(self):
        with pytest.raises(ValueError):
            FPFormat(we=5, wf=0)


class TestLayouts:
    def test_ieee_layout_is_struct(self):
        fmt = FPFormat.half()
        layout = ieee_layout(fmt)
        assert isinstance(layout, StructLayout)

    def test_ieee_layout_size(self):
        fmt = FPFormat.half()
        layout = ieee_layout(fmt)
        assert layout.size == 16

    def test_internal_layout_is_struct(self):
        fmt = FPFormat.half()
        layout = internal_layout(fmt)
        assert isinstance(layout, StructLayout)

    def test_internal_layout_size(self):
        fmt = FPFormat.half()
        layout = internal_layout(fmt)
        assert layout.size == 18
