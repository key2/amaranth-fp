"""N-point FFT using butterfly stages (pipelined)."""
from __future__ import annotations

import math

from amaranth import *

from ..pipelined import PipelinedComponent

__all__ = ["FixFFT"]


class FixFFT(PipelinedComponent):
    """N-point radix-2 DIT FFT with precomputed twiddle factors.

    Parameters
    ----------
    width : int
        Bit width of each real/imaginary component.
    n_points : int
        Number of FFT points (must be power of 2).
    """

    def __init__(self, width: int, n_points: int) -> None:
        super().__init__()
        if n_points < 2 or (n_points & (n_points - 1)) != 0:
            raise ValueError("n_points must be a power of 2")
        self.width = width
        self.n_points = n_points
        self.n_stages = int(math.log2(n_points))

        # Input/output arrays as flat signals
        self.inputs_re = [Signal(signed(width), name=f"in_re_{i}") for i in range(n_points)]
        self.inputs_im = [Signal(signed(width), name=f"in_im_{i}") for i in range(n_points)]
        self.outputs_re = [Signal(signed(width), name=f"out_re_{i}") for i in range(n_points)]
        self.outputs_im = [Signal(signed(width), name=f"out_im_{i}") for i in range(n_points)]

        butterfly_latency = 4
        self.latency = self.n_stages * butterfly_latency

    def elaborate(self, platform) -> Module:
        m = Module()
        w = self.width
        N = self.n_points
        n_stages = self.n_stages

        # Bit-reversal permutation of inputs
        def bit_reverse(x, bits):
            result = 0
            for i in range(bits):
                result = (result << 1) | ((x >> i) & 1)
            return result

        # Current stage signals - start with bit-reversed inputs
        cur_re = [None] * N
        cur_im = [None] * N
        for i in range(N):
            br = bit_reverse(i, n_stages)
            cur_re[i] = self.inputs_re[br]
            cur_im[i] = self.inputs_im[br]

        # Precompute twiddle factors as fixed-point
        frac_bits = w - 2  # leave room for sign + integer
        scale = (1 << frac_bits)

        for stage in range(n_stages):
            half_size = 1 << stage
            group_size = 2 * half_size
            next_re = [None] * N
            next_im = [None] * N

            for group_start in range(0, N, group_size):
                for j in range(half_size):
                    idx_top = group_start + j
                    idx_bot = group_start + j + half_size

                    # Twiddle factor W_N^(j * N / group_size)
                    k = j * (N // group_size)
                    angle = -2.0 * math.pi * k / N
                    tw_re_val = int(round(math.cos(angle) * scale))
                    tw_im_val = int(round(math.sin(angle) * scale))

                    # Butterfly: top' = top + tw*bot, bot' = top - tw*bot
                    top_re = cur_re[idx_top]
                    top_im = cur_im[idx_top]
                    bot_re = cur_re[idx_bot]
                    bot_im = cur_im[idx_bot]

                    # tw * bot (complex multiply)
                    tw_bot_re = Signal(signed(2 * w), name=f"twbr_s{stage}_g{group_start}_j{j}")
                    tw_bot_im = Signal(signed(2 * w), name=f"twbi_s{stage}_g{group_start}_j{j}")

                    prod_ac = Signal(signed(2 * w), name=f"pac_s{stage}_{idx_top}")
                    prod_bd = Signal(signed(2 * w), name=f"pbd_s{stage}_{idx_top}")
                    prod_ad = Signal(signed(2 * w), name=f"pad_s{stage}_{idx_top}")
                    prod_bc = Signal(signed(2 * w), name=f"pbc_s{stage}_{idx_top}")

                    m.d.comb += [
                        prod_ac.eq(bot_re * tw_re_val),
                        prod_bd.eq(bot_im * tw_im_val),
                        prod_ad.eq(bot_re * tw_im_val),
                        prod_bc.eq(bot_im * tw_re_val),
                        tw_bot_re.eq((prod_ac - prod_bd) >> frac_bits),
                        tw_bot_im.eq((prod_ad + prod_bc) >> frac_bits),
                    ]

                    # Pipeline register for this stage
                    new_top_re = Signal(signed(w), name=f"fft_re_s{stage+1}_{idx_top}")
                    new_top_im = Signal(signed(w), name=f"fft_im_s{stage+1}_{idx_top}")
                    new_bot_re = Signal(signed(w), name=f"fft_re_s{stage+1}_{idx_bot}")
                    new_bot_im = Signal(signed(w), name=f"fft_im_s{stage+1}_{idx_bot}")

                    m.d.sync += [
                        new_top_re.eq((top_re + tw_bot_re)[:w]),
                        new_top_im.eq((top_im + tw_bot_im)[:w]),
                        new_bot_re.eq((top_re - tw_bot_re)[:w]),
                        new_bot_im.eq((top_im - tw_bot_im)[:w]),
                    ]

                    next_re[idx_top] = new_top_re
                    next_im[idx_top] = new_top_im
                    next_re[idx_bot] = new_bot_re
                    next_im[idx_bot] = new_bot_im

            cur_re = next_re
            cur_im = next_im

        # Connect final stage to outputs
        for i in range(N):
            m.d.comb += [
                self.outputs_re[i].eq(cur_re[i]),
                self.outputs_im[i].eq(cur_im[i]),
            ]

        return m
