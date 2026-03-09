#!/usr/bin/env python3
"""Generate a sinusoidal waveform using amaranth-fp's FixSinCos CORDIC operator.

Outputs:
  - examples/sinusoide.vcd  — VCD waveform for GTKWave
  - examples/sinusoide.gtkw — GTKWave save file
  - examples/sinusoide.png  — Matplotlib plot

The CORDIC rotation mode converges only for |angle| <= ~π/2, so we perform
quadrant reduction: map every input angle into Q1 [0, π/2), compute sin/cos
there, then reconstruct the correct sign for the original quadrant.
"""
import math
import matplotlib.pyplot as plt
from amaranth import *
from amaranth.sim import Simulator
from amaranth_fp.operators.fix_sincos import FixSinCos

WIDTH = 16  # 16-bit fixed-point
N_SAMPLES = 256  # number of angle samples over [0, 2π)
QUARTER = (1 << WIDTH) // 4  # 2^WIDTH / 4  =>  π/2 in fixed-point units

dut = FixSinCos(width=WIDTH, iterations=WIDTH)

# Collect results
sin_values = []
cos_values = []
quadrants = []  # store quadrant per sample so we can fix signs after

sim = Simulator(dut)
sim.add_clock(1e-6)


async def testbench(ctx):
    latency = dut.latency

    for i in range(N_SAMPLES + latency + 2):
        if i < N_SAMPLES:
            # Full-range angle: i/N * full_circle
            angle_full = int(i / N_SAMPLES * (1 << WIDTH)) & ((1 << WIDTH) - 1)

            # Quadrant reduction: map into [0, π/2)
            quadrant = (angle_full // QUARTER) % 4
            quadrants.append(quadrant)

            if quadrant == 0:
                reduced = angle_full
            elif quadrant == 1:
                # π/2..π  → reflect: π - θ  (i.e., HALF - angle)
                reduced = 2 * QUARTER - angle_full
            elif quadrant == 2:
                # π..3π/2 → θ - π
                reduced = angle_full - 2 * QUARTER
            else:
                # 3π/2..2π → 2π - θ
                reduced = 4 * QUARTER - angle_full

            reduced = reduced & ((1 << WIDTH) - 1)
            ctx.set(dut.angle, reduced)

        await ctx.tick()

        if i >= latency:
            sin_raw = ctx.get(dut.sin_out)
            cos_raw = ctx.get(dut.cos_out)
            # Interpret as signed
            if sin_raw >= (1 << (WIDTH - 1)):
                sin_raw -= 1 << WIDTH
            if cos_raw >= (1 << (WIDTH - 1)):
                cos_raw -= 1 << WIDTH
            sin_values.append(sin_raw / (1 << (WIDTH - 1)))
            cos_values.append(cos_raw / (1 << (WIDTH - 1)))


sim.add_testbench(testbench)

with sim.write_vcd("examples/sinusoide.vcd", "examples/sinusoide.gtkw"):
    sim.run()

# Trim to N_SAMPLES
sin_values = sin_values[:N_SAMPLES]
cos_values = cos_values[:N_SAMPLES]

# Apply quadrant correction to collected outputs
for idx in range(N_SAMPLES):
    q = quadrants[idx]
    s, c = sin_values[idx], cos_values[idx]
    if q == 0:
        pass  # sin(θ), cos(θ) — correct as-is
    elif q == 1:
        # sin(π-θ) = sin(θ),  cos(π-θ) = -cos(θ)
        cos_values[idx] = -c
    elif q == 2:
        # sin(π+θ) = -sin(θ), cos(π+θ) = -cos(θ)
        sin_values[idx] = -s
        cos_values[idx] = -c
    elif q == 3:
        # sin(2π-θ) = -sin(θ), cos(2π-θ) = cos(θ)
        sin_values[idx] = -s

# Reference
ref_angles = [i / N_SAMPLES * 2 * math.pi for i in range(N_SAMPLES)]
ref_sin = [math.sin(a) for a in ref_angles]
ref_cos = [math.cos(a) for a in ref_angles]

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.set_title("CORDIC Sin vs Reference")
ax1.plot(ref_angles, ref_sin, "b--", alpha=0.5, label="math.sin")
ax1.plot(ref_angles, sin_values, "r-", linewidth=0.8, label="FixSinCos sin")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_title("CORDIC Cos vs Reference")
ax2.plot(ref_angles, ref_cos, "b--", alpha=0.5, label="math.cos")
ax2.plot(ref_angles, cos_values, "r-", linewidth=0.8, label="FixSinCos cos")
ax2.set_xlabel("Angle (radians)")
ax2.set_ylabel("Amplitude")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle(f"FixSinCos CORDIC — {WIDTH}-bit, {N_SAMPLES} samples", fontsize=14)
fig.tight_layout()
plt.savefig("examples/sinusoide.png", dpi=150)
print("Saved examples/sinusoide.png")

