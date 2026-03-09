"""Shared test helpers for amaranth-fp Sollya reference tests."""
from __future__ import annotations

import math
import random
from typing import List, Tuple

from amaranth_fp.format import FPFormat
from amaranth_fp.testing.sollya_reference import SollyaReference


def assert_fp_equal(
    fmt: FPFormat,
    actual: Tuple[int, int, int, int],
    expected: Tuple[int, int, int, int],
    ulp_tolerance: int = 0,
) -> None:
    """Assert two FP values (exc, sign, exp, mant) are equal within ULP tolerance.

    Parameters
    ----------
    fmt : FPFormat
        The floating-point format.
    actual : tuple
        (exception, sign, exponent, mantissa) from DUT.
    expected : tuple
        (exception, sign, exponent, mantissa) from reference.
    ulp_tolerance : int
        Maximum allowed ULP difference for normal results.
    """
    a_exc, a_sign, a_exp, a_mant = actual
    e_exc, e_sign, e_exp, e_mant = expected

    # Exception must always match
    assert a_exc == e_exc, (
        f"Exception mismatch: actual={a_exc:#04b} expected={e_exc:#04b}"
    )

    # For zero/inf/nan, sign must match (except NaN sign is don't-care)
    if e_exc in (0b00, 0b10):  # zero or inf
        assert a_sign == e_sign, (
            f"Sign mismatch: actual={a_sign} expected={e_sign}"
        )
        return
    if e_exc == 0b11:  # NaN
        return  # NaN payload/sign is implementation-defined

    # Normal: check sign, then exp+mant within ULP tolerance
    assert a_sign == e_sign, (
        f"Sign mismatch: actual={a_sign} expected={e_sign}"
    )

    if ulp_tolerance == 0:
        assert a_exp == e_exp and a_mant == e_mant, (
            f"Value mismatch: actual=(exp={a_exp}, mant={a_mant:#x}) "
            f"expected=(exp={e_exp}, mant={e_mant:#x})"
        )
    else:
        # Compute absolute value as combined exp+mant for ULP comparison
        a_val = (a_exp << fmt.wf) | a_mant
        e_val = (e_exp << fmt.wf) | e_mant
        diff = abs(a_val - e_val)
        assert diff <= ulp_tolerance, (
            f"ULP difference {diff} exceeds tolerance {ulp_tolerance}: "
            f"actual=(exp={a_exp}, mant={a_mant:#x}) "
            f"expected=(exp={e_exp}, mant={e_mant:#x})"
        )


def random_fp_values(fmt: FPFormat, n: int = 100, seed: int = 42) -> List[float]:
    """Generate n random FP values valid for the given format.

    Includes a mix of small, medium, large, and near-boundary values.
    """
    rng = random.Random(seed)
    values: List[float] = []

    max_normal = (2.0 - 2.0 ** (-fmt.wf)) * (2.0 ** fmt.emax)
    min_normal = 2.0 ** fmt.emin

    for _ in range(n):
        category = rng.randint(0, 4)
        if category == 0:
            # Small values near min_normal
            v = min_normal * rng.uniform(1.0, 10.0)
        elif category == 1:
            # Large values near max_normal
            v = max_normal * rng.uniform(0.5, 1.0)
        elif category == 2:
            # Medium values around 1.0
            v = rng.uniform(0.1, 100.0)
        elif category == 3:
            # Powers of 2
            exp = rng.randint(fmt.emin, fmt.emax)
            v = 2.0 ** exp
        else:
            # Random sign
            v = rng.uniform(min_normal, max_normal)

        if rng.random() < 0.3:
            v = -v
        values.append(v)

    return values


def simulate_operator(dut: object, *inputs: int, latency: int | None = None) -> Tuple[int, int, int, int]:
    """Run an Amaranth simulation of an FP operator and return the output.

    This is a placeholder — actual simulation requires Amaranth's simulator.
    For now, this raises NotImplementedError to indicate that hardware
    simulation tests should be written separately.

    Parameters
    ----------
    dut : object
        Amaranth FP operator module.
    inputs : int
        Input values in internal format.
    latency : int or None
        Number of clock cycles to wait for output.

    Returns
    -------
    tuple
        (exception, sign, exponent, mantissa) of the output.
    """
    raise NotImplementedError(
        "Hardware simulation not yet implemented. "
        "Use SollyaReference directly for golden reference validation."
    )
