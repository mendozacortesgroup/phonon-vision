"""Shared frequency-comparison METRIC DOCTRINE for engine tests (0.6.88).

Extracted from tests/test_engine_dq_numpy.py (where it landed at 0.6.87) the
moment a second test file needed it — the doctrine must have ONE spelling.

THE RULE (architect, 0.6.86 audit): compare |dnu| only where |nu| > NU_FLOOR;
BELOW that compare |d lambda| (lambda = sign(nu) * nu^2) instead.

WHY: nu = sign(l) * sqrt(|l|) * F, so dnu = dl / (2 nu) — the map from
eigenvalue noise to frequency noise DIVERGES as nu -> 0. Measured on a Gamma
acoustic zero: 6.8e-15 THz^2 of lambda-noise shows up as 2.4e-8 THz of
nu-noise. A |dnu| tolerance near zero therefore tests the eigensolver's float
noise, not the engine — and it is not even PORTABLE: the same engine bytes
passed a raw |dnu| < 1e-8 pin under one OpenBLAS build and failed it under
another (0.6.87, SnSe2 Tier-3). Materials with genuine instabilities also
carry many zero CROSSINGS away from Gamma, so the doctrine applies at
generic q, not just at the acoustic zeros.
"""
import numpy as np

#: THz — above this, |dnu| is meaningful; below, compare |d lambda|.
NU_FLOOR = 1e-3

#: THz^2 — |d lambda| bound near zero (measured agreement is ~1e-14).
LAM_TOL = 1e-9


def doctrine_agreement(f_ref, f_eng):
    """(max|dnu| where |nu|>NU_FLOOR, max|d lambda| where |nu|<=NU_FLOOR).

    The near-zero half is compared in lambda = sign(nu)*nu^2 — see NU_FLOOR."""
    f_ref = np.asarray(f_ref, dtype=np.float64)
    f_eng = np.asarray(f_eng, dtype=np.float64)
    big = np.abs(f_ref) > NU_FLOOR

    def lam(x):
        return np.sign(x) * x ** 2

    d_nu = float(np.max(np.abs(f_eng[big] - f_ref[big]))) if big.any() else 0.0
    d_lam = (float(np.max(np.abs(lam(f_eng[~big]) - lam(f_ref[~big]))))
             if (~big).any() else 0.0)
    return d_nu, d_lam
