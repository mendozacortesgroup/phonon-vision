"""Atomic displacement parameters (ADPs) from a phonon mesh.

Packaged location. This is the module a wheel built from ``src/`` carries, and
the one to import as ``from phonon_core.adp import adp_tensors``.

A standalone copy of the same implementation lives at ``python/phonon_adp.py``
from before the data layer was merged into this tree. It is retained
unmodified, and ``tests/test_adp_parity.py`` asserts the two agree numerically
so the duplication cannot drift silently. When the standalone copy is
eventually removed, this file is the one to keep.

This is the REFERENCE implementation — the oracle the browser module
``web/adp.js`` is validated against. Pure numpy; no phonopy, no scipy.

The physics
-----------
In the harmonic approximation each atom is a sum of independent oscillators,
one per mode. The thermal average of the displacement dyad gives the
mean-square displacement tensor of atom ``s`` (the quantity a crystallographer
refines from diffraction and draws as an ORTEP ellipsoid):

    U_s^{ab}(T) = (1/N_q) sum_{q,nu}  hbar / (2 m_s w_{q,nu})
                  * coth( hbar w_{q,nu} / (2 k_B T) )
                  * Re[ e_{s,a}^{q,nu} conj(e_{s,b}^{q,nu}) ]

with

    w = 2 pi nu             angular frequency (nu is our signed THz)
    m_s                     mass of atom s
    e_{s}^{q,nu}            the MASS-WEIGHTED polarization vector, normalized
                            sum_s |e_s|^2 = 1 (our engine's convention)
    N_q                     number of q-points averaged over

References for the physics (NOT for any code):
  - W. Cochran, "The Dynamics of Atoms in Crystals" (1973), ch. 3 — the
    harmonic mean-square displacement.
  - B. T. M. Willis & A. W. Pryor, "Thermal Vibrations in Crystallography"
    (CUP 1975) — the U tensor and its refinement conventions.
  - C. K. Johnson, ORTEP, ORNL-3794 (1965) — the probability-ellipsoid
    convention reproduced in ``ellipsoid_scale``.
  - M. T. Dove, "Introduction to Lattice Dynamics" (1993), eq. 4.16 — the
    eigenvector normalization this module assumes.

The two limits worth knowing, both asserted in the test-suite:

  * T -> 0 :  coth -> 1, so U -> sum hbar / (2 m w) > 0. Zero-point motion.
    An ADP that vanishes at 0 K is a classical (wrong) ADP.
  * T -> large : coth(x) -> 1/x, so U -> k_B T / (m w^2). Equipartition.

Sampling caveat (read before quoting a number)
----------------------------------------------
The 1/w prefactor diverges as w -> 0, so the acoustic branches near Gamma
dominate and the sum is only conditionally convergent on a discrete mesh.
``freq_floor_thz`` drops modes below a threshold and the count is REPORTED,
never silently applied. For a 2D material the out-of-plane (flexural) branch
has w ~ q^2, so U_zz keeps growing as the mesh is refined — a real physical
effect (Mermin-Wagner), not a bug. Treat U from a coarse mesh as
qualitative: excellent for showing anisotropy and temperature trends,
not a substitute for a converged calculation.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "HBAR", "KB", "AMU", "THZ_TO_RAD_S",
    "adp_tensors", "principal_axes", "ellipsoid_scale", "u_isotropic",
]

# CODATA 2018, matching the rest of the project.
HBAR = 1.054571817e-34          # J s
KB = 1.380649e-23               # J / K
AMU = 1.66053906660e-27         # kg
THZ_TO_RAD_S = 2.0 * np.pi * 1.0e12

#: Modes below this |nu| (THz) are excluded by default. Chosen to sit above
#: the numerical dust at Gamma while keeping every genuine acoustic mode a
#: coarse mesh actually resolves.
DEFAULT_FREQ_FLOOR_THZ = 0.1


def _coth(x: np.ndarray) -> np.ndarray:
    """coth(x) for x > 0, overflow-free.

    numpy's tanh saturates to exactly 1.0 for x >~ 20, so 1/tanh is safe and
    exact in that regime; the clip only guards the x -> 0 pole, which the
    frequency floor already excludes.
    """
    return 1.0 / np.tanh(np.clip(x, 1e-12, None))


def adp_tensors(frequency_thz, eigenvectors_real, eigenvectors_imag, masses_amu,
                temperature_k, *, freq_floor_thz=DEFAULT_FREQ_FLOOR_THZ,
                classical=False):
    """Mean-square displacement tensors U_s(T), in Angstrom^2.

    Parameters
    ----------
    frequency_thz : (Nq, Nb) signed THz, the engine convention.
    eigenvectors_real, eigenvectors_imag : (Nq, Nb, Na, 3) mass-weighted
        polarization vectors with sum_s |e_s|^2 = 1 per mode.
    masses_amu : (Na,) atomic masses.
    temperature_k : float, > 0.
    freq_floor_thz : modes with |nu| below this are dropped from the sum.
    classical : if True use the equipartition weight k_B T / w^2 instead of
        the quantum one. Exposed ONLY so the test-suite can assert the two
        agree in the high-temperature limit; never use it for display.

    Returns
    -------
    U : (Na, 3, 3) symmetric tensors in Angstrom^2.
    info : dict with the sampling facts a caller must be able to quote:
        n_q, n_modes_total, n_modes_dropped, freq_floor_thz, temperature_k,
        f_min_kept_thz.
    """
    f = np.asarray(frequency_thz, dtype=np.float64)
    er = np.asarray(eigenvectors_real, dtype=np.float64)
    ei = np.asarray(eigenvectors_imag, dtype=np.float64)
    m = np.asarray(masses_amu, dtype=np.float64)

    if f.ndim != 2:
        raise ValueError(f"frequency must be (Nq, Nb), got {f.shape}")
    if er.shape != ei.shape or er.ndim != 4 or er.shape[:2] != f.shape:
        raise ValueError(
            f"eigenvectors must be (Nq, Nb, Na, 3) matching frequency {f.shape}; "
            f"got real {er.shape}, imag {ei.shape}")
    if er.shape[2] != m.shape[0]:
        raise ValueError(
            f"atom count mismatch: eigenvectors have {er.shape[2]}, masses {m.shape[0]}")
    if temperature_k <= 0:
        raise ValueError("temperature must be > 0 K (use a small value for the "
                         "zero-point limit; T=0 is a removable singularity)")

    n_q, n_b, n_a, _ = er.shape
    keep = np.abs(f) >= float(freq_floor_thz)

    # |w|: an imaginary mode (negative nu in our sign convention) is not a
    # thermally populated oscillator. Using |nu| keeps the arithmetic finite
    # and is reported through n_modes_dropped/f_min_kept if it matters; a
    # genuinely unstable material's ADP is meaningless anyway.
    w = THZ_TO_RAD_S * np.abs(f)

    if classical:
        amp = np.where(keep, KB * float(temperature_k) / np.where(w > 0, w * w, 1.0), 0.0)
    else:
        x = HBAR * w / (2.0 * KB * float(temperature_k))
        amp = np.where(keep, HBAR / (2.0 * np.where(w > 0, w, 1.0)) * _coth(x), 0.0)

    # Re[e_a conj(e_b)] = er_a er_b + ei_a ei_b  (the imaginary part is the
    # antisymmetric piece and cancels in the thermal average).
    U = np.empty((n_a, 3, 3), dtype=np.float64)
    for s in range(n_a):
        dyad = (np.einsum("qba,qbc->qbac", er[:, :, s, :], er[:, :, s, :])
                + np.einsum("qba,qbc->qbac", ei[:, :, s, :], ei[:, :, s, :]))
        U[s] = np.einsum("qb,qbac->ac", amp, dyad) / (m[s] * AMU) / n_q

    U *= 1.0e20                                   # m^2 -> Angstrom^2
    U = 0.5 * (U + np.transpose(U, (0, 2, 1)))    # symmetrize away fp dust

    kept = f[keep]
    info = {
        "n_q": int(n_q),
        "n_modes_total": int(f.size),
        "n_modes_dropped": int((~keep).sum()),
        "freq_floor_thz": float(freq_floor_thz),
        "temperature_k": float(temperature_k),
        "f_min_kept_thz": float(np.abs(kept).min()) if kept.size else None,
    }
    return U, info


def u_isotropic(U):
    """U_iso = trace/3, in Angstrom^2 — the single number refinements quote."""
    return np.trace(np.asarray(U, dtype=np.float64), axis1=-2, axis2=-1) / 3.0


def principal_axes(U):
    """Eigen-decompose a symmetric (3,3) U into (eigenvalues, eigenvectors).

    Eigenvalues ascending; eigenvectors are COLUMNS, so ``vecs[:, i]`` is the
    axis for ``vals[i]``. Values are mean-square displacements along each
    principal direction (Angstrom^2); their square roots are the RMS
    displacements that scale the drawn ellipsoid.
    """
    vals, vecs = np.linalg.eigh(np.asarray(U, dtype=np.float64))
    return vals, vecs


#: ORTEP probability scale factors: the ellipsoid with semi-axes
#: c * sqrt(eigenvalue) encloses the given fraction of a trivariate normal.
#: c solves P(chi^2_3 <= c^2) = p.
_ORTEP_C = {50: 1.5381722, 90: 2.5002777, 99: 3.3682279}


def ellipsoid_scale(probability_percent=50):
    """Return the ORTEP scale factor c for a probability level.

    Semi-axis i of the drawn ellipsoid is ``c * sqrt(U_eigenvalue_i)``.
    Accepts 50, 90 or 99 (the conventional levels).
    """
    p = int(probability_percent)
    if p not in _ORTEP_C:
        raise ValueError(f"probability must be one of {sorted(_ORTEP_C)}, got {p}")
    return _ORTEP_C[p]
