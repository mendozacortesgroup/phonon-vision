"""Conformance checker: the gate object for validating any candidate phonon
eigenvector set against a frozen reference.

This is the reusable core of the conformance harness. A *candidate* is the
output of any importer or engine that claims to reproduce phonon_core's physics
— a future file-format importer, the planned JavaScript web-kinematics engine,
or a refactor of phonon_core itself. ``check_against_reference`` compares the
candidate to a trusted reference in THREE TIERS and reports which tier failed,
so the caller learns *what kind* of disagreement they have.

The three tiers
---------------
tier 1 — frequencies
    The scalar dispersion. A frequency mismatch means the dynamical matrix or
    its diagonalization differs; the eigenvectors are then comparing apples to
    oranges. (Skipped if no frequencies are supplied.)

tier 2 — per-atom magnitude ``|e|``
    The mode's *shape* in magnitude space: which atoms move how much. EVERY
    magnitude-based metric — species participation weights, the inverse
    participation ratio, the projected DOS — depends only on this. A check that
    stops here is necessary but NOT sufficient (see below).

tier 3 — per-atom COMPLEX ``e`` after canonical-phase alignment  ← LOAD-BEARING
    The full complex polarization, compared after removing the one physically
    arbitrary degree of freedom (the global phase from diagonalization) with the
    canonical rule: rotate so the largest-magnitude component is real-positive
    (the same rule as ``kinematics.canonicalize_eigenvector_phase``).

    A *global* phase difference is physically meaningless and is absorbed by the
    alignment. A *per-atom* (intra-cell) phase error is NOT — it is a different
    mode, and at finite q it changes the animation. Because the alignment removes
    only one global phase, a single corrupted atom survives tier 3 and fails it.

    Tie-robust pivot.  The engine's per-vector ``canonicalize`` chooses its pivot
    by ``argmax`` over its OWN magnitudes. That is fine for a fixed vector, but it
    is non-deterministic across a global-phase rotation when two components share
    the maximum magnitude (a real standing-wave mode): the fp noise of the complex
    multiply flips which tied component ``argmax`` returns, so the two canonical
    forms would differ by ~unity and a pure global-phase difference would falsely
    fail. To stay robust the checker derives the pivot ONCE from the REFERENCE
    (``argmax|ref|``) and rephases BOTH candidate and reference by that SAME index.
    This is still the canonical rule — the reference defines "the largest-
    magnitude component" — but it is invariant to a global phase for every mode,
    ties included. (See docs/conformance.md, "canonical-phase tie".)

Why tier 3 at finite q is the real test
---------------------------------------
The dangerous bug class is a per-atom phase error. It leaves every |e|-based
metric unchanged (so tier 2 passes), and at q = Γ all cells share the phase
factor ``exp(i q·R) = 1`` so it is invisible. It only manifests as a wrong
*relative* atomic phase, i.e. in the complex eigenvector, and its effect on the
Bloch animation grows with q. Hence: **tier 3 evaluated at a finite-q mode is
the assertion that actually guards the physics.** A harness that only checks
tier 2, or only checks Γ, would pass a phase-corrupted importer silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _align_to_pivot(eigvec: np.ndarray, pivot_idx: int) -> np.ndarray:
    """Rephase ``eigvec`` by the global phase that makes its flat component at
    ``pivot_idx`` real and positive (the canonical rule, with the pivot chosen
    externally so the same index can be applied to two vectors). Returns a copy
    unchanged if that component is ~zero.
    """
    e = np.asarray(eigvec, dtype=np.complex128)
    flat = e.reshape(-1)
    pivot = flat[pivot_idx]
    mag = abs(pivot)
    if mag < 1e-12:
        return e.copy()
    return e * (np.conj(pivot) / mag)


@dataclass
class TierResult:
    """Outcome of a single tier."""
    name: str
    passed: bool
    max_abs_dev: float
    detail: str = ""

    def __str__(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        dev = "n/a" if np.isnan(self.max_abs_dev) else f"{self.max_abs_dev:.3e}"
        return f"[{flag}] {self.name}: max|Δ|={dev} {self.detail}".rstrip()


@dataclass
class ConformanceResult:
    """Structured per-tier result of :func:`check_against_reference`."""
    tier1_frequencies: TierResult
    tier2_magnitude: TierResult
    tier3_complex_aligned: TierResult
    n_modes: int
    n_atoms: int
    finite_q_mask: list  # per-mode: True if |q|>0 (None if no qpoints given)

    @property
    def passed(self) -> bool:
        """True only if every *evaluated* tier passed."""
        return all(t.passed for t in self.tiers)

    @property
    def tiers(self) -> tuple:
        return (self.tier1_frequencies, self.tier2_magnitude,
                self.tier3_complex_aligned)

    @property
    def first_failed_tier(self) -> Optional[TierResult]:
        for t in self.tiers:
            if not t.passed:
                return t
        return None

    @property
    def has_finite_q(self) -> bool:
        return bool(self.finite_q_mask) and any(bool(x) for x in self.finite_q_mask)

    def __str__(self) -> str:
        head = (f"ConformanceResult({'PASS' if self.passed else 'FAIL'}; "
                f"{self.n_modes} modes, {self.n_atoms} atoms, "
                f"finite_q={'yes' if self.has_finite_q else 'no'})")
        return head + "\n  " + "\n  ".join(str(t) for t in self.tiers)


def _as_mode_stack(e: np.ndarray) -> np.ndarray:
    """Coerce to (N_modes, N_atoms, 3) complex; accept a single (N_atoms,3)."""
    e = np.asarray(e, dtype=np.complex128)
    if e.ndim == 2:
        e = e[np.newaxis, ...]
    if e.ndim != 3 or e.shape[2] != 3:
        raise ValueError(f"eigenvectors must be (N_modes,N_atoms,3) or "
                         f"(N_atoms,3); got {e.shape}")
    return e


def check_against_reference(
    candidate_eigvecs: np.ndarray,
    reference_eigvecs: np.ndarray,
    masses: Optional[np.ndarray],
    qpoints: Optional[np.ndarray],
    atol: float = 1e-8,
    rtol: float = 1e-5,
    *,
    candidate_freqs: Optional[np.ndarray] = None,
    reference_freqs: Optional[np.ndarray] = None,
) -> ConformanceResult:
    """Compare ``candidate`` eigenvectors to ``reference`` in three tiers.

    Parameters
    ----------
    candidate_eigvecs, reference_eigvecs : (N_modes, N_atoms, 3) complex
        Polarization vectors to compare. A single ``(N_atoms, 3)`` mode is
        accepted and promoted to a 1-mode stack.
    masses : (N_atoms,) float or None
        Atomic masses of the mode bundle. Not required by the three eigenvector
        tiers (which are mass-independent); accepted so callers can pass a
        complete mode description and validated for shape consistency. Reserved
        for an optional displacement-space tier.
    qpoints : (N_modes, 3) float or None
        Fractional q of each mode. Used to flag which modes are finite-q in the
        result (``finite_q_mask``) — the caller should confirm tier 3 passed on
        a finite-q mode, which is the load-bearing assertion. ``None`` leaves
        the mask unset.
    atol, rtol : float
        Tolerances for ``numpy.allclose`` on each tier.
    candidate_freqs, reference_freqs : (N_modes,) float or None
        Frequencies for tier 1. If either is None, tier 1 is skipped (reported
        as passed with detail="skipped").

    Returns
    -------
    ConformanceResult
        Per-tier pass/fail with max abs deviations; ``.first_failed_tier`` and
        ``.passed`` summarize.
    """
    cand = _as_mode_stack(candidate_eigvecs)
    ref = _as_mode_stack(reference_eigvecs)
    if cand.shape != ref.shape:
        raise ValueError(f"candidate {cand.shape} and reference {ref.shape} "
                         f"eigenvectors must have the same shape")
    n_modes, n_atoms = cand.shape[0], cand.shape[1]

    if masses is not None:
        masses = np.asarray(masses, dtype=np.float64).reshape(-1)
        if masses.shape[0] != n_atoms:
            raise ValueError(f"masses length {masses.shape[0]} != N_atoms "
                             f"{n_atoms}")

    if qpoints is not None:
        q = np.asarray(qpoints, dtype=np.float64).reshape(-1, 3)
        if q.shape[0] != n_modes:
            raise ValueError(f"qpoints {q.shape[0]} != N_modes {n_modes}")
        finite_mask = [bool(np.linalg.norm(q[i]) > 1e-8) for i in range(n_modes)]
    else:
        finite_mask = [None] * n_modes

    # ---- tier 1: frequencies ----
    if candidate_freqs is not None and reference_freqs is not None:
        cf = np.asarray(candidate_freqs, dtype=np.float64).reshape(-1)
        rf = np.asarray(reference_freqs, dtype=np.float64).reshape(-1)
        if cf.shape != rf.shape:
            raise ValueError("candidate/reference frequencies shape mismatch")
        dev1 = float(np.max(np.abs(cf - rf))) if cf.size else 0.0
        ok1 = bool(np.allclose(cf, rf, atol=atol, rtol=rtol))
        tier1 = TierResult("frequencies", ok1, dev1)
    else:
        tier1 = TierResult("frequencies", True, float("nan"),
                           "skipped (no frequencies provided)")

    # ---- tier 2: per-atom magnitude |e| ----
    dev2 = float(np.max(np.abs(np.abs(cand) - np.abs(ref))))
    ok2 = bool(np.allclose(np.abs(cand), np.abs(ref), atol=atol, rtol=rtol))
    tier2 = TierResult("per-atom |e|", ok2, dev2)

    # ---- tier 3: complex e after canonical-phase alignment (per mode) ----
    # Pivot is derived from the REFERENCE (argmax|ref|) and applied to BOTH, so
    # the alignment is invariant to a global phase even under a magnitude tie.
    dev3 = 0.0
    ok3 = True
    for i in range(n_modes):
        pivot_idx = int(np.argmax(np.abs(ref[i].reshape(-1))))
        b = _align_to_pivot(ref[i], pivot_idx)
        a = _align_to_pivot(cand[i], pivot_idx)
        dev3 = max(dev3, float(np.max(np.abs(a - b))) if a.size else 0.0)
        if not np.allclose(a, b, atol=atol, rtol=rtol):
            ok3 = False
    tier3 = TierResult("per-atom complex e (canonical-aligned)", ok3, dev3,
                       "LOAD-BEARING at finite q")

    return ConformanceResult(
        tier1_frequencies=tier1,
        tier2_magnitude=tier2,
        tier3_complex_aligned=tier3,
        n_modes=n_modes,
        n_atoms=n_atoms,
        finite_q_mask=finite_mask,
    )
