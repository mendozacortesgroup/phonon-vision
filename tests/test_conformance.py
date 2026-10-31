"""Conformance checker tests (Deliverable 5), including THE CROWN-JEWEL
self-test: a deliberately phase-corrupted candidate must PASS tier 2 (|e|
unchanged) and FAIL tier 3 (complex mismatch survives canonical alignment).

If that self-test ever stops failing at tier 3, the harness is not doing its
job — a magnitude-only check would wave the silent phase bug through.
"""
import json
import os

import numpy as np
import pytest

from phonon_core.conformance import check_against_reference, ConformanceResult

_HERE = os.path.dirname(__file__)
_NPZ = np.load(os.path.join(_HERE, "golden_kinematics.npz"))
with open(os.path.join(_HERE, "golden_kinematics.json")) as _f:
    _SIDE = json.load(_f)

_NAMES = list(_SIDE["modes"])
_FINITE = [n for n, m in _SIDE["modes"].items() if m["is_finite_q"]]


def _stack():
    """All golden modes as (N_modes,N_atoms,3) eigvecs + (N_modes,3) qpoints +
    (N_modes,) freqs + (N_atoms,) masses."""
    eig = np.array([_NPZ[f"{n}__eigvec_raw"] for n in _NAMES])
    q = np.array([_SIDE["modes"][n]["q_point"] for n in _NAMES])
    f = np.array([_SIDE["modes"][n]["frequency_THz"] for n in _NAMES])
    masses = _NPZ["_material_masses"]
    return eig, q, f, masses


def test_identity_passes_all_tiers():
    eig, q, f, masses = _stack()
    res = check_against_reference(eig, eig, masses, q,
                                  candidate_freqs=f, reference_freqs=f)
    assert isinstance(res, ConformanceResult)
    assert res.passed, str(res)
    assert res.tier1_frequencies.passed
    assert res.tier2_magnitude.passed
    assert res.tier3_complex_aligned.passed
    # finite-q modes are flagged so the caller can confirm tier 3 covered them
    assert res.has_finite_q
    assert sum(1 for x in res.finite_q_mask if x) == len(_FINITE)


def test_global_phase_is_absorbed_tier3():
    """A per-MODE arbitrary global phase (what diagonalization leaves) must NOT
    fail tier 3 — the canonical alignment removes it."""
    eig, q, f, masses = _stack()
    rng = np.random.default_rng(0)
    cand = eig.copy()
    for i in range(cand.shape[0]):
        cand[i] *= np.exp(1j * rng.uniform(0, 2 * np.pi))   # whole-mode rotation
    res = check_against_reference(cand, eig, masses, q)
    assert res.tier2_magnitude.passed
    assert res.tier3_complex_aligned.passed, \
        f"global phase should be absorbed: {res.tier3_complex_aligned}"


def test_tier1_frequency_mismatch_fails():
    eig, q, f, masses = _stack()
    bad = f.copy()
    bad[0] += 0.5  # 0.5 THz off
    res = check_against_reference(eig, eig, masses, q,
                                  candidate_freqs=bad, reference_freqs=f)
    assert not res.tier1_frequencies.passed
    assert res.first_failed_tier.name == "frequencies"
    # the eigenvector tiers are unaffected
    assert res.tier2_magnitude.passed and res.tier3_complex_aligned.passed


# ----------------------------------------------------------------------------
# THE CROWN-JEWEL SELF-TEST
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("name", _FINITE)
def test_per_atom_phase_corruption_passes_tier2_fails_tier3(name):
    """Take a finite-q reference mode, multiply ONE atom's eigenvector by
    exp(i*0.3), and assert: tier 2 (|e|) PASSES, tier 3 (complex, canonical-
    aligned) FAILS. This is the silent per-atom phase bug a magnitude check
    would miss."""
    ref = _NPZ[f"{name}__eigvec_raw"].copy()        # (N_atoms, 3)
    q = np.asarray(_SIDE["modes"][name]["q_point"])
    masses = _NPZ["_material_masses"]

    # corrupt the atom with the largest per-atom magnitude (guarantees the
    # rotation lands on real motion, so tier 3 has something to catch).
    per_atom_mag = np.linalg.norm(ref, axis=1)      # (N_atoms,)
    k = int(np.argmax(per_atom_mag))
    assert per_atom_mag[k] > 0.2, f"{name}: corrupt atom magnitude too small"

    cand = ref.copy()
    cand[k] = cand[k] * np.exp(1j * 0.3)            # ONE atom, intra-cell phase

    res = check_against_reference(cand, ref, masses, q[None, :])

    # tier 2: magnitudes are identical — a pure phase rotation preserves |e|.
    assert res.tier2_magnitude.passed, \
        f"{name}: |e| should be unchanged: {res.tier2_magnitude}"
    assert res.tier2_magnitude.max_abs_dev < 1e-12

    # tier 3: the per-atom phase survives canonical alignment — MUST fail.
    assert not res.tier3_complex_aligned.passed, \
        f"{name}: HARNESS BLIND SPOT — tier 3 passed a phase-corrupted mode!"
    assert res.tier3_complex_aligned.max_abs_dev > 0.05, \
        f"{name}: tier-3 deviation too small: {res.tier3_complex_aligned}"
    assert res.first_failed_tier.name.startswith("per-atom complex")


def test_crown_jewel_is_finite_q_specific():
    """Sanity: the corruption is exactly the bug class that is invisible at Γ to
    |e| metrics — confirm tier 2 cannot distinguish it on a finite-q mode."""
    name = _FINITE[0]
    ref = _NPZ[f"{name}__eigvec_raw"].copy()
    k = int(np.argmax(np.linalg.norm(ref, axis=1)))
    cand = ref.copy()
    cand[k] *= np.exp(1j * 0.3)
    # |e| per atom is bit-identical
    np.testing.assert_allclose(np.abs(cand), np.abs(ref), atol=1e-15, rtol=0)
