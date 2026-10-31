"""Golden kinematics regression (Deliverable 2) + convention assertions
(Deliverable 4), against the vendored golden fixture.

These are self-contained: they reload tests/golden_kinematics.{npz,json} and
re-run phonon_core's kinematics on the identical inputs, so they need neither
Blender nor the multi-MB phonopy source files. The fixture freezes a Γ acoustic,
a Γ optical, and — crucially — three finite-q modes (M, K, generic); the
convention assertions are checked on ALL of them, finite q included.
"""
import json
import os

import numpy as np
import pytest

from phonon_core.types import Material, PhononMode
from phonon_core.kinematics import (
    canonicalize_eigenvector_phase, canonicalize_mode, compute_bloch_phases,
    compute_displacements_at_phase, compute_trajectory,
)

_HERE = os.path.dirname(__file__)
_NPZ = np.load(os.path.join(_HERE, "golden_kinematics.npz"))
with open(os.path.join(_HERE, "golden_kinematics.json")) as _f:
    _SIDE = json.load(_f)

_MODE_NAMES = list(_SIDE["modes"])
_FINITE_Q = [n for n, m in _SIDE["modes"].items() if m["is_finite_q"]]
_REGRESSION_ATOL = 1e-10   # Deliverable 2 reproduction ceiling


def _material() -> Material:
    return Material(
        lattice=_NPZ["_material_lattice"],
        reciprocal_lattice=_NPZ["_material_reciprocal"],
        symbols=list(_SIDE["material_symbols"]),
        masses=_NPZ["_material_masses"],
        fractional_coords=_NPZ["_material_fractional"],
    )


def _mode(name: str) -> PhononMode:
    m = _SIDE["modes"][name]
    return PhononMode(
        q_point=np.asarray(m["q_point"], dtype=np.float64),
        band_index=int(m["band_index"]),
        frequency=float(m["frequency_THz"]),
        eigenvector=_NPZ[f"{name}__eigvec_raw"],
    )


def test_fixture_has_finite_q_modes():
    """The contract is meaningless without finite-q modes (they alone expose
    phase-convention errors)."""
    assert len(_FINITE_Q) >= 3, f"need finite-q modes, got {_FINITE_Q}"


# ----------------------------------------------------------------------------
# Deliverable 2 — regression: re-run kinematics, reproduce the frozen frames.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("name", _MODE_NAMES)
def test_kinematics_reproduces_golden(name):
    mat = _material()
    mode = _mode(name)
    canon = canonicalize_mode(mode)
    n_int = _NPZ["_slot_n_int"]
    j = _NPZ["_slot_j"]
    bloch = compute_bloch_phases(canon.q_point, n_int)

    amp = _SIDE["inputs"]["amplitude"]
    phi = _SIDE["inputs"]["phi"]
    nfr = _SIDE["inputs"]["n_frames"]
    mw = _SIDE["inputs"]["mass_weighted"]

    disp_phi = compute_displacements_at_phase(
        canon, mat, amp, phi, atom_indices=j, bloch_phases=bloch, mass_weighted=mw)
    disp_traj = compute_trajectory(
        canon, mat, amp, nfr, atom_indices=j, bloch_phases=bloch, mass_weighted=mw)

    np.testing.assert_allclose(disp_phi, _NPZ[f"{name}__disp_phi"],
                               atol=_REGRESSION_ATOL, rtol=0)
    np.testing.assert_allclose(disp_traj, _NPZ[f"{name}__disp_traj"],
                               atol=_REGRESSION_ATOL, rtol=0)


def test_canonicalization_is_deterministic():
    """Re-canonicalizing the raw eigenvector reproduces the frozen canonical
    one (guards the canonical-phase rule against drift)."""
    for name in _MODE_NAMES:
        raw = _NPZ[f"{name}__eigvec_raw"]
        again = canonicalize_eigenvector_phase(raw)
        np.testing.assert_allclose(again, _NPZ[f"{name}__eigvec_canon"],
                                   atol=_REGRESSION_ATOL, rtol=0)


# ----------------------------------------------------------------------------
# Deliverable 4 — convention assertions (the hub contract as executable checks).
# Checked on every mode, finite q included.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("name", _MODE_NAMES)
def test_eigenvector_unit_normalized(name):
    """sum_s |e_s|^2 = 1 (mass-weighted normal-mode convention)."""
    raw = _NPZ[f"{name}__eigvec_raw"]
    norm_sq = float(np.sum(np.abs(raw) ** 2))
    assert abs(norm_sq - 1.0) < 1e-6, f"{name}: ||e||^2={norm_sq}"


@pytest.mark.parametrize("name", _MODE_NAMES)
def test_canonical_phase_satisfied(name):
    """The engine's chosen pivot — argmax over the magnitudes of the vector it
    canonicalizes (i.e. argmax|raw|) — is real and positive after rephasing.

    Note: we check the ENGINE'S pivot, not argmax|canon|. Under an exact
    magnitude tie (see test_canonical_phase_tie_is_a_known_property) the two can
    differ, because fp noise from the rephasing multiply flips which tied
    component argmax returns. The engine's guarantee is about the pivot it
    selected, and that is what we assert here.
    """
    raw = _NPZ[f"{name}__eigvec_raw"]
    canon = _NPZ[f"{name}__eigvec_canon"]
    pivot_idx = int(np.argmax(np.abs(raw.reshape(-1))))  # the engine's pivot
    pivot = canon.reshape(-1)[pivot_idx]
    assert abs(pivot.imag) < 1e-9, f"{name}: pivot imag={pivot.imag}"
    assert pivot.real > 0, f"{name}: pivot real={pivot.real}"


def test_canonical_phase_tie_is_a_known_property():
    """REPORTED FINDING (not an engine bug): the 'generic' WSe2 mode is a real
    standing-wave mode with an EXACT magnitude tie at its maximum (two co-equal
    anti-phase Cartesian components). The engine's canonicalize is deterministic
    for a FIXED input, but argmax-based pivot selection is non-deterministic
    across a global-phase rotation under such a tie. This is why the conformance
    checker aligns tier 3 with a shared reference-derived pivot rather than
    re-deriving argmax on each vector. Documented here so the property is not
    mistaken for drift later.
    """
    raw = _NPZ["generic__eigvec_raw"].reshape(-1)
    mags = np.sort(np.abs(raw))[::-1]
    assert mags[0] - mags[1] == 0.0, "expected an exact magnitude tie at the max"
    # engine canonicalize is deterministic for fixed input
    c1 = canonicalize_eigenvector_phase(_NPZ["generic__eigvec_raw"])
    c2 = canonicalize_eigenvector_phase(_NPZ["generic__eigvec_raw"])
    assert np.array_equal(c1, c2)


@pytest.mark.parametrize("name", _MODE_NAMES)
def test_eigenvector_shape_dtype(name):
    raw = _NPZ[f"{name}__eigvec_raw"]
    assert raw.ndim == 2 and raw.shape[1] == 3, f"{name}: shape {raw.shape}"
    assert raw.shape[0] == _SIDE["modes"][name]["n_atoms"]
    assert raw.dtype == np.complex128


@pytest.mark.parametrize("name", _MODE_NAMES)
def test_mass_weighting_convention(name):
    """Mass-weighting divides Cartesian displacement by sqrt(m_s): so
    u_weighted * sqrt(m_slot) == u_unweighted."""
    mat = _material()
    canon = canonicalize_mode(_mode(name))
    j = _NPZ["_slot_j"]
    bloch = compute_bloch_phases(canon.q_point, _NPZ["_slot_n_int"])
    amp, phi = _SIDE["inputs"]["amplitude"], _SIDE["inputs"]["phi"]
    u_w = compute_displacements_at_phase(canon, mat, amp, phi, atom_indices=j,
                                         bloch_phases=bloch, mass_weighted=True)
    u_u = compute_displacements_at_phase(canon, mat, amp, phi, atom_indices=j,
                                         bloch_phases=bloch, mass_weighted=False)
    sqrt_m = np.sqrt(mat.masses[j])[:, np.newaxis]
    np.testing.assert_allclose(u_w * sqrt_m, u_u, atol=1e-12, rtol=0)
