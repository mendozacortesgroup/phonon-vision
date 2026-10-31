"""The packaged ADP module and the standalone copy must not drift.

The merge that made this tree a complete package left two copies of the
displacement-parameter implementation: the packaged
``src/phonon_core/adp.py`` (what a wheel carries) and the pre-merge
``python/phonon_adp.py`` (retained unmodified). Duplicated numerics are a
standing hazard — someone fixes one and the other silently keeps the old
answer — so the duplication is converted here into a CHECKED invariant.

These tests fail the moment the two disagree, which is the signal to delete
the standalone copy rather than to patch both.
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from phonon_core import adp as packaged      # noqa: E402
import phonon_adp as standalone              # noqa: E402


def random_mesh(seed=20260730, nq=4, nb=6, na=3):
    """Complex eigenvectors, normalized per mode — exercises the full
    Re[e conj(e)] path including the imaginary contribution."""
    rng = np.random.default_rng(seed)
    er = rng.normal(size=(nq, nb, na, 3))
    ei = rng.normal(size=(nq, nb, na, 3))
    norm = np.sqrt((er ** 2 + ei ** 2).sum(axis=(2, 3), keepdims=True))
    freq = rng.uniform(0.05, 14.0, size=(nq, nb))
    masses = np.array([12.0107, 118.71, 78.96][:na], dtype=float)
    return freq, er / norm, ei / norm, masses


def test_constants_are_identical():
    for name in ("HBAR", "KB", "AMU", "THZ_TO_RAD_S", "DEFAULT_FREQ_FLOOR_THZ"):
        assert getattr(packaged, name) == getattr(standalone, name), name


@pytest.mark.parametrize("T", [1e-3, 10.0, 300.0, 1500.0])
@pytest.mark.parametrize("floor", [0.0, 0.1, 1.0])
def test_tensors_agree_bitwise(T, floor):
    """Same inputs, same code path — the agreement should be exact, not
    approximate. A tolerance here would hide a real divergence."""
    freq, er, ei, masses = random_mesh()
    Ua, ia = packaged.adp_tensors(freq, er, ei, masses, T, freq_floor_thz=floor)
    Ub, ib = standalone.adp_tensors(freq, er, ei, masses, T, freq_floor_thz=floor)
    assert np.array_equal(Ua, Ub), f"max |delta| = {np.abs(Ua - Ub).max()}"
    assert ia == ib


def test_classical_branch_agrees():
    freq, er, ei, masses = random_mesh()
    Ua, _ = packaged.adp_tensors(freq, er, ei, masses, 900.0, classical=True)
    Ub, _ = standalone.adp_tensors(freq, er, ei, masses, 900.0, classical=True)
    assert np.array_equal(Ua, Ub)


def test_derived_helpers_agree():
    freq, er, ei, masses = random_mesh()
    U, _ = packaged.adp_tensors(freq, er, ei, masses, 300.0)
    assert np.array_equal(packaged.u_isotropic(U), standalone.u_isotropic(U))
    va, wa = packaged.principal_axes(U[0])
    vb, wb = standalone.principal_axes(U[0])
    assert np.array_equal(va, vb) and np.array_equal(wa, wb)
    for pct in (50, 90, 99):
        assert packaged.ellipsoid_scale(pct) == standalone.ellipsoid_scale(pct)


def test_both_reject_the_same_bad_input():
    freq, er, ei, masses = random_mesh()
    for mod in (packaged, standalone):
        with pytest.raises(ValueError):
            mod.adp_tensors(freq, er, ei, masses, 0.0)
        with pytest.raises(ValueError):
            mod.adp_tensors(freq, er, ei, np.array([1.0]), 300.0)


def test_packaged_module_is_importable_as_part_of_the_package():
    """The point of the merge: the ADP reference ships INSIDE phonon_core, so
    a wheel built from src/ carries it and `import phonon_core.adp` works
    without the repository layout."""
    import phonon_core
    from phonon_core.adp import adp_tensors, ellipsoid_scale  # noqa: F401
    assert hasattr(phonon_core, "__version__")
