"""Tests for phonon_core.expand (unified supercell expansion).

The load-bearing test is the halo regression: expand_cell(mode="halo") must
reproduce the supercell+image-atom output of add-on 0.5.3
(generate_supercell_slots + generate_image_atom_slots) — integer topology
(n_int, j) bitwise, positions geometry-identical to 1e-9 Å (r_cart is matmul
output and BLAS builds legally differ at the last ulp; see
_assert_halo_matches). The golden arrays in golden_expand.npz were captured
from that verbatim 0.5.3 code and stay FROZEN; see tests/capture_golden.py.

Material fixtures (lattice/symbols/masses/fractional_coords for SnSe2 and
WSe2) live in fixtures/materials.json — extracted from the phonopy band.yaml
files, which are too large to vendor.
"""
import json
import os

import numpy as np
import pytest

from phonon_core.types import Material
from phonon_core.expand import expand_cell, ExpandedAtoms

_HERE = os.path.dirname(__file__)
_GOLDEN = np.load(os.path.join(_HERE, "golden_expand.npz"))
with open(os.path.join(_HERE, "fixtures", "materials.json")) as _f:
    _MATERIALS = json.load(_f)

# golden key -> (fixture name, (nx, ny, nz))
_HALO_CASES = {
    "snse2_3x3x1": ("snse2", (3, 3, 1)),
    "wse2_3x3x1":  ("wse2", (3, 3, 1)),
}


def _material(name: str) -> Material:
    d = _MATERIALS[name]
    return Material(
        lattice=np.array(d["lattice"]),
        reciprocal_lattice=np.array(d["reciprocal_lattice"]),
        symbols=list(d["symbols"]),
        masses=np.array(d["masses"]),
        fractional_coords=np.array(d["fractional_coords"]),
    )


# ---------------------------------------------------------------------------
# strict
# ---------------------------------------------------------------------------
def test_strict_count_and_indices_wse2():
    """WSe2 3x3x1 strict -> exactly n_atoms_uc * 9; all n_int in {0,1,2}^2 x {0}."""
    mat = _material("wse2")
    nx, ny, nz = 3, 3, 1
    ea = expand_cell(mat, nx, ny, nz, mode="strict")
    assert len(ea) == mat.num_atoms * nx * ny * nz == 27
    # n_int components within the strict box
    assert set(ea.n_int[:, 0].tolist()) == {0, 1, 2}
    assert set(ea.n_int[:, 1].tolist()) == {0, 1, 2}
    assert set(ea.n_int[:, 2].tolist()) == {0}
    assert ((ea.n_int[:, 0] >= 0) & (ea.n_int[:, 0] < nx)).all()
    assert ((ea.n_int[:, 1] >= 0) & (ea.n_int[:, 1] < ny)).all()
    assert ((ea.n_int[:, 2] >= 0) & (ea.n_int[:, 2] < nz)).all()
    # j cycles 0..n_atoms-1
    assert set(ea.j.tolist()) == set(range(mat.num_atoms))


def test_strict_is_subset_prefix_of_halo():
    """halo must begin with the exact strict block (images appended after)."""
    mat = _material("snse2")
    strict = expand_cell(mat, 3, 3, 1, mode="strict")
    halo = expand_cell(mat, 3, 3, 1, mode="halo", bond_tolerance=1.2)
    n = len(strict)
    assert np.array_equal(halo.j[:n], strict.j)
    assert np.array_equal(halo.n_int[:n], strict.n_int)
    assert np.array_equal(halo.r_cart[:n], strict.r_cart)


# ---------------------------------------------------------------------------
# halo regression (the acceptance gate)
# ---------------------------------------------------------------------------
def _assert_halo_matches(ea, key):
    """Compare an expansion against the FROZEN 0.5.3 golden.

    Integer topology (n_int, j) is compared BITWISE — these carry the Bloch
    phase and must never drift. Positions are compared to atol 1e-9 Å:
    r_cart is matmul output, and BLAS builds legally differ in summation order
    (FMA/SIMD blocking), so bitwise float equality of matmul results is not a
    portable invariant — the geometry is. 1e-9 Å sits 8 orders above the
    measured cross-BLAS ulp noise (5.3e-17 Å between OpenBLAS and MKL on these
    fixtures) and 9 orders below the bond scale.
    """
    assert np.array_equal(ea.n_int, _GOLDEN[key + "__uc"])
    assert np.array_equal(ea.j, _GOLDEN[key + "__atom"])
    np.testing.assert_allclose(ea.r_cart, _GOLDEN[key + "__pos"],
                               rtol=0, atol=1e-9)


@pytest.mark.parametrize("golden_key", ["snse2_3x3x1", "wse2_3x3x1"])
def test_halo_matches_golden(golden_key):
    """halo output geometry-identical to the captured 0.5.3 golden to 1e-9 Å;
    integer topology bitwise."""
    name, (nx, ny, nz) = _HALO_CASES[golden_key]
    mat = _material(name)
    ea = expand_cell(mat, nx, ny, nz, mode="halo", bond_tolerance=1.2)
    _assert_halo_matches(ea, golden_key)


def test_halo_wse2_matches_golden():
    """As above for the WSe2 3x3x1 case (geometry-identical to 1e-9 Å)."""
    name, (nx, ny, nz) = _HALO_CASES["wse2_3x3x1"]
    mat = _material(name)
    ea = expand_cell(mat, nx, ny, nz, mode="halo", bond_tolerance=1.2)
    _assert_halo_matches(ea, "wse2_3x3x1")


def test_halo_comparator_negative_control():
    """The relaxed position tolerance must BITE: a 1e-6 Å perturbation of one
    coordinate (three orders above atol) has to fail the comparator. Guards
    against the tolerance ever being widened into vacuity."""
    import types
    name, (nx, ny, nz) = _HALO_CASES["snse2_3x3x1"]
    ea = expand_cell(_material(name), nx, ny, nz, mode="halo", bond_tolerance=1.2)
    bad = ea.r_cart.copy()
    bad[0, 0] += 1e-6
    fake = types.SimpleNamespace(n_int=ea.n_int, j=ea.j, r_cart=bad)
    with pytest.raises(AssertionError):
        _assert_halo_matches(fake, "snse2_3x3x1")


# ---------------------------------------------------------------------------
# n_int consistency (Bloch phase depends on this)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,nxyz", [
    ("snse2", (3, 3, 1)), ("wse2", (2, 2, 1)),
])
@pytest.mark.parametrize("mode", ["strict", "halo"])
def test_n_int_reconstructs_r_cart(name, nxyz, mode):
    """r_cart == (fractional_coords[j] + n_int) @ lattice within 1e-9 Angstrom.

    Convention: lattice rows are a1,a2,a3 and cartesian = fractional @ lattice
    (phonon_core.types.Material), so the reconstruction is right-multiplication
    by `lattice`, not `lattice @ ...`.
    """
    mat = _material(name)
    nx, ny, nz = nxyz
    kw = {"bond_tolerance": 1.2} if mode == "halo" else {}
    ea = expand_cell(mat, nx, ny, nz, mode=mode, **kw)
    frac = mat.fractional_coords[ea.j]                 # (N,3)
    recon = (frac + ea.n_int) @ mat.lattice            # (N,3)
    assert np.max(np.abs(recon - ea.r_cart)) < 1e-9


# ---------------------------------------------------------------------------
# position wrapping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["snse2", "wse2"])
def test_wrap_is_noop_on_fixtures(name):
    """Every shipped fixture is already in [0,1): wrapping must not move it."""
    mat = _material(name)
    assert mat.positions_were_wrapped is False


def test_wrap_handles_unwrapped_input():
    """Out-of-cell fractional coords get wrapped to [0,1) and the flag flips."""
    lat = np.eye(3) * 4.0
    mat = Material(
        lattice=lat,
        reciprocal_lattice=np.linalg.inv(lat).T,
        symbols=["C", "O"],
        masses=np.array([12.011, 15.999]),
        fractional_coords=np.array([[1.02, -0.3, 0.5],
                                    [0.0, 0.5, 2.25]]),
    )
    assert mat.positions_were_wrapped is True
    # 1.02 -> 0.02, -0.3 -> 0.7, 0.5 -> 0.5; 2.25 -> 0.25
    expected = np.array([[0.02, 0.7, 0.5], [0.0, 0.5, 0.25]])
    assert np.allclose(mat.fractional_coords, expected, atol=1e-12)
    # all wrapped coords lie in [0,1)
    assert (mat.fractional_coords >= 0.0).all()
    assert (mat.fractional_coords < 1.0).all()


def test_exact_integer_coords_unchanged():
    """0.0 stays 0.0 (floor(0)=0); only >=1 or <0 should move."""
    lat = np.eye(3) * 3.0
    mat = Material(lattice=lat, reciprocal_lattice=np.linalg.inv(lat).T,
                   symbols=["H"], masses=np.array([1.008]),
                   fractional_coords=np.array([[0.0, 0.5, 0.999999]]))
    assert mat.positions_were_wrapped is False


# ---------------------------------------------------------------------------
# window mode + scaffolding helpers
# ---------------------------------------------------------------------------
def test_window_mode_not_implemented():
    mat = _material("snse2")
    with pytest.raises(NotImplementedError):
        expand_cell(mat, 2, 2, 1, mode="window")


def test_unknown_mode_raises_value_error():
    mat = _material("snse2")
    with pytest.raises(ValueError):
        expand_cell(mat, 2, 2, 1, mode="bogus")


def test_halo_rejects_unexpected_kwargs():
    mat = _material("snse2")
    with pytest.raises(TypeError):
        expand_cell(mat, 2, 2, 1, mode="halo", bnd_tol=1.2)


def test_to_dict_from_dict_roundtrip():
    mat = _material("snse2")
    ea = expand_cell(mat, 3, 3, 1, mode="halo", bond_tolerance=1.2)
    d = ea.to_dict()
    assert isinstance(d["j"], list) and isinstance(d["n_int"], list)
    ea2 = ExpandedAtoms.from_dict(d)
    assert np.array_equal(ea.j, ea2.j)
    assert np.array_equal(ea.n_int, ea2.n_int)
    assert np.array_equal(ea.r_cart, ea2.r_cart)


def test_to_dict_is_json_serializable():
    """Scaffolding must survive a real JSON round-trip (web-viewer use)."""
    mat = _material("wse2")
    ea = expand_cell(mat, 2, 2, 1, mode="strict")
    s = json.dumps(ea.to_dict())
    ea2 = ExpandedAtoms.from_dict(json.loads(s))
    assert np.array_equal(ea.r_cart, ea2.r_cart)
