"""Tests for convert_qe_dyn — the .dyn → mesh_modes payload route.

Pins are measured truth, not hope: the CsSnI3 numbers asserted here were
computed first and pinned second, and every symmetry assertion is a fact of
the tetragonal crystal that the pipeline has no way to fake — if the
eigenvector layout, the mass mapping or the grid weighting were wrong, the
site symmetries of the U tensors would break before anything else.
"""

import copy
import glob
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phonon_core.adp import adp_tensors, u_isotropic          # noqa: E402
from phonon_core.qe_import import convert_qe_dyn, read_qe_dyn_set  # noqa: E402

REAL_DIR = os.environ.get("PV_QE_DYN_FIXTURES", os.path.join(
    os.path.dirname(__file__), "fixtures", "qe", "real", "CsSnI3_epirelax"))


def real_paths():
    if not os.path.isdir(REAL_DIR):
        return []
    return sorted(glob.glob(os.path.join(REAL_DIR, "CsSnI3.dyn*")))


@pytest.fixture(scope="module")
def real_set():
    paths = real_paths()
    if not paths:
        pytest.skip("real ph.x .dyn output not present")
    return read_qe_dyn_set(paths)


@pytest.fixture(scope="module")
def payload_info(real_set):
    return convert_qe_dyn(real_set)


# ----------------------------------------------------------------- structure

def test_payload_shape_and_oracle(payload_info):
    payload, info = payload_info
    assert sorted(payload.keys()) == ["band", "mesh_modes", "meta"]
    assert "surface" not in payload and "dos" not in payload
    assert info["n_q"] == 64 and info["grid"] == [4, 4, 4]
    assert info["n_bands"] == 15 and info["polar"] is True
    # the ph.x-printed frequencies survive the whole conversion
    assert info["freq_oracle_thz"] < 2e-6
    src = payload["meta"]["source"]
    assert src["origin"] == "qe-dyn" and src["arbitrary_q"] is False
    assert payload["meta"]["material"]["dimensionality"] == 3


def test_band_block_is_an_honest_spectrum(payload_info):
    payload, _ = payload_info
    band = payload["band"]
    dist = np.array(band["distance"])
    q_cart = np.array(band["q_cart"])
    # x-axis IS |q|, ascending — no invented path ordering
    assert np.allclose(dist, np.linalg.norm(q_cart, axis=1), atol=1e-12)
    assert np.all(np.diff(dist) >= -1e-12)
    assert dist[0] == 0.0, "Gamma present and first"
    assert band["breaks"] == [] and band["high_symmetry_points"] == []
    assert band["labels"] == [["Γ", ""]]
    char = np.array(band["character"])
    assert char.min() >= -1e-9 and char.max() <= 1 + 1e-9


def test_character_partitions_over_species(real_set):
    """group-A weights over the full species partition must sum to 1 per
    mode (Dove normalization) — checks the |e|^2 bookkeeping end to end."""
    tot = None
    for sym in ("Sn", "Cs", "I"):
        payload, _ = convert_qe_dyn(real_set, group_a=sym)
        c = np.array(payload["band"]["character"])
        tot = c if tot is None else tot + c
    assert np.allclose(tot, 1.0, atol=1e-9)


def test_eigenvector_fidelity(payload_info):
    payload, _ = payload_info
    band_e = (np.array(payload["band"]["eigenvectors_real"])
              + 1j * np.array(payload["band"]["eigenvectors_imag"]))
    norms = (np.abs(band_e) ** 2).sum(axis=(2, 3))
    assert np.allclose(norms, 1.0, atol=1e-12), "band eigvecs full precision"
    mesh_e = (np.array(payload["mesh_modes"]["eigenvectors_real"])
              + 1j * np.array(payload["mesh_modes"]["eigenvectors_imag"]))
    # Mesh copies are 6-significant-figure rounded: with ~75 components per
    # mode that puts the overlap within ~1e-6 of unity, not 1e-8 (a first
    # version demanded 1e-8 — wrong arithmetic in the TEST; measured floor
    # 0.9999989, and values can exceed 1 slightly because the rounded copy
    # is not renormalized).
    ov = np.abs(np.einsum("qbad,qbad->qb", band_e.conj(), mesh_e))
    assert np.abs(ov - 1.0).max() < 1e-5, f"mesh/band overlap dev {np.abs(ov-1).max()}"
    # and the two blocks agree on frequencies bitwise (both full precision)
    assert payload["band"]["frequency"] == payload["mesh_modes"]["frequency"]


def test_conversion_is_deterministic(real_set):
    p1, _ = convert_qe_dyn(real_set)
    p2, _ = convert_qe_dyn(real_set)
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)


# ------------------------------------------------- the equal-weight licence

def _surgery(real_set, drop=None, dup=None):
    d = copy.copy(real_set)
    idx = list(range(len(real_set.qpoints_frac)))
    if drop is not None:
        idx.remove(drop)
    if dup is not None:
        idx.append(dup)
    d.qpoints_frac = real_set.qpoints_frac[idx]
    d.qpoints_cart = real_set.qpoints_cart[idx]
    d.dynmat = real_set.dynmat[idx]
    d.file_frequencies = None
    d.frequency_index = None
    return d


def test_incomplete_star_is_refused(real_set):
    with pytest.raises(ValueError, match="incomplete|fill"):
        convert_qe_dyn(_surgery(real_set, drop=10))


def test_duplicate_q_is_refused(real_set):
    with pytest.raises(ValueError, match="duplicate"):
        convert_qe_dyn(_surgery(real_set, drop=10, dup=20))


def test_off_grid_q_is_refused(real_set):
    d = _surgery(real_set)
    d.qpoints_frac = d.qpoints_frac.copy()
    d.qpoints_frac[3, 0] += 0.01
    with pytest.raises(ValueError, match="fill|commensurate|not on the"):
        convert_qe_dyn(d)


def test_unknown_group_a_is_refused(real_set):
    with pytest.raises(ValueError, match="group_a"):
        convert_qe_dyn(real_set, group_a="Xe")


# ------------------------------------------------------- the physics payoff

def test_adp_from_the_payload_reproduces_perovskite_physics(payload_info):
    """The point of the route: thermal ellipsoids for a 3D polar material.
    Every assertion is a symmetry or physics fact:

    * tetragonal site symmetry — Sn and Cs sit on 4/mmm-type sites, so
      Uxx = Uyy exactly; the apical iodine likewise;
    * the two in-plane iodines are mapped onto each other by the x<->y
      symmetry, so I1.Uxx = I2.Uyy and their U_iso are equal;
    * Cs is the A-site RATTLER of the halide perovskite: its U_iso dwarfs
      the framework's (measured 0.72 A^2 vs Sn 0.032 at 300 K — 23x).
    """
    payload, _ = payload_info
    mm = payload["mesh_modes"]
    syms = payload["meta"]["material"]["symbols"]
    masses = np.array(payload["meta"]["material"]["masses"])
    U, info = adp_tensors(np.array(mm["frequency"]),
                          np.array(mm["eigenvectors_real"]),
                          np.array(mm["eigenvectors_imag"]),
                          masses, 300.0)
    assert info["n_q"] == 64

    iso = u_isotropic(U)
    by = {s: [] for s in set(syms)}
    for s, u, t in zip(syms, iso, U):
        by[s].append((u, t))

    # tetragonal site symmetry: x and y equivalent for Sn, Cs, apical I
    for name, (u, t) in (("Sn", by["Sn"][0]), ("Cs", by["Cs"][0])):
        assert abs(t[0, 0] - t[1, 1]) < 1e-6, f"{name}: Uxx != Uyy"
    # iodines: two in-plane (mirror pair) + one apical (Uxx == Uyy)
    i_t = [t for _, t in by["I"]]
    aniso = [abs(t[0, 0] - t[1, 1]) for t in i_t]
    apical = i_t[int(np.argmin(aniso))]
    plane = [t for k, t in enumerate(i_t) if k != int(np.argmin(aniso))]
    assert abs(apical[0, 0] - apical[1, 1]) < 1e-6
    assert len(plane) == 2
    assert abs(plane[0][0, 0] - plane[1][1, 1]) < 1e-6, "in-plane mirror pair"
    assert abs(plane[0][1, 1] - plane[1][0, 0]) < 1e-6

    # the rattler
    u_cs = by["Cs"][0][0]
    u_sn = by["Sn"][0][0]
    assert u_cs == max(iso), "Cs must carry the largest thermal ellipsoid"
    assert u_cs > 10 * u_sn, f"rattler contrast lost: Cs {u_cs} vs Sn {u_sn}"
    # magnitudes on the physical scale (soft, near-unstable perovskite: large)
    assert 0.01 < u_sn < 0.1 and 0.3 < u_cs < 1.5

    # Stability characterization, pinned to what the data MEASURES rather
    # than what was hoped: this geometry is genuinely soft — an entire branch
    # is unstable across much of the zone, 63 of 960 modes (6.6%) below
    # -floor. (A first version asserted <= 1%, extrapolating from the Gamma
    # row's 3 negatives; the zone disagreed.) The ADP treats those via |nu|,
    # so for THIS material the tensors are qualitative — anisotropy and the
    # rattler contrast are robust, absolute magnitudes are not. The pin keeps
    # the count characteristic: nonzero (the softness is real) and a
    # minority (the crystal is not simply melted).
    f = np.array(mm["frequency"])
    n_neg = int((f < -info["freq_floor_thz"]).sum())
    assert 0 < n_neg < 0.10 * f.size, f"{n_neg} strongly negative modes"


def test_written_js_is_loadable(real_set, tmp_path):
    out = str(tmp_path / "navigator_data.js")
    _, info = convert_qe_dyn(real_set, output=out)
    text = open(out).read()
    assert text.startswith("window.NAV_DATA = ")
    d = json.loads(text[len("window.NAV_DATA = "):].rstrip().rstrip(";"))
    assert sorted(d.keys()) == ["band", "mesh_modes", "meta"]
    assert info["output"] == out
    # raw-JSON flavour too
    out2 = str(tmp_path / "payload.json")
    convert_qe_dyn(real_set, output=out2)
    json.load(open(out2))
