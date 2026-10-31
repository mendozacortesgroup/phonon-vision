"""phononwebsite interop tests (phonon-core 0.1.6).

Every format expectation here is traceable to the phononwebsite source now in
hand (cited in json_v1.py): field set from phonon.py::write_json; routing from
phononjson.js::getFromJson; vectors [Nq][Nb][Na][3][2] [Re,Im]; the
√(M_avg/m_s) mass scaling from phononyaml.js; raw mass-weighted e_s in internal
files. Importer fixture is a trimmed copy of the real
data/localdb/graphene/data.json.
"""
import json
import os

import numpy as np
import pytest

from phonon_core.types import Material, BandPath
from phonon_core.io import (
    PhononDocument, write_json_v1, read_json_v1,
    to_phononwebsite_json, from_phononwebsite_json,
)
from phonon_core.units import THZ_TO_CM1

_HERE = os.path.dirname(__file__)
_FX = os.path.join(_HERE, "fixtures", "json_v1")
_PW = os.path.join(_HERE, "fixtures", "phononwebsite")
_GRAPHENE = os.path.join(_PW, "graphene_data.json")

# Exact internal-format key set (phonon.py::write_json, lines 219-232) plus
# line_breaks and the avg-mass-normalized convention fields (average_mass,
# mode_amplitude_convention) that phononyaml.js sets (lines 291-292) — all
# read by getFromInternalJson; none triggers the pymatgen/OpenData branches.
_INTERNAL_KEYS = {
    "name", "natoms", "lattice", "atom_types", "atom_numbers", "formula",
    "qpoints", "repetitions", "atom_pos_car", "atom_pos_red", "eigenvalues",
    "distances", "highsym_qpts", "vectors", "line_breaks",
    "mode_amplitude_convention", "average_mass",
}


def _bandpath(name):
    d = np.load(os.path.join(_FX, name + ".npz"))
    mat = Material(d["lattice"], d["reciprocal_lattice"],
                   [str(s) for s in d["symbols"]], d["masses"],
                   d["fractional_coords"])
    return BandPath(mat, d["q_points"], d["distances"], d["frequencies"],
                    d["eig_real"] + 1j * d["eig_imag"],
                    [int(n) for n in d["segment_nqpoint"]],
                    [(str(a), str(b)) for a, b in d["labels"]])


def _graphene_bandpath():
    return from_phononwebsite_json(_GRAPHENE).band


# ---------------------------------------------------------------------------
# PART 1 — exporter conformance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["wse2", "snse2", "graphene"])
def test_export_emits_exactly_internal_keys(name):
    bp = _graphene_bandpath() if name == "graphene" else _bandpath(name)
    pw = to_phononwebsite_json(bp.material, bp)
    assert set(pw.keys()) == _INTERNAL_KEYS, set(pw.keys()) ^ _INTERNAL_KEYS
    # routing guards: must NOT trigger pymatgen or OpenData branches
    assert "@class" not in pw
    assert not all(k in pw for k in ("qpoints", "frequencies",
                                     "eigendisplacements", "structure"))


@pytest.mark.parametrize("name", ["wse2", "snse2", "graphene"])
def test_export_shapes_and_units(name):
    bp = _graphene_bandpath() if name == "graphene" else _bandpath(name)
    mat = bp.material
    pw = to_phononwebsite_json(mat, bp)
    nq, nb, na = bp.eigenvectors.shape[:3]
    assert np.array(pw["vectors"]).shape == (nq, nb, na, 3, 2)   # [Re,Im]
    assert np.array(pw["eigenvalues"]).shape == (nq, nb)
    assert np.array(pw["lattice"]).shape == (3, 3)
    # eigenvalues are cm⁻¹ (= THz × 33.35641)
    assert np.allclose(np.array(pw["eigenvalues"]), bp.frequencies * THZ_TO_CM1)
    # highsym_qpts is a list of [int, label]
    assert all(isinstance(p[0], int) and isinstance(p[1], str)
               for p in pw["highsym_qpts"])
    # atom counts consistent
    assert pw["natoms"] == na == len(pw["atom_types"]) == len(pw["atom_numbers"])


def test_avg_mass_normalized_convention_fields():
    """Export declares the avg-mass-normalized convention (phononyaml.js:291-292)
    and the flag is purely declarative — stored vectors are NOT re-scaled."""
    for name in ("wse2", "snse2", "graphene"):
        bp = _graphene_bandpath() if name == "graphene" else _bandpath(name)
        mat = bp.material
        pw = to_phononwebsite_json(mat, bp)
        # both fields present with correct values
        assert pw["mode_amplitude_convention"] == "avg-mass-normalized"
        assert np.isclose(pw["average_mass"], float(np.mean(mat.masses)))
        # stored vectors unchanged: exactly the √(M_avg/m_s) pre-scaled e_s,
        # i.e. the convention fields add no further per-atom scaling.
        m = np.asarray(mat.masses)
        scale = np.sqrt(m.mean() / m).reshape(1, 1, -1, 1)
        expected = np.stack([(bp.eigenvectors * scale).real,
                             (bp.eigenvectors * scale).imag], axis=-1)
        assert np.array_equal(np.array(pw["vectors"]), expected)


def test_convention_fields_do_not_break_internal_routing():
    """Adding the two fields must not trigger pymatgen/OpenData routing
    (phononjson.js::getFromJson) nor add a dynamical_matrix (which would enable
    the on-demand re-scaler)."""
    bp = _bandpath("snse2")
    pw = to_phononwebsite_json(bp.material, bp)
    assert "@class" not in pw
    assert not all(k in pw for k in ("qpoints", "frequencies",
                                     "eigendisplacements", "structure"))
    assert "dynamical_matrix" not in pw   # keeps applyModeAmplitude... a no-op


def test_export_json_serializable_and_routes_internal():
    bp = _bandpath("wse2")
    s = json.dumps(to_phononwebsite_json(bp.material, bp))
    d = json.loads(s)
    assert "@class" not in d and "eigendisplacements" not in d


# ---------------------------------------------------------------------------
# PART 1 — mass convention (the load-bearing physics check)
# ---------------------------------------------------------------------------
def test_mass_prescale_matches_displacement_up_to_mode_constant():
    """SnSe2 (multi-mass): exported per-atom relative amplitudes equal
    e_s/√m_s up to a per-mode constant — i.e. what our renderer displays and
    what phononwebsite's YAML path (√(M_avg/m_s) scaling) produces."""
    bp = _bandpath("snse2")
    mat = bp.material
    m = mat.masses
    assert len(set(np.round(m, 6))) > 1, "fixture must be multi-mass"

    pw = to_phononwebsite_json(mat, bp)
    V = np.array(pw["vectors"])
    export = V[..., 0] + 1j * V[..., 1]                  # (Nq,Nb,Na,3)
    displacement = bp.eigenvectors / np.sqrt(m).reshape(1, 1, -1, 1)  # e_s/√m_s

    # For every mode with motion, export == displacement * (single complex const).
    nq, nb = export.shape[:2]
    checked = 0
    for q in range(0, nq, 7):
        for n in range(nb):
            flat_e = export[q, n].reshape(-1)
            flat_d = displacement[q, n].reshape(-1)
            big = np.abs(flat_d) > 1e-10
            if big.sum() < 2:
                continue
            ratio = flat_e[big] / flat_d[big]
            # all ratios equal (constant) → relative amplitudes identical
            assert np.std(np.abs(ratio)) / np.mean(np.abs(ratio)) < 1e-9
            checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# PART 2 — importer
# ---------------------------------------------------------------------------
def test_import_internal_graphene():
    doc = from_phononwebsite_json(_GRAPHENE)
    raw = json.load(open(_GRAPHENE))
    mat, band = doc.material, doc.band
    assert mat.symbols == ["C", "C"]
    # masses from phononwebsite's atomic_data.js table (C = 12.0107)
    assert np.allclose(mat.masses, 12.0107)
    nq = len(raw["qpoints"])
    nb = len(raw["eigenvalues"][0])
    na = raw["natoms"]
    assert band.q_points.shape == (nq, 3)
    assert band.frequencies.shape == (nq, nb)
    # eigenvalues cm⁻¹ -> THz
    assert np.allclose(band.frequencies, np.array(raw["eigenvalues"]) / THZ_TO_CM1)
    # vectors -> complex (Nq,Nb,Na,3)
    assert band.eigenvectors.shape == (nq, nb, na, 3)
    assert np.iscomplexobj(band.eigenvectors)
    V = np.array(raw["vectors"])
    assert np.array_equal(band.eigenvectors.real, V[..., 0])
    assert np.array_equal(band.eigenvectors.imag, V[..., 1])
    # qpoints stay fractional (unchanged)
    assert np.allclose(band.q_points, raw["qpoints"])
    # segments are valid (sum to nq)
    assert sum(band.segment_nqpoint) == nq


def test_import_internal_vectors_stored_raw():
    """Internal files carry raw mass-weighted e_s; importer stores them as-is
    (no √m scaling) — our renderer applies 1/√m_s."""
    raw = json.load(open(_GRAPHENE))
    band = from_phononwebsite_json(raw).band
    V = np.array(raw["vectors"])
    assert np.array_equal(band.eigenvectors.real, V[..., 0])


def test_import_rejects_pymatgen_and_opendata_as_followup():
    with pytest.raises(NotImplementedError, match="pymatgen"):
        from_phononwebsite_json({"@class": "PhononBandStructureSymmLine"})
    opendata = {"qpoints": [], "frequencies": [], "eigendisplacements": [],
                "structure": {}}
    with pytest.raises(NotImplementedError, match="opendata"):
        from_phononwebsite_json(opendata)


# ---------------------------------------------------------------------------
# PART 3 — material_id round-trips as uninterpreted metadata
# ---------------------------------------------------------------------------
def test_material_id_metadata_roundtrip(tmp_path):
    bp = _bandpath("wse2")
    doc = PhononDocument.from_bandpath(bp, metadata={"material_id": "mp-1234"})
    path = str(tmp_path / "with_id.json")
    write_json_v1(doc, path)
    back = read_json_v1(path)
    assert back.metadata["material_id"] == "mp-1234"   # verbatim, uninterpreted
