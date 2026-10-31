"""Tests for phonon_core.io.json_v1 (JSON v1 base layer, Phase A).

The load-bearing guarantee is LOSSLESS round-trip: write_json_v1 → read_json_v1
preserves ints/strings/labels exactly, floats within 1e-12, and eigenvector
real/imag arrays **bitwise**. Round-trip fixtures (WSe2, SnSe2) are
cached parsed BandPaths in fixtures/json_v1/*.npz (the phonopy band.yaml files
are 3–35 MB and too large to vendor); the cache preserves float64 bits, so it
is equivalent to a fresh phonopy parse for the purpose of testing the JSON
write→read round-trip.
"""
import json
import os

import numpy as np
import pytest

from phonon_core.types import Material, BandPath, MeshGrid
from phonon_core.io import (
    PhononDocument, EngineProvenance, write_json_v1, read_json_v1,
    to_phononwebsite_json, JsonV1Error, SCHEMA_VERSION,
)
from phonon_core.io.json_v1 import build_document_dict

_HERE = os.path.dirname(__file__)
_FX = os.path.join(_HERE, "fixtures", "json_v1")
_SCHEMA_PATH = os.path.join(_HERE, "..", "docs", "json_v1_schema.json")

jsonschema = pytest.importorskip("jsonschema")  # test-only dependency

_FIXTURES = ["wse2", "snse2"]


def _bandpath_from_npz(name: str) -> BandPath:
    d = np.load(os.path.join(_FX, name + ".npz"), allow_pickle=False)
    mat = Material(
        lattice=d["lattice"],
        reciprocal_lattice=d["reciprocal_lattice"],
        symbols=[str(s) for s in d["symbols"]],
        masses=d["masses"],
        fractional_coords=d["fractional_coords"],
    )
    mat.positions_were_wrapped = bool(d["positions_were_wrapped"])
    return BandPath(
        material=mat,
        q_points=d["q_points"],
        distances=d["distances"],
        frequencies=d["frequencies"],
        eigenvectors=d["eig_real"] + 1j * d["eig_imag"],
        segment_nqpoint=[int(n) for n in d["segment_nqpoint"]],
        labels=[(str(a), str(b)) for a, b in d["labels"]],
    )


# ---------------------------------------------------------------------------
# lossless round-trip — the acceptance gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", _FIXTURES)
def test_roundtrip_lossless(name, tmp_path):
    bp = _bandpath_from_npz(name)
    doc = PhononDocument.from_bandpath(bp, source_code="phonopy",
                                       source_version="test")
    path = str(tmp_path / f"{name}.json")
    write_json_v1(doc, path)
    back = read_json_v1(path)

    m0, m1 = bp.material, back.material
    b1 = back.band
    # ints / strings / labels EXACT
    assert list(m0.symbols) == list(m1.symbols)
    assert list(bp.labels) == list(b1.labels)
    assert list(bp.segment_nqpoint) == list(b1.segment_nqpoint)
    assert m0.positions_were_wrapped == m1.positions_were_wrapped
    # floats within 1e-12 (lattice, masses, fractional, q-points, distances, freqs)
    assert np.max(np.abs(m0.lattice - m1.lattice)) < 1e-12
    assert np.max(np.abs(m0.masses - m1.masses)) < 1e-12
    assert np.max(np.abs(m0.fractional_coords - m1.fractional_coords)) < 1e-12
    assert np.max(np.abs(bp.q_points - b1.q_points)) < 1e-12
    assert np.max(np.abs(bp.distances - b1.distances)) < 1e-12
    assert np.max(np.abs(bp.frequencies - b1.frequencies)) < 1e-12
    # eigenvectors real/imag BITWISE identical
    assert np.array_equal(bp.eigenvectors.real, b1.eigenvectors.real)
    assert np.array_equal(bp.eigenvectors.imag, b1.eigenvectors.imag)
    # complex array as a whole, bitwise
    assert np.array_equal(bp.eigenvectors, b1.eigenvectors)


def test_roundtrip_mesh_lossless(tmp_path):
    """MeshGrid path (with weights + optional eigenvectors), bitwise."""
    bp = _bandpath_from_npz("wse2")
    mat = bp.material
    nq, nb, na = 8, mat.num_bands, mat.num_atoms
    rng = np.random.default_rng(0)
    ev = rng.standard_normal((nq, nb, na, 3)) + 1j * rng.standard_normal((nq, nb, na, 3))
    mg = MeshGrid(
        material=mat, mesh=(2, 2, 2),
        q_points=rng.standard_normal((nq, 3)),
        frequencies=rng.standard_normal((nq, nb)),
        eigenvectors=ev,
        weights=np.ones(nq, dtype=np.int64),
    )
    doc = PhononDocument.from_meshgrid(mg)
    path = str(tmp_path / "mesh.json")
    write_json_v1(doc, path)
    back = read_json_v1(path)
    assert back.mesh is not None and back.band is None
    assert tuple(back.mesh.mesh) == (2, 2, 2)
    assert np.array_equal(mg.weights, back.mesh.weights)
    assert np.array_equal(mg.eigenvectors.real, back.mesh.eigenvectors.real)
    assert np.array_equal(mg.eigenvectors.imag, back.mesh.eigenvectors.imag)


# ---------------------------------------------------------------------------
# versioning + conventions
# ---------------------------------------------------------------------------
def _minimal_doc_dict(tmp_path):
    bp = _bandpath_from_npz("wse2")
    return build_document_dict(PhononDocument.from_bandpath(bp))


def test_schema_version_is_1_1_0(tmp_path):
    # 1.1.0 at 0.6.79: the ADDITIVE extensions.engine provenance block
    # (Level-2 Stage 2 Bump 1); the v1.0 core is unchanged.
    d = _minimal_doc_dict(tmp_path)
    assert d["schema_version"] == "1.1.0" == SCHEMA_VERSION


def test_conventions_present_and_complete(tmp_path):
    d = _minimal_doc_dict(tmp_path)
    conv = d["conventions"]
    for key in ("frequency_units", "eigenvector_normalization", "q_point_format",
                "lattice_vectors", "dynamical_matrix_fourier", "nac_method",
                "source_code", "source_version"):
        assert key in conv
    assert conv["frequency_units"] == "THz"
    assert conv["eigenvector_normalization"] == "mass_weighted"
    assert conv["lattice_vectors"] == "rows"
    assert conv["q_point_format"] == "fractional"
    assert d["generator"] == {"name": "phonon-core", "version": __import__("phonon_core").__version__}


def test_missing_conventions_block_fails(tmp_path):
    d = _minimal_doc_dict(tmp_path)
    del d["conventions"]
    path = tmp_path / "no_conv.json"
    path.write_text(json.dumps(d))
    with pytest.raises(JsonV1Error, match="conventions"):
        read_json_v1(str(path))


def test_incomplete_conventions_block_fails(tmp_path):
    d = _minimal_doc_dict(tmp_path)
    del d["conventions"]["nac_method"]
    path = tmp_path / "bad_conv.json"
    path.write_text(json.dumps(d))
    with pytest.raises(JsonV1Error, match="missing required keys"):
        read_json_v1(str(path))


def test_unknown_minor_version_reads_ok(tmp_path):
    d = _minimal_doc_dict(tmp_path)
    d["schema_version"] = "1.9.0"   # unknown MINOR within MAJOR 1
    path = tmp_path / "minor.json"
    path.write_text(json.dumps(d))
    doc = read_json_v1(str(path))           # must NOT raise
    assert doc.band is not None


def test_unknown_major_version_rejected(tmp_path):
    d = _minimal_doc_dict(tmp_path)
    d["schema_version"] = "2.0.0"   # unknown MAJOR
    path = tmp_path / "major.json"
    path.write_text(json.dumps(d))
    with pytest.raises(JsonV1Error, match="major"):
        read_json_v1(str(path))


def test_missing_schema_version_rejected(tmp_path):
    d = _minimal_doc_dict(tmp_path)
    del d["schema_version"]
    path = tmp_path / "nover.json"
    path.write_text(json.dumps(d))
    with pytest.raises(JsonV1Error, match="schema_version"):
        read_json_v1(str(path))


def test_sidecar_storage_rejected(tmp_path):
    d = _minimal_doc_dict(tmp_path)
    d["core"]["band"]["storage"] = "sidecar"
    path = tmp_path / "sidecar.json"
    path.write_text(json.dumps(d))
    with pytest.raises(JsonV1Error, match="sidecar"):
        read_json_v1(str(path))


# ---------------------------------------------------------------------------
# phononwebsite export
# ---------------------------------------------------------------------------
def test_to_phononwebsite_keys_and_shape():
    bp = _bandpath_from_npz("wse2")
    pw = to_phononwebsite_json(bp.material, bp)
    required = ["name", "natoms", "lattice", "atom_types", "atom_numbers",
                "formula", "repetitions", "atom_pos_car", "atom_pos_red",
                "qpoints", "distances", "eigenvalues", "vectors",
                "highsym_qpts", "line_breaks"]
    for k in required:
        assert k in pw, f"phononwebsite export missing {k!r}"
    nq, nb, na = bp.eigenvectors.shape[:3]
    # vectors: [n_q][n_branch][n_atom][3][2]  (last axis = [Re, Im])
    v = np.array(pw["vectors"])
    assert v.shape == (nq, nb, na, 3, 2)
    # eigenvalues: [n_q][n_branch], converted to cm^-1
    ev = np.array(pw["eigenvalues"])
    assert ev.shape == (nq, nb)
    # cm^-1 magnitudes are ~33x the THz values
    assert np.max(np.abs(ev)) > np.max(np.abs(bp.frequencies))
    # atom_numbers correct (W=74, Se=34)
    assert pw["atom_numbers"] == [74, 34, 34]
    assert pw["natoms"] == 3
    # highsym_qpts is a list of [index, label] pairs
    assert all(len(pair) == 2 and isinstance(pair[0], int) for pair in pw["highsym_qpts"])


def test_to_phononwebsite_is_json_serializable():
    bp = _bandpath_from_npz("snse2")
    s = json.dumps(to_phononwebsite_json(bp.material, bp))
    assert json.loads(s)["formula"] == "SnSe2"


# ---------------------------------------------------------------------------
# JSON Schema validation
# ---------------------------------------------------------------------------
def _schema():
    with open(_SCHEMA_PATH) as f:
        return json.load(f)


@pytest.mark.parametrize("name", _FIXTURES)
def test_written_file_validates_against_schema(name, tmp_path):
    bp = _bandpath_from_npz(name)
    doc = PhononDocument.from_bandpath(bp)
    path = str(tmp_path / f"{name}.json")
    write_json_v1(doc, path)
    with open(path) as f:
        instance = json.load(f)
    jsonschema.validate(instance=instance, schema=_schema())   # raises on failure


def test_schema_rejects_missing_conventions(tmp_path):
    d = _minimal_doc_dict(tmp_path)
    del d["conventions"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=d, schema=_schema())


def test_schema_allows_extensions():
    """extensions namespace must be permitted (forward-compat)."""
    bp = _bandpath_from_npz("wse2")
    d = build_document_dict(PhononDocument.from_bandpath(bp))
    d["extensions"] = {"group_velocities": {"anything": [1, 2, 3]}}
    jsonschema.validate(instance=d, schema=_schema())   # must NOT raise


# ---------------------------------------------------------------------------
# extensions.engine — evaluation provenance (schema 1.1.0, 0.6.79)
# ---------------------------------------------------------------------------
_ENGINE_FULL = EngineProvenance(
    kind="phonopy-A1", evaluated_band=True, evaluated_mesh=True,
    fc_md5="ab" * 16)


def _engine_doc_path(tmp_path, engine, fname="engine.json"):
    bp = _bandpath_from_npz("wse2")
    doc = PhononDocument.from_bandpath(bp, engine=engine)
    path = str(tmp_path / fname)
    write_json_v1(doc, path)
    return path


def test_engine_roundtrip_and_absent(tmp_path):
    # (a) WITH provenance: write -> read -> equal field-by-field.
    path = _engine_doc_path(tmp_path, _ENGINE_FULL)
    back = read_json_v1(path)
    assert back.engine is not None
    assert back.engine == _ENGINE_FULL           # frozen dataclass equality
    assert back.engine.kind == "phonopy-A1"
    assert back.engine.evaluated_band is True
    assert back.engine.evaluated_mesh is True
    assert back.engine.fc_md5 == "ab" * 16
    assert back.engine.phonopy_version is None
    assert back.engine.phonopy_yaml_md5 is None
    # ... and the wire form is the documented one.
    with open(path) as f:
        wire = json.load(f)["extensions"]["engine"]
    assert wire == {"kind": "phonopy-A1",
                    "evaluated": {"band": True, "mesh": True},
                    "phonopy_version": None,
                    "fc_md5": "ab" * 16,
                    "phonopy_yaml_md5": None}
    # WITHOUT provenance: written dict keeps the 1.0.0 shape (extensions {})
    # and reads back engine None.
    bp = _bandpath_from_npz("wse2")
    d = build_document_dict(PhononDocument.from_bandpath(bp))
    assert d["extensions"] == {}
    p2 = str(tmp_path / "noeng.json")
    write_json_v1(PhononDocument.from_bandpath(bp), p2)
    assert read_json_v1(p2).engine is None


def test_engine_document_validates_against_schema(tmp_path):
    # the code's wire output and docs/json_v1_schema.json must agree.
    path = _engine_doc_path(tmp_path, _ENGINE_FULL)
    with open(path) as f:
        instance = json.load(f)
    jsonschema.validate(instance=instance, schema=_schema())   # must NOT raise


def test_engine_validation_bites(tmp_path):
    # (b) STRICT on required keys/types; every error names the offending key.
    path = _engine_doc_path(tmp_path, _ENGINE_FULL)
    with open(path) as f:
        good = json.load(f)

    def _write_variant(mutate, fname):
        d = json.loads(json.dumps(good))    # deep copy
        mutate(d["extensions"]["engine"])
        p = str(tmp_path / fname)
        with open(p, "w") as f:
            json.dump(d, f)
        return p

    p = _write_variant(lambda e: e.pop("kind"), "nokind.json")
    with pytest.raises(JsonV1Error, match="engine.kind"):
        read_json_v1(p)
    p = _write_variant(lambda e: e["evaluated"].__setitem__("band", "yes"),
                       "strband.json")
    with pytest.raises(JsonV1Error, match="evaluated.band"):
        read_json_v1(p)
    p = _write_variant(lambda e: e.__setitem__("fc_md5", "a" * 31),
                       "shortmd5.json")
    with pytest.raises(JsonV1Error, match="fc_md5"):
        read_json_v1(p)


def test_engine_minor_compat_unknown_key_inside(tmp_path):
    # (c) a 1.2.0-style file that GROWS the engine block must not break this
    # reader: unknown extra keys inside engine are ignored, provenance intact.
    path = _engine_doc_path(tmp_path, _ENGINE_FULL)
    with open(path) as f:
        d = json.load(f)
    d["extensions"]["engine"]["future_field"] = 1
    p = str(tmp_path / "future.json")
    with open(p, "w") as f:
        json.dump(d, f)
    back = read_json_v1(p)
    assert back.engine == _ENGINE_FULL


def test_engine_absent_with_other_extensions(tmp_path):
    # (d) legacy: extensions carrying OTHER keys but no engine reads with
    # engine None — the untouched v1.0 ignore-unknown-keys contract.
    bp = _bandpath_from_npz("wse2")
    d = build_document_dict(PhononDocument.from_bandpath(bp))
    d["extensions"] = {"group_velocities": {"anything": [1, 2, 3]}}
    p = str(tmp_path / "legacyext.json")
    with open(p, "w") as f:
        json.dump(d, f)
    back = read_json_v1(p)
    assert back.engine is None
    assert back.band is not None
