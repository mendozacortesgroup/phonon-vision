"""Web bundle tests (Deliverable 3): the single artifact round-trips — load the
bundle, recompute displacements via phonon_core kinematics, check against the
bundled golden vectors (three tiers + displacement frames) → all pass.
"""
import json
import os

import numpy as np
import pytest

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from phonon_core import load_band_hdf5
from phonon_core.io import (
    PhononDocument, SCHEMA_VERSION, write_web_bundle, verify_web_bundle)

_HERE = os.path.dirname(__file__)
_BAND = os.path.join(_HERE, "fixtures", "phonopy", "SnSe2_band.hdf5")
_SCHEMA = os.path.join(_HERE, os.pardir, "docs", "json_v1_schema.json")

jsonschema = pytest.importorskip("jsonschema")


def _doc():
    bp = load_band_hdf5(_BAND)
    return PhononDocument(material=bp.material, band=bp, source_version="2.x")


def test_bundle_layout_and_manifest(tmp_path):
    out = str(tmp_path / "bundle")
    manifest = write_web_bundle(_doc(), out)
    for fn in ("document.json", "golden_kinematics.npz",
               "golden_kinematics.json", "bundle.json"):
        assert os.path.isfile(os.path.join(out, fn)), fn
    # single-sourced from json_v1 (web_bundle stamps SCHEMA_VERSION), so the
    # test tracks the constant — a literal pin re-arms the stale-pin trap that
    # required this amendment at 0.6.79 (architect-logged scope miss).
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["n_bundled_modes"] >= 3
    assert manifest["has_finite_q"]


def test_bundle_document_validates_against_schema(tmp_path):
    out = str(tmp_path / "bundle")
    write_web_bundle(_doc(), out)
    with open(os.path.join(out, "document.json")) as f:
        instance = json.load(f)
    with open(_SCHEMA) as f:
        schema = json.load(f)
    jsonschema.validate(instance=instance, schema=schema)   # raises on failure


def test_bundle_roundtrips_all_tiers_and_displacements(tmp_path):
    """The cross-language contract in Python: recompute displacements from the
    bundled document and check against the bundled golden vectors."""
    out = str(tmp_path / "bundle")
    write_web_bundle(_doc(), out)
    chk = verify_web_bundle(out)
    assert chk.passed, str(chk.conformance)
    assert chk.conformance.tier1_frequencies.passed
    assert chk.conformance.tier2_magnitude.passed
    assert chk.conformance.tier3_complex_aligned.passed
    assert chk.conformance.has_finite_q
    # displacement frames reproduce at the float floor
    assert chk.max_disp_dev < 1e-9, chk.max_disp_dev


def test_bundle_golden_format_matches_0_2_0(tmp_path):
    """The bundled golden files use the 0.2.0 golden-vector format (so the same
    tooling reads them): material arrays, slot arrays, per-mode eigvec/disp."""
    out = str(tmp_path / "bundle")
    write_web_bundle(_doc(), out)
    npz = np.load(os.path.join(out, "golden_kinematics.npz"))
    keys = set(npz.files)
    for k in ("_material_lattice", "_material_masses", "_slot_j", "_slot_n_int"):
        assert k in keys
    side = json.load(open(os.path.join(out, "golden_kinematics.json")))
    assert set(side["inputs"]) == {"amplitude", "phi", "supercell",
                                   "n_frames", "mass_weighted"}
    a_mode = next(iter(side["modes"]))
    for suf in ("eigvec_raw", "eigvec_canon", "bloch", "disp_phi", "disp_traj"):
        assert f"{a_mode}__{suf}" in keys
