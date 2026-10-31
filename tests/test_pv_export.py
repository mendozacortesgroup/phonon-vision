"""pv_export CLI tests (Deliverable 1), against committed fixtures so a clean
`pytest` runs them with no external setup.

- band-only export -> JSON that passes the schema validator AND the conformance
  gate at Γ and finite q;
- band + mesh (same material) -> one merged document with both blocks;
- auto-detection of band vs mesh from real phonopy file shape;
- a genuinely inconsistent band/mesh pair is refused.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

_HERE = os.path.dirname(__file__)
_FIX = os.path.join(_HERE, "fixtures", "phonopy")
_BAND = os.path.join(_FIX, "SnSe2_band.hdf5")          # SnSe2, 909-pt Γ-M-K-Γ
_MESH = os.path.join(_FIX, "vasp_SnSe2_mesh.yaml")     # same material (subset)
_SCHEMA = os.path.join(_HERE, os.pardir, "docs", "json_v1_schema.json")

jsonschema = pytest.importorskip("jsonschema")

# import the CLI module by path
_spec = importlib.util.spec_from_file_location(
    "pv_export", os.path.join(_HERE, os.pardir, "tools", "pv_export.py"))
pv_export = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pv_export)


def _schema():
    with open(_SCHEMA) as f:
        return json.load(f)


def test_detect_kind_from_real_files():
    assert pv_export.detect_kind(_BAND) == "band"
    assert pv_export.detect_kind(_MESH) == "mesh"


def test_band_only_export(tmp_path):
    out = str(tmp_path / "band.json")
    rc = pv_export.main([_BAND, "-o", out, "--source-version", "2.x"])
    assert rc == 0
    d = json.load(open(out))
    jsonschema.validate(instance=d, schema=_schema())   # schema OK
    assert d["core"]["band"] is not None
    assert d["core"]["mesh"] is None
    assert d["conventions"]["source_version"] == "2.x"


def test_band_plus_mesh_merges_one_document(tmp_path):
    out = str(tmp_path / "merged.json")
    rc = pv_export.main([_BAND, _MESH, "-o", out])
    assert rc == 0
    d = json.load(open(out))
    jsonschema.validate(instance=d, schema=_schema())
    assert d["core"]["band"] is not None and d["core"]["mesh"] is not None
    # single shared material, serialized once
    assert d["core"]["material"]["symbols"] == ["Sn", "Se", "Se"]


def test_no_validate_skips_schema(tmp_path):
    out = str(tmp_path / "b.json")
    assert pv_export.main([_BAND, "-o", out, "--no-validate", "--no-conformance"]) == 0
    assert os.path.isfile(out)


def test_inconsistent_material_is_refused(tmp_path):
    """A band + a mesh whose masses differ must be refused (exit 2), not merged."""
    # make a mesh yaml copy with a perturbed mass
    bad = tmp_path / "bad_mesh.yaml"
    with open(_MESH) as f:
        text = f.read()
    text = text.replace("mass: 118.710000", "mass: 200.000000", 1)
    bad.write_text(text)
    with pytest.raises(SystemExit) as ei:
        pv_export.main([_BAND, str(bad), "-o", str(tmp_path / "x.json")])
    assert ei.value.code == 2


def test_bundle_emitted_and_listed(tmp_path):
    out = str(tmp_path / "doc.json")
    bdir = str(tmp_path / "bundle")
    rc = pv_export.main([_BAND, "-o", out, "--bundle", bdir])
    assert rc == 0
    for fn in ("document.json", "golden_kinematics.npz",
               "golden_kinematics.json", "bundle.json"):
        assert os.path.isfile(os.path.join(bdir, fn)), fn
