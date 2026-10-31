"""tests/emit_web_golden.py — the web-consumable golden re-emission (0.6.5).

Proves it is a faithful re-emission of the FROZEN fixtures, not a second physics
path: disp_phi is the SAME numbers as the npz, and the eigenvectors decode to
canonicalize(eigvec_raw). Also guards that the script never touches the uploads
area. test_kinematics_golden.py is untouched and remains the source of truth for
the kinematics themselves.
"""
import importlib.util
import json
import os

import numpy as np

_HERE = os.path.dirname(__file__)
_NPZ = os.path.join(_HERE, "golden_kinematics.npz")
_JSON = os.path.join(_HERE, "golden_kinematics.json")
_SCRIPT = os.path.join(_HERE, "emit_web_golden.py")
_MODES = ("gamma_acoustic", "gamma_optical", "m_point", "k_point", "generic")


def _emitter():
    spec = importlib.util.spec_from_file_location("emit_web_golden", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _emit(tmp_path):
    out = _emitter().emit_web_golden(out_path=str(tmp_path / "golden_kinematics_web.json"))
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def test_web_golden_disp_phi_is_verbatim(tmp_path):
    W = _emit(tmp_path)
    z = np.load(_NPZ)
    for m in _MODES:
        assert np.array_equal(np.array(W["modes"][m]["disp_phi"]),
                              z[f"{m}__disp_phi"]), f"disp_phi mismatch for {m}"


def test_web_golden_eigenvectors_decode_to_canonical(tmp_path):
    from phonon_core.kinematics import canonicalize_eigenvector_phase
    W = _emit(tmp_path)
    z = np.load(_NPZ)
    for m in _MODES:
        er = np.array(W["modes"][m]["eigenvector_real"])
        ei = np.array(W["modes"][m]["eigenvector_imag"])
        assert np.array_equal(er + 1j * ei,
                              canonicalize_eigenvector_phase(z[f"{m}__eigvec_raw"])), m


def test_web_golden_metadata_matches_source(tmp_path):
    W = _emit(tmp_path)
    with open(_JSON, encoding="utf-8") as f:
        g = json.load(f)
    for m in _MODES:
        assert W["modes"][m]["band_index"] == g["modes"][m]["band_index"]
        assert W["modes"][m]["is_finite_q"] == g["modes"][m]["is_finite_q"]
        assert W["modes"][m]["q_frac"] == g["modes"][m]["q_point"]
        assert W["modes"][m]["frequency_THz"] == g["modes"][m]["frequency_THz"]
    assert W["inputs"]["amplitude"] == g["inputs"]["amplitude"]
    assert W["inputs"]["phi"] == g["inputs"]["phi"]
    assert W["inputs"]["mass_weighted"] == g["inputs"]["mass_weighted"]


def test_web_golden_carries_masses_from_npz(tmp_path):
    """0.6.6: the web golden carries primitive masses (verbatim from the npz, same
    order as the eigenvector atom axis) so the JS gate can recompute the
    mass-weighted displacement."""
    W = _emit(tmp_path)
    z = np.load(_NPZ)
    na = len(W["modes"]["generic"]["eigenvector_real"])
    assert len(W["masses"]) == na
    assert np.array_equal(np.array(W["masses"]), z["_material_masses"])


def test_emitter_does_not_touch_uploads():
    src = open(_SCRIPT, encoding="utf-8").read()
    assert ("/mnt/" + "user-data/uploads") not in src
    assert "capture_golden" not in src.split('"""', 2)[-1]  # no regen-path call in code
