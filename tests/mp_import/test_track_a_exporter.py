"""mp_import Track A — the converted material through the EXISTING exporter
(0.6.89): dim-3 band-only engine export, the first MP-sourced input to that
branch.

Reuses the 0.6.88 test_navigator_engine idiom: the contract asserted below is
READ OFF THE EXPORTER (build_payload: surface/bz_polygon/ibz_polygon/z_range
gate on dim==2; mesh_modes additionally on the engine surface; dos on explicit
flags only) — exactly the key set the 0.6.88 dim-3 engine test pinned, with NO new
provenance keys (MP-source tagging is a later version).

The exporter needs a band.hdf5 (path spec + cross-check oracle). MP ships no
hdf5, so the battery generates a small SELF-CONSISTENT one from the SAME
converted FC via phonopy's own run_band_structure — the same move the 0.6.80
battery used for its mesh.hdf5.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

from conftest import load_band, requires_fixtures
from engine_metric import LAM_TOL, NU_FLOOR, doctrine_agreement

pytestmark = requires_fixtures

pytest.importorskip("scipy")
pytest.importorskip("matplotlib")
pytest.importorskip("h5py")

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "pv_navigator_export",
    os.path.join(_HERE, os.pardir, os.pardir, "tools", "pv_navigator_export.py"))
nav = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nav)

#: The dim==3 engine-mode key contract (read off the exporter; pinned at
#: 0.6.88 on a 3D engine-mode bake — asserted UNCHANGED for an MP-sourced input).
_DIM3_KEYS = {"meta", "band", "provenance"}
_2D_ONLY_KEYS = {"surface", "bz_polygon", "ibz_polygon", "z_range", "mesh_modes"}


@pytest.fixture(scope="module")
def si_export(tmp_path_factory):
    """Converted silicon + a self-consistent band.hdf5 + the dim-3 payload."""
    from phonon_core.mp_import import convert_pheasy
    from conftest import fixture_dir
    import phonopy

    out = str(tmp_path_factory.mktemp("mp_export_si"))
    res = convert_pheasy(fixture_dir("silicon_mp-149"), out)
    ph = phonopy.load(phonopy_yaml=res["phonopy_yaml"],
                      force_constants_filename=res["force_constants"],
                      symmetrize_fc=False, log_level=0)
    # a 3-segment Gamma-X-W-K path (51 pts/segment) from the MP labels
    _qs, _fr, b = load_band("silicon_mp-149")
    L = b["labels_dict"]
    segs = [np.linspace(L[a], L[c], 51)
            for a, c in (("\\Gamma", "X"), ("X", "W"), ("W", "K"))]
    ph.run_band_structure(segs, with_eigenvectors=True)
    band_h5 = os.path.join(out, "band.hdf5")
    ph.band_structure.write_hdf5(filename=band_h5)
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    payload = nav.build_payload(band_h5, None, group_a=("Si",), dim=3,
                                engine=res["phonopy_yaml"])
    return res, payload


def test_mp_dim3_contract(si_export):
    """Keys EXACTLY {meta, band, provenance}; dimensionality 3; q_cart keeps
    all three components; evaluated.mesh False; no provenance.surface; the
    exporter's own FC<->band cross-check < 1e-6 THz. No surprises = the
    dim-3 contract holds for MP-sourced inputs unchanged."""
    _res, p = si_export
    assert set(p.keys()) == _DIM3_KEYS, sorted(p.keys())
    assert not (_2D_ONLY_KEYS & set(p.keys()))
    assert p["meta"]["material"]["dimensionality"] == 3
    assert p["meta"]["material"]["symbols"] == ["Si", "Si"]
    src = p["meta"]["source"]
    assert src["engine_kind"] == "phonopy-A1"
    assert "mesh" not in src and "surface_grid" not in src
    prov = p["provenance"]
    assert prov["engine"]["evaluated"] == {"band": True, "mesh": False}
    assert "surface" not in prov
    cc = prov["cross_check"]["band_max_dw_THz"]
    qc = np.asarray(p["band"]["q_cart"])
    assert qc.ndim == 2 and qc.shape[1] == 3
    nq, nb = np.asarray(p["band"]["frequency"]).shape
    print(f"\n[mp-export] silicon dim=3: keys = {sorted(p.keys())}, "
          f"{nq} q x {nb} bands, cross-check = {cc:.3e} THz, "
          f"evaluated = {prov['engine']['evaluated']}")
    assert nb == 6
    assert cc < 1e-6


def test_mp_dim3_single_source(si_export):
    """SINGLE-SOURCE INVARIANT on the MP-sourced payload: fresh evaluate_dq
    at every payload band q_frac matches to 1e-6 THz (metric doctrine below
    NU_FLOOR) / overlap 1 - 1e-6."""
    from phonon_core.engine import evaluate_dq, load_engine
    from phonon_core.engine.compare import mode_agreement
    res, p = si_export
    eng = load_engine(res["phonopy_yaml"])
    band = p["band"]
    qs = np.asarray(band["q_frac"])
    f_pay = np.asarray(band["frequency"])
    v_pay = (np.asarray(band["eigenvectors_real"])
             + 1j * np.asarray(band["eigenvectors_imag"]))
    f0, v0 = evaluate_dq(eng, qs[0])
    f_new = np.empty_like(f_pay)
    v_new = np.empty(v_pay.shape, dtype=np.complex128)
    f_new[0], v_new[0] = f0, v0
    for i in range(1, len(qs)):
        f_new[i], v_new[i] = evaluate_dq(eng, qs[i])
    d_nu, d_lam = doctrine_agreement(f_pay, f_new)
    _df, ov, pd = mode_agreement(f_pay, v_pay, f_new, v_new)
    print(f"\n[mp-export] single source ({len(qs)} q): "
          f"max|dnu| (|nu|>{NU_FLOOR}) = {d_nu:.3e} THz, "
          f"max|dlambda| near 0 = {d_lam:.3e} THz^2, "
          f"overlap deficit = {1 - ov:.3e}, proj dev = {pd:.3e}")
    assert d_nu < 1e-6
    assert d_lam < LAM_TOL
    assert ov > 1 - 1e-6
    assert pd < 1e-6
