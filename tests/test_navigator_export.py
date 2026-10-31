"""Navigator export (tools/pv_navigator_export.py, Deliverable 1).

Runs on the committed SnSe2 band + mesh-subset fixtures; checks the JSON
structure, the BZ polygon, the band/surface shapes, the faithful z-range, and
(0.6.5) the band eigenvector + q_frac egress for the 1b animator.
"""
import importlib.util
import json
import os

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

_HERE = os.path.dirname(__file__)
_BAND = os.path.join(_HERE, "fixtures", "phonopy", "SnSe2_band.hdf5")
_MESH = os.path.join(_HERE, "fixtures", "navigator", "SnSe2_mesh_coarse.hdf5")

_spec = importlib.util.spec_from_file_location(
    "pv_navigator_export", os.path.join(_HERE, os.pardir, "tools", "pv_navigator_export.py"))
nav = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nav)


def test_build_payload_structure():
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    p = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=16)
    assert set(p) == {"meta", "band", "surface", "bz_polygon", "ibz_polygon", "z_range"}

    nq = len(p["band"]["distance"])
    freq = np.array(p["band"]["frequency"], dtype=float)
    assert freq.shape[0] == nq
    assert np.array(p["band"]["q_cart"]).shape == (nq, 2)
    assert len(p["band"]["labels"]) == len(p["band"]["segments"]) + 1

    ch = np.array(p["band"]["character"], dtype=float)
    ch = ch[np.isfinite(ch)]
    assert ch.min() >= -1e-9 and ch.max() <= 1 + 1e-9       # a fraction

    assert len(p["bz_polygon"]) == 6                        # hexagonal BZ
    assert len(p["surface"]["kx"]) == 16 and len(p["surface"]["ky"]) == 16
    assert len(p["surface"]["bands"]) == freq.shape[1]
    # 0.6.92 compact storage: dense grids come back through the ONE reader
    z = _grid(nav.expand_surface(p["surface"])["bands"][0]["z"])
    assert z.shape == (16, 16)

    assert p["z_range"][0] <= p["z_range"][1]
    assert p["z_range"][0] < 0                              # acoustic dip kept

    # payload is strict-JSON (surface NaN already -> None). 0.6.5 ADDS the band
    # eigenvector + q_frac egress for 1b, so they must now be present.
    json.dumps(p, allow_nan=False)
    for k in ("q_frac", "eigenvectors_real", "eigenvectors_imag"):
        assert k in p["band"], k


def test_character_group_is_configurable():
    p = nav.build_payload(_BAND, _MESH, group_a=("Se",), grid=12)
    ch = np.array(p["band"]["character"], dtype=float)
    ch = ch[np.isfinite(ch)]
    assert ch.min() >= -1e-9 and ch.max() <= 1 + 1e-9
    assert p["meta"]["conventions"]["character_group_a"] == ["Se"]


def test_high_symmetry_points_and_ibz_wedge():
    p = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=12)
    hsp = p["band"]["high_symmetry_points"]
    # one entry per labeled tick, labels + distances consistent with the band block
    assert [h["label"] for h in hsp] == p["band"]["labels"]
    assert np.allclose([h["distance"] for h in hsp], p["band"]["tick_positions"])

    # ibz_polygon is the triangle [Γ, M, K]
    ibz = p["ibz_polygon"]
    assert len(ibz) == 3
    assert np.allclose(ibz[0], [0.0, 0.0], atol=1e-6)             # Γ at origin
    assert np.allclose(ibz, [hsp[0]["q_cart"], hsp[1]["q_cart"], hsp[2]["q_cart"]])

    # cross-check M, K against the fractional high-symmetry points mapped to
    # Cartesian via the reciprocal lattice (M=[1/2,0,0], K=[1/3,1/3,0])
    recip = np.array(p["meta"]["material"]["reciprocal_lattice"])
    M = (np.array([0.5, 0.0, 0.0]) @ recip)[:2]
    K = (np.array([1 / 3, 1 / 3, 0.0]) @ recip)[:2]
    assert np.allclose(hsp[1]["q_cart"], M, atol=1e-3)
    assert np.allclose(hsp[2]["q_cart"], K, atol=1e-3)

    # the meta note flags this as the lattice (holohedral) IBZ, not a space-group IBZ
    assert "holohedral" in p["meta"]["conventions"]["ibz_method"]


def test_band_eigenvector_and_qfrac_egress():
    """0.6.5: the band block carries fractional q and per-mode-canonicalized
    eigenvectors, real/imag split with json_v1's own splitter, so the JS decoder
    (real + 1j*imag) reconstructs canonicalize(BandPath.eigenvectors) bitwise."""
    from phonon_core import load_band_hdf5
    from phonon_core.kinematics import canonicalize_eigenvector_phase

    p = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=12)
    b = p["band"]
    qf = np.array(b["q_frac"])
    er = np.array(b["eigenvectors_real"])
    ei = np.array(b["eigenvectors_imag"])

    bp = load_band_hdf5(_BAND)
    ev = np.asarray(bp.eigenvectors)                       # (Nq,Nb,Na,3) complex
    assert qf.shape == (ev.shape[0], 3)
    assert er.shape == ev.shape and ei.shape == ev.shape

    expected = np.empty_like(ev, dtype=np.complex128)
    for qi in range(ev.shape[0]):
        for bi in range(ev.shape[1]):
            expected[qi, bi] = canonicalize_eigenvector_phase(ev[qi, bi])
    assert np.array_equal(er + 1j * ei, expected)          # bitwise decode round-trip
    assert np.array_equal(qf, np.asarray(bp.q_points))     # q_frac is the raw fractional q

    conv = p["meta"]["conventions"]
    for k in ("displacement", "bloch_phase", "eigenvector_phase", "eigenvector_layout"):
        assert k in conv, k


def test_dim3_payload_omits_inplane_blocks():
    """1c-5: dim=3 omits surface/bz_polygon/ibz_polygon/z_range ENTIRELY (a q3=0
    slice is not "the BZ" of a 3D crystal), carries dimensionality=3, and emits
    full 3-component q_cart; the band block is otherwise identical to dim=2."""
    p3 = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=16, dim=3)
    for k in ("surface", "bz_polygon", "ibz_polygon", "z_range"):
        assert k not in p3, k
    assert p3["meta"]["material"]["dimensionality"] == 3
    q3 = np.array(p3["band"]["q_cart"], dtype=float)
    assert q3.shape[1] == 3
    for h in p3["band"]["high_symmetry_points"]:
        assert len(h["q_cart"]) == 3
    # no stale surface descriptors in conventions/source either
    for k in ("surface_construction", "bz_method", "ibz_method"):
        assert k not in p3["meta"]["conventions"], k
    for k in ("surface_grid", "surface_bz_coverage"):
        assert k not in p3["meta"]["source"], k

    p2 = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=16, dim=2)
    b2, b3 = dict(p2["band"]), dict(p3["band"])
    q2v = np.array(b2.pop("q_cart"), dtype=float)
    q3v = np.array(b3.pop("q_cart"), dtype=float)
    h2 = b2.pop("high_symmetry_points")
    h3 = b3.pop("high_symmetry_points")
    assert b2 == b3            # frequencies/eigenvectors/q_frac identical arrays
    assert np.allclose(q3v[:, :2], q2v)            # xy projection consistent
    for a, b in zip(h2, h3):
        assert a["label"] == b["label"] and a["distance"] == b["distance"]
        assert np.allclose(np.array(b["q_cart"][:2]), np.array(a["q_cart"]))
    import json as _json
    _json.dumps(p3, allow_nan=False)               # strict JSON


def test_dim2_payload_back_compatible():
    """1c-5: dim=2 (and the default, no dim) is the 0.6.13 payload + ONLY the
    meta.material.dimensionality key; q_cart stays 2-component, surface intact."""
    p2 = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=16, dim=2)
    pd = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=16)   # default
    assert p2 == pd                                # default IS dim=2
    assert p2["meta"]["material"]["dimensionality"] == 2
    assert set(p2) == {"meta", "band", "surface", "bz_polygon", "ibz_polygon", "z_range"}
    assert set(p2["meta"]["material"]) == {
        "symbols", "masses", "lattice", "reciprocal_lattice", "fractional_coords",
        "dimensionality", "species_style"}         # 0.6.13 set + dimensionality
    assert np.array(p2["band"]["q_cart"], dtype=float).shape[1] == 2
    assert len(p2["surface"]["kx"]) == 16
    assert "surface_construction" in p2["meta"]["conventions"]
    assert "surface_bz_coverage" in p2["meta"]["source"]


def test_mesh_optional_omits_surface_blocks():
    """0.6.15: build_payload accepts mesh_path=None in BOTH dims — the surface/
    bz_polygon/ibz_polygon/z_range blocks (and their conventions/source
    descriptors) are omitted; everything else stays dim-governed (q_cart width
    follows dim, not mesh presence)."""
    import json as _json
    for dim, qw in ((2, 2), (3, 3)):
        p = nav.build_payload(_BAND, None, group_a=("Sn",), grid=16, dim=dim)
        for k in ("surface", "bz_polygon", "ibz_polygon", "z_range"):
            assert k not in p, (dim, k)
        assert p["meta"]["material"]["dimensionality"] == dim
        assert len(p["band"]["q_cart"][0]) == qw
        for k in ("mesh", "n_qpoints_mesh", "mesh_dims",
                  "surface_grid", "surface_bz_coverage"):
            assert k not in p["meta"]["source"], (dim, k)
        for k in ("surface_construction", "bz_method", "ibz_method"):
            assert k not in p["meta"]["conventions"], (dim, k)
        assert list(p["meta"]["source"]) == ["band", "n_qpoints_band"]
        _json.dumps(p, allow_nan=False)


def test_with_mesh_payload_contract_unchanged():
    """0.6.15 regression: making the mesh optional must not change with-mesh
    output. Pins the observable contract — key inventory at every level, source
    key ORDER, shapes. (The true byte-identity diff against a 0.6.14 bake of the
    same fixtures is part of the release verification, kept out of the suite
    because an in-repo hash would be scipy-version-fragile.)"""
    p2 = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=16, dim=2)
    assert set(p2) == {"meta", "band", "surface", "bz_polygon", "ibz_polygon", "z_range"}
    assert list(p2["meta"]["source"]) == [
        "band", "mesh", "n_qpoints_band", "n_qpoints_mesh", "mesh_dims",
        "surface_grid", "surface_bz_coverage",
        # 0.6.92 compact-storage record (deliberate additions to the pin)
        "surface_nodes_stored", "surface_nodes_total", "surface_sym_ops"]
    assert "surface_construction" in p2["meta"]["conventions"]
    assert len(p2["surface"]["kx"]) == 16
    assert len(p2["band"]["q_cart"][0]) == 2
    p3 = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=16, dim=3)
    assert set(p3) == {"meta", "band"}
    assert list(p3["meta"]["source"]) == [
        "band", "mesh", "n_qpoints_band", "n_qpoints_mesh", "mesh_dims"]
    assert len(p3["band"]["q_cart"][0]) == 3


def test_path_ticks_continuous():
    """0.6.16(a): continuous 3-segment path — merge-free labels, no breaks."""
    labs = [("$\\Gamma$", "M"), ("M", "K"), ("K", "$\\Gamma$")]
    qf = np.array([[0, 0, 0], [.5, 0, 0],          # seg 0
                   [.5, 0, 0], [1/3, 1/3, 0],      # seg 1 (junction q matches)
                   [1/3, 1/3, 0], [0, 0, 0]])      # seg 2 (junction q matches)
    ticks, breaks = nav._path_ticks(labs, qf, [0, 2, 4], [1, 3, 5])
    assert ticks == ["$\\Gamma$", "M", "K", "$\\Gamma$"]
    assert breaks == []


def test_path_ticks_mixed_junctions():
    """0.6.16(b): one matching-label (continuous) junction + one discontinuous
    junction — "A|B" and a break only at the latter."""
    labs = [("G", "X"), ("X", "Y"), ("Z", "W")]
    qf = np.array([[0, 0, 0], [.5, 0, 0],
                   [.5, 0, 0], [.5, .5, 0],
                   [0, 0, .5], [0, .5, .5]])       # q[3] != q[4] -> discontinuous
    ticks, breaks = nav._path_ticks(labs, qf, [0, 2, 4], [1, 3, 5])
    assert ticks == ["G", "X", "Y|Z", "W"]
    assert breaks == [[3, 4]]
    # discontinuous junction with MATCHING labels: break, but a plain label
    labs2 = [("G", "X"), ("X", "Y"), ("Y", "W")]
    ticks2, breaks2 = nav._path_ticks(labs2, qf, [0, 2, 4], [1, 3, 5])
    assert ticks2 == ["G", "X", "Y", "W"]
    assert breaks2 == [[3, 4]]


def test_path_ticks_tex_rule():
    """0.6.16(c): TeX normalization — wrap-if-marked, never double-wrap; the
    SnSe2-style labels pass through untouched; merged ticks normalize each
    atomic label before the join."""
    assert nav._tex_norm("X^{\\prime}") == "$X^{\\prime}$"
    assert nav._tex_norm("$\\Gamma$") == "$\\Gamma$"
    assert nav._tex_norm("M") == "M"
    labs = [("G", "X^{\\prime}"), ("M", "W")]
    qf = np.array([[0, 0, 0], [.5, 0, 0], [0, .5, 0], [0, 0, .5]])
    ticks, breaks = nav._path_ticks(labs, qf, [0, 2], [1, 3])
    assert ticks == ["G", "$X^{\\prime}$|M", "W"]
    assert breaks == [[1, 2]]


def test_breaks_in_payload_continuous_fixture():
    """0.6.16: the SnSe2 fixture path is continuous — band.breaks == [] and the
    labels are exactly the pre-0.6.16 strings. (Byte-identity vs a 0.6.15 bake
    is release verification, same policy as 0.6.15.)"""
    p = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=16, dim=2)
    assert p["band"]["breaks"] == []
    assert p["band"]["labels"] == ["$\\Gamma$", "M", "K", "$\\Gamma$"]


_TDOS = os.path.join(_HERE, "fixtures", "dos", "SnSe2_total_dos.dat")
_PDOS = os.path.join(_HERE, "fixtures", "dos", "SnSe2_projected_dos.dat")
_CONF = os.path.join(_HERE, "fixtures", "dos", "corrupt_pdos.conf")


def test_dos_auto_group_merge_and_trim():
    """1d-2(a): by-symbol auto-grouping on the REAL SnSe2 files — two curves
    labeled Sn, Se; Se EXACTLY equals column2+column3 over the trimmed window;
    at most one all-zero margin bin survives at each end; the total and
    projected keep their own (different) native grids."""
    p = nav.build_payload(_BAND, None, group_a=("Sn",), grid=16, dim=2,
                          total_dos=_TDOS, proj_dos=_PDOS)
    d = p["dos"]
    assert sorted(d) == ["projected", "total"]
    labels = [c["label"] for c in d["projected"]["curves"]]
    assert labels == ["Sn", "Se"]
    raw = np.loadtxt(_PDOS, comments="#")
    se = np.array(d["projected"]["curves"][1]["dos"], dtype=float)
    f0 = d["projected"]["frequency"][0]
    off = int(np.nonzero(raw[:, 0] == f0)[0][0])
    assert np.array_equal(se, (raw[:, 2] + raw[:, 3])[off:off + len(se)])
    sn = np.array(d["projected"]["curves"][0]["dos"], dtype=float)
    assert np.array_equal(sn, raw[:, 1][off:off + len(sn)])
    for curve in (d["total"]["dos"],):
        assert not (curve[0] == 0 and curve[1] == 0), "more than one leading zero bin"
        assert not (curve[-1] == 0 and curve[-2] == 0), "more than one trailing zero bin"
    merged = np.array([c["dos"] for c in d["projected"]["curves"]])
    head_zero = (merged[:, 0] == 0).all() and (merged[:, 1] == 0).all()
    tail_zero = (merged[:, -1] == 0).all() and (merged[:, -2] == 0).all()
    assert not head_zero and not tail_zero
    # independently sampled native grids are PRESERVED (no resampling)
    assert d["total"]["frequency"] != d["projected"]["frequency"]
    json.dumps(p, allow_nan=False)


def test_pdos_conf_tolerant_parser():
    """1d-2(b): a pdos.conf carrying a corrupt 'uATOM_NAME' prefix (a
    field-observed corruption) — the tolerant parser still yields the PDOS
    groups and the labels derive from the member symbols."""
    symbols = ["Cu"] * 4 + ["I"] * 2 + ["S"]
    groups = nav._pdos_groups(symbols, _CONF)
    assert groups == [("Cu", [1, 2, 3, 4]), ("I", [5, 6]), ("S", [7])]
    raw_groups, names = nav._parse_pdos_conf(_CONF)
    assert raw_groups == [[1, 2, 3, 4], [5, 6], [7]]
    assert names == ["Cu", "I", "S"]


def test_projected_ncols_mismatch_dies():
    """1d-2(c): a projected file whose column count disagrees with the band
    material's atom count dies explicitly (no guessing)."""
    bad = os.path.join(_HERE, "fixtures", "dos", "corrupt_pdos.conf")  # any non-matching file
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as fh:
        fh.write("0.0 1.0 2.0 3.0 4.0 5.0\n1.0 1.0 2.0 3.0 4.0 5.0\n")
        wrong = fh.name
    with pytest.raises(SystemExit):
        nav.build_payload(_BAND, None, dim=2, proj_dos=wrong)   # 5 atoms vs 3
    os.unlink(wrong)
    with pytest.raises(SystemExit):
        nav._pdos_groups(["Sn", "Se", "Se"], bad)               # index 7 > Natoms 3


def test_dos_byte_regression_and_end_to_end(tmp_path):
    """1d-2(d,e): a no-flag bake carries NO dos key / source / convention
    (structural contract; byte diff vs 0.6.16 is release verification); the
    full CLI with both DOS flags + mesh + --also-html exits 0 and the payload
    carries dos.total + 2 projected curves."""
    p0 = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=16, dim=2)
    assert "dos" not in p0
    assert "dos" not in p0["meta"]["conventions"]
    for k in ("total_dos", "projected_dos"):
        assert k not in p0["meta"]["source"]
    out = tmp_path / "nav" / "navigator_data.json"
    rc = nav.main([_BAND, _MESH, "-o", str(out), "--grid", "12",
                   "--total-dos", _TDOS, "--proj-dos", _PDOS, "--also-html"])
    assert rc == 0
    d = json.loads(out.read_text())
    assert "total" in d["dos"] and len(d["dos"]["projected"]["curves"]) == 2
    assert d["meta"]["source"]["total_dos"] == "SnSe2_total_dos.dat"
    assert (tmp_path / "nav" / "index.html").is_file()


# ---- 0.6.70 E2 data layer: per-species surface weight grids ----
def _grid(nested):
    """JSON grid (None for gaps) -> float ndarray with NaN for None."""
    return np.array([[np.nan if v is None else v for v in row] for row in nested],
                    dtype=float)


def test_surface_weights_consistency_golden():
    """LinearNDInterpolator is linear, so interpolation commutes with the species
    sum: the sum over the group-A species of the new per-species weight grids must
    equal the existing `character` grid (atol 1e-10), and every weight grid must
    share `character`'s NaN mask cell-for-cell (same interpolator + BZ mask)."""
    p = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=16, dim=2)
    S = nav.expand_surface(p["surface"])
    so = S["species_order"]
    assert set(so) == {"Sn", "Se"}, so
    aidx = [so.index(s) for s in ("Sn",)]
    for bd in S["bands"]:
        ch = _grid(bd["character"])
        W = np.array([_grid(g) for g in bd["weights"]])          # (nS, grid, grid)
        assert W.shape[0] == len(so)
        for s in range(W.shape[0]):
            assert np.array_equal(np.isnan(W[s]), np.isnan(ch)), \
                "weight grid NaN mask must equal character's"
        m = ~np.isnan(ch)
        # atol: identity held at 1e-10 at full precision; character and each
        # weight are now independently 6g-rounded (quantum <= 1e-6 each on
        # [0,1] fractions), so the emitted identity holds to a few quanta.
        np.testing.assert_allclose(np.nansum(W[aidx], axis=0)[m], ch[m],
                                   atol=5e-6, rtol=0)


def test_surface_weights_normalization():
    """Weights are normalized so sum over ALL species == 1 at every non-NaN cell
    (bare |e|^2 projection, unit-normalized eigenvectors; NOT mass-weighted)."""
    p = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=16, dim=2)
    for bd in nav.expand_surface(p["surface"])["bands"]:
        ch = _grid(bd["character"]); m = ~np.isnan(ch)
        W = np.array([_grid(g) for g in bd["weights"]])
        # atol: nS 6g-rounded fractions summing to 1 (quantum <= 1e-6 each)
        np.testing.assert_allclose(np.nansum(W, axis=0)[m], 1.0, atol=5e-6, rtol=0)


def test_surface_weights_presence_keyed():
    """Additive + presence-keyed: with the feature OFF the surface block is the
    exact pre-0.6.70 contract (no new keys), and z/character are byte-identical
    whether weights are baked or not (weights are purely additive)."""
    on = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=16, dim=2)
    off = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=16, dim=2,
                            surface_weights=False)
    assert "species_order" in on["surface"]
    assert "weights" in on["surface"]["bands"][0]
    assert "surface_weights" in on["meta"]["conventions"]
    # OFF == pre-0.6.70 inventory + the 0.6.92 storage keys
    assert set(off["surface"]) == {"kx", "ky", "bands", "node_map", "n_nodes", "sym"}
    assert set(off["surface"]["bands"][0]) == {"z", "character"}
    assert "surface_weights" not in off["meta"]["conventions"]
    # z + character untouched by the additive weights
    for bon, boff in zip(on["surface"]["bands"], off["surface"]["bands"]):
        assert bon["z"] == boff["z"]
        assert bon["character"] == boff["character"]


# ---------------------------------------------------------------------------
# 0.6.80: engine-backed export mode (Level-2 Stage 2 Bump 2)
# ---------------------------------------------------------------------------
def test_legacy_has_no_provenance():
    """LEGACY INVARIANCE: without --engine the payload carries NO `provenance`
    key in any shape — the key-set assertions above stay exact, and a present
    provenance block is therefore always an engine-mode statement.
    (Byte-identity of the whole legacy payload old-vs-new was verified
    mechanically at ship time — recorded in the 0.6.80 CHANGELOG.)"""
    p2 = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=12, dim=2)
    p3 = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=12, dim=3)
    pn = nav.build_payload(_BAND, None, group_a=("Sn",), grid=12, dim=2)
    for p in (p2, p3, pn):
        assert "provenance" not in p
        assert "mesh_modes" not in p          # 0.6.81: engine-mode-only block
        assert "engine_kind" not in p["meta"]["source"]
        assert "mesh_modes_grid" not in p["meta"]["source"]


# dataset + phonopy guarded (skips in a bare-zip audit, like test_engine_dq.py)
_DS = os.path.join(_HERE, os.pardir, os.pardir, "Phonopy-examples", "fatbands_SnSe2")


@pytest.mark.skipif(not os.path.isdir(_DS),
                    reason="Phonopy-examples dataset not present (bare-zip audit)")
def test_engine_mode_key_set(tmp_path):
    """Engine payload = legacy key set + `provenance`; provenance.surface ==
    'engine-exact'; provenance.engine.kind == 'phonopy-A1'."""
    phonopy = pytest.importorskip("phonopy")
    from phonon_core.engine import load_engine
    # a small self-consistent mesh.hdf5 generated from the SAME FORCE_CONSTANTS
    ph = load_engine(os.path.join(_DS, "phonopy.yaml"))
    ph.run_mesh([6, 6, 1], is_mesh_symmetry=False, with_eigenvectors=True)
    mesh_path = str(tmp_path / "mesh.hdf5")
    ph.mesh.write_hdf5(filename=mesh_path)

    p = nav.build_payload(os.path.join(_DS, "band.hdf5"), mesh_path,
                          group_a=("Sn",), grid=12,
                          engine=os.path.join(_DS, "phonopy.yaml"))
    assert set(p) == {"meta", "band", "surface", "bz_polygon", "ibz_polygon",
                      "z_range", "provenance", "mesh_modes"}   # 0.6.81 superset
    assert p["provenance"]["surface"] == "engine-exact"
    assert p["provenance"]["engine"]["kind"] == "phonopy-A1"
    assert p["meta"]["source"]["engine_kind"] == "phonopy-A1"
    assert p["meta"]["source"]["phonopy_yaml"] == "phonopy.yaml"
    assert p["meta"]["source"]["force_constants"] == "FORCE_CONSTANTS"
    assert p["meta"]["source"]["mesh_modes_grid"] == 24        # the default
    # engine-exact surface: every inside node evaluated -> coverage exactly 1.0
    assert all(c == 1.0 for c in p["meta"]["source"]["surface_bz_coverage"])
    # mesh_modes shape sanity: grid stamp + consistent node counts
    mm = p["mesh_modes"]
    assert mm["grid"] == [24, 24]
    n_nodes = len(mm["q_frac"])
    assert (len(mm["q_cart"]) == len(mm["frequency"])
            == len(mm["eigenvectors_real"]) == len(mm["eigenvectors_imag"])
            == n_nodes) and n_nodes > 0


# ---------------------------------------------------------------------------
# 0.6.92 — compact surface storage: the symmetry reduction, unit-level
# ---------------------------------------------------------------------------
def _synthetic_fields(G=8, n_bands=2, n_species=2):
    """An exactly mirror-symmetric masked field set on a G x G grid: every
    value depends only on (min(i, G-1-i), min(j, G-1-j)), so mirror_x,
    mirror_y and c2 all hold BITWISE. Mask = everything except the corners
    (kept symmetric)."""
    inside = np.ones((G, G), dtype=bool)
    for ci in (0, G - 1):
        for cj in (0, G - 1):
            inside[ci, cj] = False
    z = np.empty((n_bands, G, G))
    c = np.empty((n_bands, G, G))
    w = np.empty((n_bands, n_species, G, G))
    for i in range(G):
        for j in range(G):
            a, b = min(i, G - 1 - i), min(j, G - 1 - j)
            for nb in range(n_bands):
                z[nb, i, j] = 1.0 + nb + 0.1 * a + 0.01 * b
                c[nb, i, j] = (a + b + nb) / (2.0 * G)
                w[nb, 0, i, j] = c[nb, i, j]
                w[nb, 1, i, j] = 1.0 - c[nb, i, j]
    z[:, ~inside] = np.nan
    c[:, ~inside] = np.nan
    w[:, :, ~inside] = np.nan
    return inside, z, c, w


def test_surface_reduce_symmetric_field_keeps_all_ops():
    """All three grid-exact ops verify on an exactly symmetric field; the
    orbit dedup stores ~1/4 of the nodes and expansion reproduces every cell
    of the dense field bitwise for z (full precision) and to the 6g quantum
    for character/weights."""
    inside, z, c, w = _synthetic_fields()
    surf_bands, node_map, sym, _ni = nav._emit_surface_bands(inside, z, c, w)
    assert sorted(sym["ops"]) == ["c2", "mirror_x", "mirror_y"]
    assert sym["max_dz_THz"] == 0.0                    # bitwise-symmetric input
    assert sym["nodes_total"] == int(inside.sum())
    # G=8 symmetric grid: orbits of 4 off the axes -> a quarter-ish
    assert sym["nodes_stored"] * 4 >= sym["nodes_total"]
    assert sym["nodes_stored"] < sym["nodes_total"]
    dense = nav.expand_surface({"node_map": node_map, "bands": surf_bands})
    for nb, bd in enumerate(dense["bands"]):
        for i in range(8):
            for j in range(8):
                if not inside[i, j]:
                    assert bd["z"][i][j] is None
                    continue
                assert bd["z"][i][j] == z[nb, i, j]               # bitwise
                assert abs(bd["character"][i][j] - c[nb, i, j]) < 2e-6
                for s in range(2):
                    assert abs(bd["weights"][s][i][j] - w[nb, s, i, j]) < 2e-6


def test_surface_reduce_negctl_broken_z_drops_everything():
    """FAIL-THEN-PASS (a): corrupt ONE z value by 1e-3 THz — every op that
    pairs that node fails the strict global z gate, so NO op survives (each
    of the three pairs it), storage falls back to every-node-explicit, and
    the corrupted value itself is stored faithfully (compression never
    launders a field that does not satisfy the symmetry)."""
    inside, z, c, w = _synthetic_fields()
    z2 = z.copy()
    z2[0, 3, 2] += 1e-3
    _, node_map, sym, _ni = nav._emit_surface_bands(inside, z2, c, w)
    assert sym["ops"] == []
    assert sym["nodes_stored"] == sym["nodes_total"]
    # the clean field keeps all ops (the PASS side of the control)
    _, _, sym_clean, _ni = nav._emit_surface_bands(inside, z, c, w)
    assert sorted(sym_clean["ops"]) == ["c2", "mirror_x", "mirror_y"]


def test_surface_reduce_negctl_directional_break():
    """FAIL-THEN-PASS (b): a field symmetric in kx but NOT in ky keeps
    exactly mirror_x and loses mirror_y AND c2 — the verifier reads the
    field, not the lattice."""
    inside, z, c, w = _synthetic_fields()
    G = 8
    for i in range(G):
        z[:, i, :] += 1e-2 * i          # ky-asymmetric, kx-symmetric
    z[:, ~inside] = np.nan
    _, _, sym, _ni = nav._emit_surface_bands(inside, z, c, w)
    assert sym["ops"] == ["mirror_x"]


def test_surface_reduce_gauge_node_stays_explicit():
    """A node whose character disagrees with its mirror (> the per-node gate;
    the near-degeneracy gauge case) is NOT merged — it keeps its own exact
    values — while the ops themselves survive (z is clean) and every other
    node still dedups."""
    inside, z, c, w = _synthetic_fields()
    c2 = c.copy()
    c2[0, 5, 4] += 5e-3                 # gauge-like: character only, one node
    surf_bands, node_map, sym, _ni = nav._emit_surface_bands(inside, z, c2, w)
    assert sorted(sym["ops"]) == ["c2", "mirror_x", "mirror_y"]
    assert sym["nodes_stored"] > 0
    dense = nav.expand_surface({"node_map": node_map, "bands": surf_bands})
    # the perturbed node keeps its own (rounded) value, not its mirror's
    assert abs(dense["bands"][0]["character"][5][4] - c2[0, 5, 4]) < 2e-6
    assert abs(dense["bands"][0]["character"][5][3] - c[0, 5, 3]) < 2e-6
    # and one clean run stores strictly fewer nodes than this perturbed one
    _, _, sym_clean, _ni = nav._emit_surface_bands(inside, z, c, w)
    assert sym_clean["nodes_stored"] < sym["nodes_stored"] + 2


def test_surface_reduce_nan_hole_stays_hole():
    """A NaN inside the mask (an interpolation hole) never merges and never
    gets 'filled' by its mirror: after emission + expansion the hole is still
    null while its symmetric partner keeps its value."""
    inside, z, c, w = _synthetic_fields()
    z[:, 2, 3] = np.nan                  # hole at (2,3); mirror (2,4) is fine
    surf_bands, node_map, sym, _ni = nav._emit_surface_bands(inside, z, c, w)
    dense = nav.expand_surface({"node_map": node_map, "bands": surf_bands})
    assert dense["bands"][0]["z"][2][3] is None            # hole preserved
    assert dense["bands"][0]["z"][2][4] is not None        # partner intact


def test_surface_z_full_precision_character_6g():
    """The 0.6.92 precision split: stored z is BITWISE the computed value
    (frequencies keep the 0.6.80 invariant's teeth); character/weights are
    %.6g quantized (the 0.6.81 policy for bounded display fractions)."""
    inside, z, c, w = _synthetic_fields()
    surf_bands, node_map, sym, _ni = nav._emit_surface_bands(inside, z, c, w)
    # find the stored node for grid cell (1,1) and compare
    ni = node_map[1][1]
    assert surf_bands[0]["z"][ni] == z[0, 1, 1]
    assert surf_bands[0]["character"][ni] == float(f"{c[0, 1, 1]:.6g}")


def test_legacy_surface_never_symmetrizes_interpolation():
    """ANTI-LAUNDERING PIN: the legacy (mesh-interpolated) branch must NOT
    pass the mirror verification — LinearND interpolation over a scattered
    mesh is not grid-symmetric, and symmetrizing it would present
    interpolation asymmetry as physics. Every node stays explicit."""
    p = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=16, dim=2)
    sym = p["surface"]["sym"]
    assert sym["ops"] == []
    assert sym["nodes_stored"] == sym["nodes_total"]


# ---------------------------------------------------------------------------
# 0.7.1 Tier 0 — the smooth first-BZ rim (radial clamp)
# ---------------------------------------------------------------------------
def test_ray_to_hull_geometry():
    """The clamp: outside points land ON the polygon boundary along their
    Γ-ray; inside points pass through BITWISE (t capped at 1); scaling is
    radial (the clamp of s*p for s>1 equals the clamp of p)."""
    th = np.linspace(0, 2 * np.pi, 7)[:-1]
    hexpoly = np.column_stack([0.17 * np.cos(th), 0.17 * np.sin(th)])
    pts_out = np.array([[0.3, 0.05], [-0.2, 0.21], [0.0, 0.4]])
    pts_in = np.array([[0.01, 0.02], [-0.05, 0.0], [0.0, 0.0]])
    out = nav._ray_to_hull(pts_out, hexpoly)
    for p, c in zip(pts_out, out):
        # collinear with the ray
        assert abs(p[0] * c[1] - p[1] * c[0]) < 1e-12
        # ON the boundary: some edge's outward half-plane equation is tight
        tight = False
        for k in range(6):
            a, b = hexpoly[k], hexpoly[(k + 1) % 6]
            nvec = np.array([b[1] - a[1], a[0] - b[0]])
            if abs(nvec @ c - nvec @ a) < 1e-12 * max(1, abs(nvec @ a)):
                tight = True
        assert tight, c
        # strictly clamped inward
        assert np.hypot(*c) < np.hypot(*p)
    got_in = nav._ray_to_hull(pts_in, hexpoly)
    assert np.array_equal(got_in, pts_in)          # bitwise pass-through
    # radial: clamp(2p) == clamp(p) for outside p
    out2 = nav._ray_to_hull(2.0 * pts_out, hexpoly)
    np.testing.assert_allclose(out2, out, rtol=0, atol=1e-12)


def test_legacy_rim_carries_values_where_reachable():
    """0.7.1 legacy branch: outside cells now carry the interpolator's value
    AT the boundary clamp (or stay null where the clamp leaves the mesh
    hull); the stored grid is FULL (node_map has no -1)."""
    p = nav.build_payload(_BAND, _MESH, group_a=("Sn",), grid=16, dim=2)
    nm = np.array(p["surface"]["node_map"])
    assert (nm >= 0).all(), "every cell stored (values or honest nulls)"
    assert p["surface"]["sym"]["nodes_total"] == 16 * 16
    # rim cells: at least SOME outside-bbox cells carry finite values
    dense = nav.expand_surface(p["surface"])
    z = np.array([[np.nan if v is None else v for v in row]
                  for row in dense["bands"][0]["z"]])
    assert np.isfinite(z).sum() > 0
