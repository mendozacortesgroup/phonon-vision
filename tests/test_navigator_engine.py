"""Engine-backed navigator export — the load-bearing battery (0.6.80).

Dataset + phonopy guarded (skips in a bare-zip audit, like test_engine_dq.py).
Inputs: the SnSe2 dataset's own SELF-CONSISTENT quadruple — band.hdf5 +
phonopy.yaml/FORCE_CONSTANTS (Stage 1 proved the FC reproduce that band file
to ~1e-10 THz) + a mesh.hdf5 generated ONCE from the same FC via phonopy
itself (the dataset ships only mesh.yaml; the exporter requires hdf5 — the
mesh is only the same-material gate in engine mode anyway).

What is proven:
- SINGLE-SOURCE INVARIANT (band-within-mesh, mechanized): independently
  re-evaluating D(q) reproduces the payload's band frequencies at every band
  q AND the payload's surface z at random in-BZ grid nodes, both < 1e-6 THz —
  band and surface provably come from ONE engine + ONE force-constant set.
- CROSS-FORMAT CONSISTENCY: payload provenance.engine == json_v1's
  _engine_block(EngineProvenance(...)) key-for-key, so the navigator payload
  and the JSON-v1 wire form cannot drift.
- STRICT FAIL MODES (negative controls): (a) perturbed FORCE_CONSTANTS ->
  the export hard-dies on the cross-check naming max|dω| (never emits);
  (b) phonopy forced-unavailable -> hard-dies naming the `engine` extra
  (never a silent legacy downgrade); (c) the invariant checker itself bites
  when a payload surface value is corrupted (fail-then-pass, in-suite).
"""
import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("matplotlib")
phonopy = pytest.importorskip("phonopy")

_HERE = os.path.dirname(__file__)
_DS = os.path.join(_HERE, os.pardir, os.pardir, "Phonopy-examples",
                   "fatbands_SnSe2")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(_DS),
    reason="Phonopy-examples dataset not present (bare-zip audit)")

_spec = importlib.util.spec_from_file_location(
    "pv_navigator_export",
    os.path.join(_HERE, os.pardir, "tools", "pv_navigator_export.py"))
nav = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nav)

_BAND = os.path.join(_DS, "band.hdf5")
_YAML = os.path.join(_DS, "phonopy.yaml")
_FC = os.path.join(_DS, "FORCE_CONSTANTS")
_GRID = 20
FREQ_TOL = 1e-6   # THz


@pytest.fixture(scope="module")
def eng():
    """The Stage-1 engine for the dataset (independent re-evaluation source)."""
    from phonon_core.engine import load_engine
    return load_engine(_YAML)


@pytest.fixture(scope="module")
def mesh_h5(eng, tmp_path_factory):
    """Self-consistent mesh.hdf5, generated once from the SAME FC."""
    out = str(tmp_path_factory.mktemp("mesh") / "mesh.hdf5")
    eng.run_mesh([6, 6, 1], is_mesh_symmetry=False, with_eigenvectors=True)
    eng.mesh.write_hdf5(filename=out)
    return out


@pytest.fixture(scope="module")
def payload(mesh_h5):
    """One engine-mode payload for the whole battery."""
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    return nav.build_payload(_BAND, mesh_h5, group_a=("Sn",), grid=_GRID,
                             engine=_YAML)


def _surface_nodes_check(payload, ph, rng_seed=7, n_nodes=25):
    """THE single-source surface invariant: pick random finite surface nodes,
    map (kx, ky) -> fractional q, independently evaluate D(q), compare to the
    payload's z. Returns max|dω| over the sampled nodes (all bands)."""
    from phonon_core.engine import evaluate_dq
    recip = np.asarray(payload["meta"]["material"]["reciprocal_lattice"])
    inv_recip = np.linalg.inv(recip)
    surf = nav.expand_surface(payload["surface"])   # 0.6.92: the ONE reader
    kx = surf["kx"]
    ky = surf["ky"]
    bands = surf["bands"]
    # 0.7.1 smooth rim: an outside cell's stored value lives at the cell's
    # Γ-ray boundary clamp, so the invariant evaluates THERE (the clamp is
    # identity for in-zone cells — this strengthens the check to cover the
    # rim, it never weakens it).
    poly = np.asarray(payload["bz_polygon"], dtype=float)
    # z[j][i] <-> (kx[i], ky[j])  (meshgrid 'xy': rows are the ky axis)
    finite = [(i, j) for j in range(len(ky)) for i in range(len(kx))
              if bands[0]["z"][j][i] is not None]
    rng = np.random.default_rng(rng_seed)
    picks = [finite[k] for k in rng.choice(len(finite), size=n_nodes,
                                           replace=False)]
    max_df = 0.0
    for (i, j) in picks:
        kc = nav._ray_to_hull(np.array([[kx[i], ky[j]]]), poly)[0]
        q_frac = np.array([kc[0], kc[1], 0.0]) @ inv_recip
        f, _v = evaluate_dq(ph, q_frac)
        z_payload = np.array([bands[b]["z"][j][i] for b in range(len(bands))],
                             dtype=np.float64)
        max_df = max(max_df, float(np.max(np.abs(f - z_payload))))
    return max_df


def test_single_source_invariant_band(payload, eng):
    """Independent re-evaluation reproduces the payload BAND frequencies at
    EVERY band q — the band block provably comes from this engine + FC."""
    from phonon_core.engine import evaluate_dq
    q_frac = np.asarray(payload["band"]["q_frac"])
    freq = np.asarray(payload["band"]["frequency"], dtype=np.float64)
    max_df = 0.0
    for qi in range(len(q_frac)):
        f, _v = evaluate_dq(eng, q_frac[qi])
        max_df = max(max_df, float(np.max(np.abs(f - freq[qi]))))
    print(f"\n[nav-engine] band invariant over {len(q_frac)} q: "
          f"max|dω| = {max_df:.3e} THz")
    assert max_df < FREQ_TOL


def test_single_source_invariant_surface(payload, eng):
    """Independent re-evaluation reproduces the payload SURFACE z at random
    in-BZ nodes — band + surface share ONE D(q): band-within-mesh, mechanized."""
    max_df = _surface_nodes_check(payload, eng)
    print(f"\n[nav-engine] surface invariant (25 random nodes, all bands): "
          f"max|dω| = {max_df:.3e} THz")
    assert max_df < FREQ_TOL


def test_cross_format_consistency(payload):
    """payload provenance.engine == json_v1's wire serializer output for the
    equivalent EngineProvenance — the two formats cannot drift."""
    import hashlib
    from phonon_core.io.json_v1 import EngineProvenance, _engine_block

    def _md5(p):
        with open(p, "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()

    ep = EngineProvenance(
        kind="phonopy-A1", evaluated_band=True, evaluated_mesh=True,
        phonopy_version=str(phonopy.__version__),
        fc_md5=_md5(_FC), phonopy_yaml_md5=_md5(_YAML))
    assert payload["provenance"]["engine"] == _engine_block(ep)
    # and the recorded cross-check is the Stage-1 reproduction level
    assert 0.0 <= payload["provenance"]["cross_check"]["band_max_dw_THz"] < FREQ_TOL


def test_negctl_perturbed_fc_dies(mesh_h5, tmp_path):
    """(a) A perturbed FORCE_CONSTANTS must HARD-DIE on the cross-check naming
    max|dω| — the FC<->band provenance mismatch is refused, never emitted."""
    d = tmp_path / "perturbed"
    d.mkdir()
    shutil.copy(_YAML, d / "phonopy.yaml")
    with open(_FC) as fh:
        lines = fh.readlines()
    out = []
    for ln in lines:
        parts = ln.split()
        if len(parts) == 3:
            try:
                out.append("  ".join(f"{float(x) * 1.02:.15f}" for x in parts)
                           + "\n")
                continue
            except ValueError:
                pass
        out.append(ln)
    (d / "FORCE_CONSTANTS").write_text("".join(out))

    with pytest.raises(SystemExit) as exc:
        nav.build_payload(_BAND, mesh_h5, group_a=("Sn",), grid=12,
                          engine=str(d / "phonopy.yaml"))
    assert exc.value.code == 3


def test_negctl_phonopy_unavailable_dies(mesh_h5, monkeypatch, capsys):
    """(b) --engine with phonopy forced-unavailable must HARD-DIE naming the
    `engine` extra — no silent downgrade to the legacy path."""
    monkeypatch.setitem(sys.modules, "phonopy", None)   # import -> ImportError
    with pytest.raises(SystemExit) as exc:
        nav.build_payload(_BAND, mesh_h5, group_a=("Sn",), grid=12,
                          engine=_YAML)
    assert exc.value.code == 2
    assert "phonon-core[engine]" in capsys.readouterr().err


def test_negctl_invariant_bites(payload, eng):
    """(c) fail-then-pass for the surface invariant: corrupting ONE expected
    surface node value in a COPY of the payload makes the invariant checker
    FAIL; the clean payload passes (test_single_source_invariant_surface)."""
    corrupted = json.loads(json.dumps(payload))
    # expand the compact block first (0.6.92) — the negctl exercises the
    # CHECKER's teeth on the dense view the checker actually reads
    corrupted["surface"] = nav.expand_surface(corrupted["surface"])
    bands = corrupted["surface"]["bands"]
    kx, ky = corrupted["surface"]["kx"], corrupted["surface"]["ky"]
    # corrupt the FIRST finite node of band 0 by 1 THz
    done = False
    for j in range(len(ky)):
        for i in range(len(kx)):
            if bands[0]["z"][j][i] is not None:
                bands[0]["z"][j][i] += 1.0
                done = True
                break
        if done:
            break
    assert done
    # deterministic bite: check exactly the corrupted node via the same math
    # the random-sample checker uses (no seed luck involved)
    from phonon_core.engine import evaluate_dq
    recip = np.asarray(corrupted["meta"]["material"]["reciprocal_lattice"])
    q_frac = np.array([kx[i], ky[j], 0.0]) @ np.linalg.inv(recip)
    f, _v = evaluate_dq(eng, q_frac)
    z_bad = np.array([bands[b]["z"][j][i] for b in range(len(bands))])
    df_bad = float(np.max(np.abs(f - z_bad)))
    print(f"\n[nav-engine] NEGCTL(c): corrupted node deviates by "
          f"{df_bad:.3f} THz (>= 1.0 injected) -> invariant BITES")
    assert df_bad >= 0.999            # the corruption is seen...
    with pytest.raises(AssertionError):
        assert df_bad < FREQ_TOL      # ...and the invariant assertion FAILS on it


# ---------------------------------------------------------------------------
# 0.6.81: mesh_modes — exact engine modes on an independent in-BZ grid
# ---------------------------------------------------------------------------
def _mesh_modes_agreement(payload, ph):
    """(max|dω|, min overlap, max projector dev) between the payload's
    mesh_modes block and a FRESH evaluate_dq at every stated q_frac, via the
    same degeneracy-aware comparison the exporter's cross-check gate uses.
    Canonicalization is a per-mode global phase and the overlap/projector
    metrics are phase-invariant, so fresh (un-canonicalized) engine output
    compares directly; 6-sig-fig rounding costs ~1e-13 in overlap."""
    from phonon_core.engine import evaluate_dq
    from phonon_core.engine.compare import mode_agreement
    mm = payload["mesh_modes"]
    q_frac = np.asarray(mm["q_frac"])
    f_pl = np.asarray(mm["frequency"], dtype=np.float64)          # (Nn, Nb)
    v_pl = (np.asarray(mm["eigenvectors_real"], dtype=np.float64)
            + 1j * np.asarray(mm["eigenvectors_imag"], dtype=np.float64))
    f_l, v_l = [], []
    for k in range(len(q_frac)):
        f, v = evaluate_dq(ph, q_frac[k])
        f_l.append(f)
        v_l.append(v)
    return mode_agreement(f_pl, v_pl, np.asarray(f_l), np.asarray(v_l))


def test_single_source_invariant_mesh_modes(payload, eng):
    """THE Stage-3 guarantee, mechanized: every mesh_modes entry is a REAL
    engine mode at its stated q — frequencies to full precision, eigenvectors
    phase-invariantly (projectors across degenerate subspaces)."""
    max_df, min_ov, max_pd = _mesh_modes_agreement(payload, eng)
    n = len(payload["mesh_modes"]["q_frac"])
    print(f"\n[nav-engine] mesh_modes invariant over {n} nodes: "
          f"max|dω| = {max_df:.3e} THz, min overlap = 1-{1 - min_ov:.2e}, "
          f"max projector dev = {max_pd:.3e}")
    assert max_df < FREQ_TOL
    assert min_ov > 1 - 1e-6
    assert max_pd < 1e-6


def test_mesh_modes_size_budget(payload):
    """The 6-sig-fig eigenvector policy keeps the block within budget."""
    blob = json.dumps(payload["mesh_modes"], allow_nan=False)
    n = len(payload["mesh_modes"]["q_frac"])
    print(f"\n[nav-engine] mesh_modes serialized: {len(blob):,} bytes "
          f"({n} nodes, grid {payload['mesh_modes']['grid']})")
    assert len(blob) < 2_000_000


def test_negctl_mesh_modes_eigvec_bites(payload, eng):
    """(a) fail-then-pass: corrupting ONE eigenvector component in a COPY
    makes the invariant's eigenvector check fail; the clean payload passes
    (test_single_source_invariant_mesh_modes)."""
    corrupted = json.loads(json.dumps(payload))
    corrupted["mesh_modes"]["eigenvectors_real"][0][0][0][0] += 0.5
    max_df, min_ov, max_pd = _mesh_modes_agreement(corrupted, eng)
    print(f"\n[nav-engine] NEGCTL(a): corrupted eigvec component -> "
          f"min overlap = {min_ov:.6f}, max projector dev = {max_pd:.3e} "
          f"-> eigenvector check BITES")
    assert (min_ov < 1 - 1e-6) or (max_pd > 1e-6)   # the corruption is seen...
    with pytest.raises(AssertionError):              # ...and the invariant FAILS
        assert min_ov > 1 - 1e-6 and max_pd < 1e-6


def test_negctl_mesh_modes_freq_bites(payload, eng):
    """(b) fail-then-pass: corrupting ONE frequency in a COPY makes the
    invariant's max|dω| check fail; the clean payload passes."""
    corrupted = json.loads(json.dumps(payload))
    corrupted["mesh_modes"]["frequency"][0][0] += 1.0
    max_df, _min_ov, _max_pd = _mesh_modes_agreement(corrupted, eng)
    print(f"\n[nav-engine] NEGCTL(b): corrupted frequency -> "
          f"max|dω| = {max_df:.3f} THz -> frequency check BITES")
    assert max_df >= 0.999                           # the corruption is seen...
    with pytest.raises(AssertionError):              # ...and the invariant FAILS
        assert max_df < FREQ_TOL


# ---------------------------------------------------------------------------
# 0.6.85: MESH-OPTIONAL engine mode — the phonopy.yaml (not a mesh file) is the
# material authority; the surface + mesh_modes are built from D(q) with no mesh.
# ---------------------------------------------------------------------------
_WSE2_YAML = os.path.join(_HERE, os.pardir, os.pardir, "Phonopy-examples",
                          "WSe2", "phonopy.yaml")


@pytest.fixture(scope="module")
def payload_nomesh():
    """Engine-mode payload built WITHOUT a mesh (0.6.85) — identical inputs to
    `payload` minus the mesh file."""
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    return nav.build_payload(_BAND, None, group_a=("Sn",), grid=_GRID,
                             engine=_YAML)


def test_engine_without_mesh_full_keyset(payload_nomesh, eng):
    """Engine mode WITHOUT a mesh still emits the FULL engine key set, the
    surface is engine-exact, and every block is a REAL engine evaluation —
    band + sampled surface nodes + mesh_modes all reproduce from a fresh D(q)
    with no mesh anywhere in the pipeline. (Fails on the pre-0.6.85 exporter,
    which omits the surface/mesh_modes blocks when no mesh is given.)"""
    from phonon_core.engine import evaluate_dq
    assert set(payload_nomesh) == {
        "meta", "band", "surface", "bz_polygon", "ibz_polygon", "z_range",
        "provenance", "mesh_modes"}
    assert payload_nomesh["provenance"]["surface"] == "engine-exact"
    q_frac = np.asarray(payload_nomesh["band"]["q_frac"])
    freq = np.asarray(payload_nomesh["band"]["frequency"], dtype=np.float64)
    band_df = max(
        float(np.max(np.abs(evaluate_dq(eng, q_frac[qi])[0] - freq[qi])))
        for qi in range(len(q_frac)))
    surf_df = _surface_nodes_check(payload_nomesh, eng)
    mm_df, mm_ov, mm_pd = _mesh_modes_agreement(payload_nomesh, eng)
    print(f"\n[nav-engine] NO-MESH invariants: band {band_df:.3e}, surface "
          f"{surf_df:.3e}, mesh_modes {mm_df:.3e} THz / overlap 1-{1 - mm_ov:.1e}")
    assert band_df < FREQ_TOL and surf_df < FREQ_TOL and mm_df < FREQ_TOL
    assert mm_ov > 1 - 1e-6 and mm_pd < 1e-6


@pytest.mark.skipif(not os.path.isfile(_WSE2_YAML),
                    reason="WSe2 dataset not present (material-mismatch fixture)")
def test_engine_material_mismatch_dies(capsys):
    """SnSe2 band + a DIFFERENT material's phonopy.yaml (WSe2: W,Se,Se + a
    different reciprocal lattice) -> _die exit 2 with the NAMED material
    message, BEFORE (and instead of) the max|dω| cross-check. Pre-0.6.85 had
    no such gate — the mismatch surfaced only as an exit-3 cross-check failure
    — so this test fails on the pre-edit exporter."""
    with pytest.raises(SystemExit) as exc:
        nav.build_payload(_BAND, None, group_a=("Sn",), grid=12,
                          engine=_WSE2_YAML)
    assert exc.value.code == 2
    assert "engine phonopy.yaml are not the same material" in capsys.readouterr().err


def test_engine_mesh_vs_nomesh_equivalent(payload, payload_nomesh):
    """WITH-mesh and WITHOUT-mesh engine payloads on the same inputs are
    identical EXCEPT the three meta.source mesh entries — the mesh contributes
    NO number to the physics. (Near-zero eigenvector components can differ at
    the ~1e-34 denormal level between the two independent engine passes under
    multi-threaded BLAS — verified: two no-mesh builds differ by the same
    amount, so it is LAPACK noise, not a mesh contribution. Eigenvector arrays
    are therefore compared at atol=1e-20 — 20 orders below the 6-sig-fig wire
    precision — while EVERYTHING else is asserted bitwise-identical.)"""
    a = json.loads(json.dumps(payload))          # WITH mesh
    b = json.loads(json.dumps(payload_nomesh))   # WITHOUT mesh
    # the source diff is EXACTLY the three mesh entries, nothing else
    assert set(a["meta"]["source"]) - set(b["meta"]["source"]) == {
        "mesh", "n_qpoints_mesh", "mesh_dims"}
    assert set(b["meta"]["source"]) - set(a["meta"]["source"]) == set()
    for k in ("mesh", "n_qpoints_mesh", "mesh_dims"):
        a["meta"]["source"].pop(k)

    def _strip_eig(doc):
        out = json.loads(json.dumps(doc))
        for blk in ("band", "mesh_modes"):
            for k in ("eigenvectors_real", "eigenvectors_imag"):
                out[blk].pop(k, None)
        return out

    # everything except the eigenvector arrays is bitwise-identical
    assert _strip_eig(a) == _strip_eig(b)
    # the eigenvectors match to 1e-20 (a real mesh contribution would appear at
    # the 6-sig-fig level, >= ~1e-6; only denormal BLAS noise is absorbed)
    maxd = 0.0
    for blk in ("band", "mesh_modes"):
        for k in ("eigenvectors_real", "eigenvectors_imag"):
            maxd = max(maxd, float(np.max(np.abs(
                np.asarray(a[blk][k]) - np.asarray(b[blk][k])))))
    print(f"\n[nav-engine] mesh vs no-mesh: non-eigvec content bitwise-equal; "
          f"max|Δ eigvec| = {maxd:.2e} (denormal noise; tol 1e-20)")
    assert maxd < 1e-20


# ---------------------------------------------------------------------------
# 0.6.92 — compact surface storage on REAL engine data
# ---------------------------------------------------------------------------
def test_surface_symmetry_verified_and_applied(payload):
    """On the real SnSe2 engine surface all three grid-exact mirror ops are
    VERIFIED by the field itself (P-3m1: the in-plane group contains both
    axis mirrors and C2 = time reversal), the measured invariance residual is
    orders of magnitude under the gate, and the dedup lands the full 4x."""
    sym = payload["surface"]["sym"]
    assert sorted(sym["ops"]) == ["c2", "mirror_x", "mirror_y"], sym["ops"]
    assert sym["max_dz_THz"] < 1e-9, sym          # physics symmetry, not tol-luck
    assert sym["nodes_stored"] * 4 >= sym["nodes_total"]      # exact orbits
    assert sym["nodes_stored"] * 3 < sym["nodes_total"]       # ~4x, not ~3x
    print(f"\n[nav-engine] 0.6.92 storage: {sym['nodes_total']} -> "
          f"{sym['nodes_stored']} nodes "
          f"({sym['nodes_total']/sym['nodes_stored']:.2f}x), "
          f"max|dz| = {sym['max_dz_THz']:.2e} THz, "
          f"max|dchar| = {sym['max_dchar']:.2e}")


def test_surface_expansion_matches_engine_everywhere(payload, eng):
    """The EXPANDED surface z equals an independent D(q) evaluation at every
    sampled node INCLUDING symmetry-reconstructed ones (not just stored
    representatives) — the reconstruction is physics, verified end-to-end.
    z is full precision (0.6.92 kept the 0.6.80 invariant's teeth)."""
    max_df = _surface_nodes_check(payload, eng, rng_seed=92, n_nodes=40)
    print(f"\n[nav-engine] 0.6.92 expansion invariant (40 nodes): "
          f"max|dω| = {max_df:.3e} THz")
    assert max_df < FREQ_TOL


def test_rim_clamp_matches_direct_engine_evaluation(payload, eng):
    """0.7.1 Tier 0 accuracy: a stored OUTSIDE-zone cell's value must equal a
    direct D(q) evaluation at the Γ-ray boundary clamp — the rim is exact,
    not resampled."""
    from matplotlib.path import Path as _Path
    from phonon_core.engine import evaluate_dq
    S = payload["surface"]
    kx, ky = np.array(S["kx"]), np.array(S["ky"])
    poly = np.array(payload["bz_polygon"])
    recip = np.asarray(payload["meta"]["material"]["reciprocal_lattice"])
    inv = np.linalg.inv(recip)
    dense = nav.expand_surface(S)
    KX, KY = np.meshgrid(kx, ky)
    inside = _Path(poly).contains_points(
        np.column_stack([KX.ravel(), KY.ravel()])).reshape(len(ky), len(kx))
    oi, oj = np.where(~inside)
    assert len(oi) > 0, "the padded bbox must have outside cells"
    rng = np.random.default_rng(71)
    picks = rng.choice(len(oi), size=6, replace=False)
    worst = 0.0
    for k in picks:
        i, j = int(oi[k]), int(oj[k])
        clamp = nav._ray_to_hull(np.array([[kx[j], ky[i]]]), poly)[0]
        qf = np.array([clamp[0], clamp[1], 0.0]) @ inv
        f, _v = evaluate_dq(eng, qf)
        stored = np.array([dense["bands"][b]["z"][i][j]
                           for b in range(len(dense["bands"]))], dtype=float)
        worst = max(worst, float(np.max(np.abs(f - stored))))
    print(f"\n[nav-engine] 0.7.1 rim: 6 outside cells, all bands: "
          f"max|dω| = {worst:.3e} THz")
    assert worst < 1e-6


def test_extended_zone_licence_omega_and_moduli_G_periodic(eng):
    """0.7.2 Tier 1 LICENCE (mirrors magnon-vision's
    test_energies_and_participation_are_G_periodic): D(q+G) = U D(q) U† is a
    unitary similarity, so ω_ν(q+G) == ω_ν(q) and |e_{s,ν}(q+G)|² ==
    |e_{s,ν}(q)|² exactly. The viewer's extended-zone tiling draws first-zone
    data at its translates — this test is what makes that honest. If it ever
    fails, the tiling is a lie."""
    from phonon_core.engine import evaluate_dq
    rng = np.random.default_rng(72)
    qs = np.column_stack([rng.uniform(-0.5, 0.5, (4, 2)), np.zeros(4)])
    worst_w = worst_m = 0.0
    for G in ([1, 0, 0], [0, 1, 0], [1, 1, 0], [-1, 2, 0]):
        for q in qs:
            f0, v0 = evaluate_dq(eng, q)
            f1, v1 = evaluate_dq(eng, q + np.asarray(G, dtype=float))
            worst_w = max(worst_w, float(np.max(np.abs(f1 - f0))))
            m0 = np.sum(np.abs(v0) ** 2, axis=-1)     # (Nb, Na) per-atom moduli
            m1 = np.sum(np.abs(v1) ** 2, axis=-1)
            worst_m = max(worst_m, float(np.max(np.abs(m1 - m0))))
    print(f"\n[nav-engine] 0.7.2 licence: max|Δω| = {worst_w:.2e} THz, "
          f"max|Δ|e|²| = {worst_m:.2e} over 4 G × 4 q")
    assert worst_w < 1e-9
    assert worst_m < 1e-9
