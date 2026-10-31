"""Proof-by-mechanism battery for the A1 D(q) engine (0.6.78, Level-2 Stage 1).

Runs against the phonopy datasets in ../../Phonopy-examples (a sibling of the
src tree, like the navigator_site real-payload probe in tests-js) — SKIPS from
a bare-zip audit where the datasets are absent, and skips entirely without
phonopy (the ``engine`` optional extra).

What is proven, per discovered material:

1. PATH reproduction — the engine evaluated at band.hdf5/band.yaml's own
   q-points reproduces phonopy's stored frequencies (< 1e-6 THz against
   float64-q hdf5 — measured ~1e-10; text-yaml references round their PRINTED
   q to 7 decimals while their frequencies were computed at the unrounded q,
   so there the honest bound is 2e-5 = |domega/dq| * 5e-8, and
   test_yaml_path_exact_q_reconstruction PINS that mechanism by rebuilding the
   exact float64 path and collapsing the error to ~1e-10). Eigenvectors match
   up to per-mode global phase (|<v_e|v_ref>| > 1 - 1e-6); DEGENERATE
   subspaces (|df| < 1e-4 THz) are compared as projectors, not individual
   vectors (a basis of a degenerate eigenspace is arbitrary).
2. GRID reproduction — same against mesh.yaml (frequencies always;
   eigenvectors where the file carries them; text-q tolerance).
3. ARBITRARY-q sanity — off-path/off-grid q: D(q) Hermitian; eigenvalues of
   the exposed D map to evaluate_dq's frequencies through phonopy's own
   unit factor (signed-sqrt); the eigenvector matrix is unitary; omega(q) is
   continuous along a fine off-symmetry segment; acoustic sum rule at Gamma
   (REFERENCE-AWARE: WSe2's raw FORCE_CONSTANTS genuinely violate the ASR by
   0.34 THz and its own band.hdf5 carries that violation — there the engine
   must reproduce the reference, not mask the dataset's property).
4. DIMENSION-AGNOSTIC — one evaluate_dq code path, parametrized over every
   discovered material; 3D bulk datasets join automatically when present
   under Phonopy-examples/.

ACKNOWLEDGED (task): A1 *is* phonopy, so 1./2. are near-exact by
construction; this stage pins extraction/shaping + arbitrary-q + sanity, and
establishes the oracle A2 (numpy D(q), Stage 4) must reproduce.
"""
from pathlib import Path

import numpy as np
import pytest

phonopy = pytest.importorskip("phonopy")

from phonon_core.engine import dynamical_matrix_at, evaluate_dq, load_engine

DATA = Path(__file__).resolve().parents[2] / "Phonopy-examples"

pytestmark = pytest.mark.skipif(
    not DATA.is_dir(),
    reason="Phonopy-examples datasets not present (bare-zip audit)")

FREQ_TOL = 1e-6      # THz — reproduction against float64 q (band.hdf5, or the
                     # exact reconstructed path); measured ~1e-10
# Text YAML stores q-position rounded to 7 DECIMALS while its frequencies were
# computed at the unrounded q, so evaluating at the stored q costs
# |domega/dq| * |dq| with |dq| <= 5e-8 — measured up to 8.7e-6 THz on
# graphene's steep bands, and PINNED as the mechanism by
# test_yaml_path_exact_q_reconstruction (exact float64 path q -> 5.7e-10 THz).
FREQ_TOL_TEXT = 2e-5
OVERLAP_TOL = 1e-6   # 1 - |<v_e|v_ref>| for non-degenerate modes
PROJ_TOL = 1e-6      # max|P_e - P_ref| for degenerate subspaces
DEG_TOL = 1e-4       # THz — modes closer than this are one subspace
ASR_TOL = 2e-3       # THz — numerical-dust level at Gamma (task: ~1e-3 order;
                     # measured: SnSe2 2.8e-4, graphene 9.8e-4)
HERM_TOL = 1e-8      # max|D - D^H|
STEP_TOL = 0.25      # THz — max per-step jump on the 401-pt continuity segment
                     # (measured: SnSe2 0.019, WSe2 0.021, graphene 0.123)


def _discover():
    """Every Phonopy-examples subdir with phonopy.yaml + FORCE_CONSTANTS.
    Reference files are discovered, not hardcoded (WSe2 prefixes its band
    files); band.hdf5 preferred over band.yaml (float64-exact)."""
    mats = []
    if not DATA.is_dir():
        return mats
    for d in sorted(DATA.iterdir()):
        if not ((d / "phonopy.yaml").is_file()
                and (d / "FORCE_CONSTANTS").is_file()):
            continue
        band_h5 = sorted(d.glob("*band*.hdf5"))
        band_yl = sorted(d.glob("*band*.yaml"))
        band = band_h5[0] if band_h5 else (band_yl[0] if band_yl else None)
        mesh = d / "mesh.yaml"
        mats.append({"name": d.name, "dir": d, "band": band,
                     "mesh": mesh if mesh.is_file() else None})
    return mats


MATERIALS = _discover()
_IDS = [m["name"] for m in MATERIALS]
_CACHE: dict = {}



@pytest.fixture(scope="module", params=MATERIALS, ids=_IDS)
def mat(request):
    """Material bundle: engine + parsed references, loaded once per material.
    ``band_is_text`` records whether the band reference stores q rounded to
    text precision (band.yaml) or exact float64 (band.hdf5) — it selects the
    frequency tolerance (see FREQ_TOL_TEXT)."""
    m = request.param
    name = m["name"]
    if name not in _CACHE:
        bundle = {"name": name, "band": None, "mesh": None,
                  "band_is_text": False}
        bundle["ph"] = load_engine(str(m["dir"] / "phonopy.yaml"))
        if m["band"] is not None:
            if m["band"].suffix == ".hdf5":
                pytest.importorskip("h5py")
                from phonon_core.parsers import load_band_hdf5
                bundle["band"] = load_band_hdf5(str(m["band"]))
            else:
                from phonon_core.parsers import load_band_yaml
                bundle["band"] = load_band_yaml(str(m["band"]))
                bundle["band_is_text"] = True
        if m["mesh"] is not None:
            from phonon_core.parsers import load_mesh_yaml
            bundle["mesh"] = load_mesh_yaml(str(m["mesh"]), eigenvectors=True)
        _CACHE[name] = bundle
    return _CACHE[name]


def _deg_groups(freqs, tol=DEG_TOL):
    """Chain consecutive modes with |df| < tol into one degenerate subspace
    (freqs ascending, as phonopy and the reference files store them)."""
    groups, start = [], 0
    for i in range(1, len(freqs) + 1):
        if i == len(freqs) or abs(freqs[i] - freqs[i - 1]) > tol:
            groups.append(list(range(start, i)))
            start = i
    return groups


def _compare_modes(f_ref, v_ref, f_eng, v_eng):
    """(max|df|, min singleton overlap, max projector deviation) for one q.

    v_* : (N_b, N_a, 3) complex. Overlap is phase-invariant; degenerate
    subspaces compare sum_b |v_b><v_b| projectors (basis-invariant)."""
    df = float(np.max(np.abs(f_eng - f_ref)))
    A = v_ref.reshape(v_ref.shape[0], -1)
    B = v_eng.reshape(v_eng.shape[0], -1)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    min_ov, max_pd = 1.0, 0.0
    for grp in _deg_groups(f_ref):
        if len(grp) == 1:
            b = grp[0]
            min_ov = min(min_ov, float(abs(np.vdot(B[b], A[b]))))
        else:
            P_ref = sum(np.outer(A[b], A[b].conj()) for b in grp)
            P_eng = sum(np.outer(B[b], B[b].conj()) for b in grp)
            max_pd = max(max_pd, float(np.max(np.abs(P_eng - P_ref))))
    return df, min_ov, max_pd


def _reproduce(ph, q_points, f_ref_all, v_ref_all):
    """Run the engine over a reference set; return worst-case stats."""
    max_df, min_ov, max_pd = 0.0, 1.0, 0.0
    for qi in range(len(q_points)):
        f_eng, v_eng = evaluate_dq(ph, q_points[qi])
        if v_ref_all is None:
            max_df = max(max_df, float(np.max(np.abs(f_eng - f_ref_all[qi]))))
            continue
        df, ov, pd = _compare_modes(f_ref_all[qi], v_ref_all[qi], f_eng, v_eng)
        max_df, min_ov, max_pd = max(max_df, df), min(min_ov, ov), max(max_pd, pd)
    return max_df, min_ov, max_pd


# --- 1. PATH reproduction ----------------------------------------------------
def test_path_reproduction(mat):
    band = mat["band"]
    if band is None:
        pytest.skip(f"{mat['name']}: no band reference file")
    tol = FREQ_TOL_TEXT if mat["band_is_text"] else FREQ_TOL
    max_df, min_ov, max_pd = _reproduce(
        mat["ph"], band.q_points, band.frequencies, band.eigenvectors)
    print(f"\n[engine] {mat['name']} PATH ({len(band.q_points)} q, "
          f"{'text' if mat['band_is_text'] else 'float64'} q): "
          f"max|dω| = {max_df:.3e} THz (tol {tol:.0e}), "
          f"min overlap = 1-{1 - min_ov:.2e}, max projector dev = {max_pd:.3e}")
    assert max_df < tol
    assert min_ov > 1 - OVERLAP_TOL
    assert max_pd < PROJ_TOL


def test_yaml_path_exact_q_reconstruction(mat):
    """THE q-rounding mechanism pin: a text band.yaml stores q to 7 decimals
    while its frequencies were computed at the unrounded q. Rebuilding the
    exact float64 path (per-segment linspace between the segment endpoints —
    exactly how phonopy's CLI builds BAND with BAND_POINTS) must collapse the
    frequency error to the float64 level, proving the FREQ_TOL_TEXT slack is
    the reference file's print precision, NOT engine error."""
    band = mat["band"]
    if band is None or not mat["band_is_text"]:
        pytest.skip(f"{mat['name']}: no text band.yaml reference")
    qs, start = [], 0
    for n in band.segment_nqpoint:
        seg = band.q_points[start:start + n]
        qs.append(np.linspace(seg[0], seg[-1], n))   # endpoints print exactly
        start += n
    qs = np.concatenate(qs)
    max_df = 0.0
    for qi in range(len(qs)):
        f_eng, _ = evaluate_dq(mat["ph"], qs[qi])
        max_df = max(max_df, float(np.max(np.abs(f_eng - band.frequencies[qi]))))
    print(f"\n[engine] {mat['name']} PATH at EXACT reconstructed q: "
          f"max|dω| = {max_df:.3e} THz (vs {FREQ_TOL:.0e})")
    assert max_df < FREQ_TOL


# --- 2. GRID reproduction ----------------------------------------------------
def test_grid_reproduction(mat):
    mesh = mat["mesh"]
    if mesh is None:
        pytest.skip(f"{mat['name']}: no mesh.yaml")
    # mesh.yaml is always text -> q rounded at 7 decimals (see FREQ_TOL_TEXT)
    max_df, min_ov, max_pd = _reproduce(
        mat["ph"], mesh.q_points, mesh.frequencies, mesh.eigenvectors)
    has_eig = mesh.eigenvectors is not None
    print(f"\n[engine] {mat['name']} GRID ({len(mesh.q_points)} q, "
          f"eigenvectors {'yes' if has_eig else 'NO'}): "
          f"max|dω| = {max_df:.3e} THz (tol {FREQ_TOL_TEXT:.0e}), "
          f"min overlap = 1-{1 - min_ov:.2e}, "
          f"max projector dev = {max_pd:.3e}")
    assert max_df < FREQ_TOL_TEXT
    if has_eig:
        assert min_ov > 1 - OVERLAP_TOL
        assert max_pd < PROJ_TOL


# --- 3. ARBITRARY-q sanity ---------------------------------------------------
# Off-path, off-grid probes (the 91x91 grids and the G-M-K-G path never hit
# these); z=0 keeps them in the physically meaningful plane for these layered
# materials but nothing in the engine knows or cares.
_OFF_QS = [(0.123456, 0.054321, 0.0), (0.311, 0.207, 0.0),
           (0.048, 0.443, 0.0), (0.271828, 0.141421, 0.0)]


def test_offgrid_hermiticity_and_consistency(mat):
    ph = mat["ph"]
    factor = float(ph.unit_conversion_factor)   # phonopy's own THz factor
    for q in _OFF_QS:
        D = dynamical_matrix_at(ph, q)
        herm = float(np.max(np.abs(D - D.conj().T)))
        assert herm < HERM_TOL, f"q={q}: max|D - D^H| = {herm:.2e}"
        # the exposed D is the SAME object behind evaluate_dq: its eigenvalues
        # map to the returned frequencies through phonopy's signed-sqrt factor
        lam = np.linalg.eigvalsh((D + D.conj().T) / 2.0)
        f_from_D = np.sign(lam) * np.sqrt(np.abs(lam)) * factor
        f_eng, v_eng = evaluate_dq(ph, q)
        assert float(np.max(np.abs(np.sort(f_from_D) - f_eng))) < FREQ_TOL
        # frequencies ascending, finite; eigenvector matrix unitary
        assert np.all(np.isfinite(f_eng)) and np.all(np.diff(f_eng) >= -1e-12)
        V = v_eng.reshape(v_eng.shape[0], -1)
        gram = V.conj() @ V.T
        assert float(np.max(np.abs(gram - np.eye(len(V))))) < 1e-8


def test_gamma_acoustic_sum_rule(mat):
    """REFERENCE-AWARE ASR: where the dataset's own reference is at numerical
    dust at Gamma, the engine must be too; where the dataset's raw
    FORCE_CONSTANTS violate the ASR (WSe2: 0.34 THz — its band.hdf5 carries
    the same values), the engine must faithfully REPRODUCE the reference
    rather than mask the dataset's property."""
    ph, band = mat["ph"], mat["band"]
    f_eng, _ = evaluate_dq(ph, (0.0, 0.0, 0.0))
    dust_eng = np.sort(np.abs(f_eng))[:3]
    ref_row = None
    if band is not None:
        on_gamma = np.where(np.all(np.abs(band.q_points) < 1e-12, axis=1))[0]
        if len(on_gamma):
            ref_row = band.frequencies[on_gamma[0]]
    if ref_row is not None and np.sort(np.abs(ref_row))[:3].max() > ASR_TOL:
        dmax = float(np.max(np.abs(f_eng - ref_row)))
        print(f"\n[engine] {mat['name']} GAMMA: dataset FC violates the ASR "
              f"(reference acoustic |ω| up to "
              f"{np.sort(np.abs(ref_row))[2]:.3e} THz) — engine reproduces "
              f"the reference to {dmax:.3e} THz")
        assert dmax < FREQ_TOL
    else:
        print(f"\n[engine] {mat['name']} GAMMA acoustic |ω| = "
              f"{dust_eng} THz (dust bound {ASR_TOL})")
        assert dust_eng.max() < ASR_TOL


def test_continuity_along_offsymmetry_segment(mat):
    """omega(q) along a fine synthetic segment between two OFF-symmetry
    points: finite everywhere, no spurious jumps (sorted eigenvalues of a
    continuous Hermitian family are continuous)."""
    ph = mat["ph"]
    qa = np.array([0.07, 0.03, 0.0])
    qb = np.array([0.41, 0.27, 0.0])
    ts = np.linspace(0.0, 1.0, 401)
    F = np.array([evaluate_dq(ph, (1 - t) * qa + t * qb)[0] for t in ts])
    assert np.all(np.isfinite(F))
    step = float(np.max(np.abs(np.diff(F, axis=0))))
    print(f"\n[engine] {mat['name']} CONTINUITY (401 pts): "
          f"max per-step |dω| = {step:.4f} THz (bound {STEP_TOL})")
    assert step < STEP_TOL
