"""A2 (pure-numpy D(q) engine) oracle battery — Level-2 Stage 4 (0.6.86).

Three tiers, guarded so the file NEVER hard-fails on missing optionals:

- Tier 1 (ALWAYS runs — numpy + the package's own hard deps only): invariants
  on a hand-built, translationally consistent toy (no phonopy, no files);
  FORCE_CONSTANTS text parser round-trips; compact-vs-full equivalence;
  batch == loop-of-singles BITWISE; the BORN refusal; the no-phonopy-import
  grep guard. NOTE: deliberately NO module-level ``pytest.importorskip`` —
  that would skip Tier 1 too.
- Tier 2 (phonopy present — THE ORACLE): seeded random translationally
  consistent toys built IN-TEST via phonopy (3D rocksalt-like 2x2x2 and 2D
  hex slab 3x3x1 — the audit instruments), saved as phonopy.yaml
  (settings={'force_constants': False}) + FORCE_CONSTANTS; A2 vs A1 at a
  40-pt synthetic path, a 125-point random cloud, Gamma, and zone-boundary
  q with guaranteed multiplicity ties. |dnu| < 1e-8 THz; overlap
  > 1 - 1e-10 (non-degenerate); degenerate projector < 1e-8. The signed
  convention is asserted on real negative frequencies.
- Tier 3 (Phonopy-examples dataset present): A2 vs A1 on the REAL SnSe2
  data_task quadruple (band-path q + random in-BZ cloud) and A2 vs the
  band.hdf5 FILE directly — transitively closing A2 -> A1 -> file.

NEGATIVE CONTROLS (each must BITE, fail-then-pass in-suite):
(a) gauge flip — drop the (tau_j - tau_i) offset from the phase via a
    rebuilt svecs table: frequencies stay ~0 (gauge-invariant similarity
    transform) while eigenvector overlap COLLAPSES at generic q. This is
    the per-atom-phase trap the battery exists to catch: a frequency-only
    check would pass a gauge-broken engine.
(b) tie truncation — keep only the FIRST shortest vector per pair: |dnu|
    blows up where tie partners carry DIFFERENT phases. Sharper pin than
    "zone boundary": at SUPERCELL-COMMENSURATE q the tie partners differ by
    supercell lattice vectors, q . dv is an integer, the phases are EQUAL
    and truncation is invisible (asserted!); the blow-up shows at
    incommensurate q — including the hex toy's zone boundary (1/2, 0, 0),
    which is incommensurate with its 3x3x1 supercell.
(c) corrupt one FC block ON DISK -> the Tier-2 oracle fails loudly.
"""
import dataclasses
import importlib.util
import os
import shutil

import numpy as np
import pytest

from phonon_core.types import Material
from phonon_core.engine.compare import mode_agreement
from phonon_core.engine import dq_numpy
from phonon_core.engine.dq_numpy import (
    NumpyEngine,
    build_shortest_vectors,
    evaluate_dq_np,
    evaluate_dq_np_batch,
    dynamical_matrix_at_np,
    load_engine_np,
    parse_force_constants_text,
)

_HAVE_PHONOPY = importlib.util.find_spec("phonopy") is not None
_HAVE_H5PY = importlib.util.find_spec("h5py") is not None
_HERE = os.path.dirname(__file__)
_DS = os.path.join(_HERE, os.pardir, os.pardir, "Phonopy-examples",
                   "fatbands_SnSe2", "data_task")

tier2 = pytest.mark.skipif(not _HAVE_PHONOPY,
                           reason="phonopy not present (bare-zip audit) — "
                                  "Tier-2 oracle skipped")
tier3 = pytest.mark.skipif(not os.path.isdir(_DS),
                           reason="Phonopy-examples data_task not present "
                                  "(bare-zip audit) — Tier-3 skipped")


FREQ_TOL = 1e-8          # THz — A2 vs A1 (task-pinned)
OV_TOL = 1e-10           # non-degenerate overlap deficit
PROJ_TOL = 1e-8          # degenerate-subspace projector deviation

# METRIC DOCTRINE (architect, 0.6.86 audit) — moved to the SHARED helper
# tests/engine_metric.py at 0.6.88 (a second file needed it; one spelling).
from engine_metric import NU_FLOOR, LAM_TOL, doctrine_agreement  # noqa: E402


# ===========================================================================
# Tier 1 — numpy only (always runs)
# ===========================================================================
def _hand_toy(seed=5):
    """A translationally consistent 2-atom cubic toy on a 2x2x2 supercell,
    built WITHOUT phonopy and WITHOUT files.

    FC construction: for each supercell pair, the block is the AVERAGE of the
    isotropic-spring kernel g(v) = f1(|v|) I + f2(|v|) vhat vhat^T over ALL
    shortest (minimum-image) vectors of the pair. g(-v) = g(v) = g(v)^T and
    the shortest SET is supercell-translation invariant, so the FC is
    translationally consistent and Hermitian-paired by construction. The
    self-block then enforces the acoustic sum rule exactly:
    Phi(i, i) -= sum_j Phi(i, j).
    """
    plat = 3.1 * np.eye(3)
    tau = np.array([[0.0, 0.0, 0.0], [0.4, 0.55, 0.35]])
    masses = np.array([10.5, 27.3])
    n_rep = 2
    slat = n_rep * plat
    # phonopy supercell order: atom-major (all images of atom 0, then atom 1)
    cells = np.array([[i, j, k] for i in range(n_rep)
                      for j in range(n_rep) for k in range(n_rep)],
                     dtype=np.float64)
    sfrac = np.concatenate([(tau[a] + cells) / n_rep for a in range(2)])
    n_cell = len(cells)
    n_s = 2 * n_cell
    r_s = sfrac @ slat
    p2s = np.array([0, n_cell], dtype=np.int64)
    sp2p = np.repeat(np.arange(2), n_cell).astype(np.int64)

    # minimum-image shortest sets for EVERY supercell pair (shell 1 suffices
    # for this geometry; the engine's own table is validated elsewhere)
    rng3 = np.arange(-1, 2)
    L = np.stack(np.meshgrid(rng3, rng3, rng3, indexing="ij"),
                 axis=-1).reshape(-1, 3) @ slat
    d = r_s[None, :, None, :] - r_s[:, None, None, :] + L[None, None, :, :]
    dist = np.linalg.norm(d, axis=-1)
    dmin = dist.min(axis=-1)
    tie = dist <= dmin[..., None] * (1 + 1e-5) + 1e-12

    def g(v):
        """Isotropic-spring kernel with the CONVENTIONAL sign (negative
        off-diagonal coupling), so the ASR self-blocks come out positive and
        the toy is dynamically STABLE (real optical branches; the three
        acoustic zeros are then the FIRST ascending modes at Gamma)."""
        r = np.linalg.norm(v)
        if r < 1e-12:
            return 1.7 * np.eye(3)           # overwritten by the ASR anyway
        vhat = v / r
        return -(np.exp(-r / 2.0) * np.eye(3)
                 + 0.7 * np.exp(-r / 3.0) * np.outer(vhat, vhat))

    fc_full = np.zeros((n_s, n_s, 3, 3))
    for a in range(n_s):
        for b in range(n_s):
            vs = d[a, b][tie[a, b]]
            fc_full[a, b] = np.mean([g(v) for v in vs], axis=0)
    for a in range(n_s):                      # exact acoustic sum rule
        fc_full[a, a] -= fc_full[a].sum(axis=0)

    svecs, counts, offsets = build_shortest_vectors(slat, plat, r_s, p2s)
    material = Material(lattice=plat,
                        reciprocal_lattice=np.linalg.inv(plat).T,
                        symbols=["X", "Y"], masses=masses,
                        fractional_coords=tau)
    eng = NumpyEngine(plat=plat, masses=masses, p2s=p2s, sp2p=sp2p,
                      fc=fc_full[p2s], svecs=svecs, counts=counts,
                      offsets=offsets, material=material)
    return eng, fc_full


@pytest.fixture(scope="module")
def hand_toy():
    return _hand_toy()


def test_no_phonopy_import_guard():
    """THE VENDORING CONTRACT: zero phonopy imports in dq_numpy.py — checked
    on the raw source text (comments included) so it cannot regress."""
    with open(dq_numpy.__file__, encoding="utf-8") as fh:
        src = fh.read()
    assert "import phonopy" not in src
    assert "from phonopy" not in src


def test_hermiticity_random_q(hand_toy):
    eng, _ = hand_toy
    rng = np.random.default_rng(3)
    for q in rng.uniform(-0.7, 0.7, (10, 3)):
        D = dynamical_matrix_at_np(eng, q)
        assert np.max(np.abs(D - D.conj().T)) < 1e-12


def test_asr_gamma_acoustic_zero(hand_toy):
    """The ASR-enforced toy has EXACTLY three ~0 THz acoustic modes at Gamma
    (imposed sum rule -> translational invariance -> zero acoustic)."""
    eng, _ = hand_toy
    f, _v = evaluate_dq_np(eng, [0.0, 0.0, 0.0])
    assert np.max(np.abs(f[:3])) < 1e-5     # THz (measured ~1e-7)
    assert f[3] > 0.1                        # optical gap is real


def test_batch_equals_singles_bitwise(hand_toy):
    """PINNED: the batch is BITWISE the loop of singles — no BLAS-blocking
    or contraction-order dependence on Nq (engineered via fixed-order
    einsum; the module comments name the trap)."""
    eng, _ = hand_toy
    rng = np.random.default_rng(4)
    qs = np.vstack([np.zeros(3), [[0.5, 0, 0]], rng.uniform(-0.5, 0.5, (37, 3))])
    fb, vb = evaluate_dq_np_batch(eng, qs)
    for k in range(len(qs)):
        fs, vs = evaluate_dq_np(eng, qs[k])
        assert np.array_equal(fb[k], fs)
        assert np.array_equal(vb[k], vs)


def test_continuity_fine_segment(hand_toy):
    eng, _ = hand_toy
    seg = np.linspace([0, 0, 0], [0.5, 0.5, 0.0], 400)
    f, _v = evaluate_dq_np_batch(eng, seg)
    assert float(np.max(np.abs(np.diff(f, axis=0)))) < 1.0   # THz per step


def test_fc_text_roundtrip(tmp_path, hand_toy):
    """Write compact AND full FORCE_CONSTANTS text of the SAME physics;
    parse both; identical arrays after compaction and IDENTICAL D."""
    eng, fc_full = hand_toy
    n_s = fc_full.shape[0]

    def write_fc(path, fc):
        with open(path, "w") as fh:
            fh.write(f"{fc.shape[0]} {fc.shape[1]}\n")
            for i in range(fc.shape[0]):
                for j in range(fc.shape[1]):
                    fh.write(f"{i + 1} {j + 1}\n")
                    for row in fc[i, j]:
                        fh.write("  ".join(f"{x:.15f}" for x in row) + "\n")

    p_full = tmp_path / "FC_FULL"
    p_comp = tmp_path / "FC_COMPACT"
    write_fc(p_full, fc_full)
    write_fc(p_comp, fc_full[eng.p2s])
    got_full = parse_force_constants_text(str(p_full))
    got_comp = parse_force_constants_text(str(p_comp))
    want = np.round(fc_full, 15)
    assert np.allclose(got_full, fc_full, atol=1e-14)
    assert np.allclose(got_comp, fc_full[eng.p2s], atol=1e-14)
    # compact vs full of the same physics -> the SAME compact rows -> D
    # bitwise identical
    e_full = dataclasses.replace(eng, fc=dq_numpy._to_compact(
        got_full, eng.p2s, n_s))
    e_comp = dataclasses.replace(eng, fc=dq_numpy._to_compact(
        got_comp, eng.p2s, n_s))
    q = np.array([0.21, 0.37, 0.11])
    assert np.array_equal(dynamical_matrix_at_np(e_full, q),
                          dynamical_matrix_at_np(e_comp, q))


def test_fc_text_corruption_detected(tmp_path):
    (tmp_path / "FC").write_text("2 2\n1 1\n1 0 0\n")   # truncated
    with pytest.raises(ValueError, match="token count"):
        parse_force_constants_text(str(tmp_path / "FC"))


def test_born_refusal(tmp_path):
    """A sibling BORN file is a HARD refusal (A2 has no NAC) — checked
    before any parsing, so dummy files suffice."""
    (tmp_path / "phonopy.yaml").write_text("dummy: true\n")
    (tmp_path / "FORCE_CONSTANTS").write_text("1 1\n1 1\n0 0 0\n0 0 0\n0 0 0\n")
    (tmp_path / "BORN").write_text("1.0\n")
    with pytest.raises(RuntimeError, match="non-analytic.*A1"):
        load_engine_np(str(tmp_path / "phonopy.yaml"))


# ===========================================================================
# Tier 2 — the phonopy ORACLE (seeded random toys built in-test)
# ===========================================================================
def _build_phonopy_toy(dirpath, kind, seed):
    """Seeded random-FC toy saved as phonopy.yaml + full FORCE_CONSTANTS."""
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.file_IO import write_FORCE_CONSTANTS

    if kind == "rock3d":        # rocksalt-like fcc primitive, 2x2x2
        cell = PhonopyAtoms(
            symbols=["Na", "Cl"],
            cell=np.array([[0.0, 2.85, 2.85],
                           [2.85, 0.0, 2.85],
                           [2.85, 2.85, 0.0]]),
            scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
            masses=[22.99, 35.45])
        smat = np.diag([2, 2, 2])
    elif kind == "hex2d":       # hex slab with vacuum, 3x3x1
        cell = PhonopyAtoms(
            symbols=["B", "N"],
            cell=np.array([[2.5, 0.0, 0.0],
                           [-1.25, 2.165063509461097, 0.0],
                           [0.0, 0.0, 15.0]]),
            scaled_positions=[[0, 0, 0.5], [1 / 3, 2 / 3, 0.5]],
            masses=[10.81, 14.007])
        smat = np.diag([3, 3, 1])
    else:                       # pragma: no cover
        raise ValueError(kind)
    ph = Phonopy(cell, supercell_matrix=smat,
                 primitive_matrix=np.eye(3), log_level=0)
    n_s = len(ph.supercell)
    rng = np.random.default_rng(seed)
    fc = rng.uniform(-1.0, 1.0, (n_s, n_s, 3, 3))
    ph.force_constants = fc
    ph.save(os.path.join(dirpath, "phonopy.yaml"),
            settings={"force_constants": False})
    write_FORCE_CONSTANTS(fc, filename=os.path.join(dirpath,
                                                    "FORCE_CONSTANTS"))
    return os.path.join(dirpath, "phonopy.yaml")


def _toy_qs(rng_seed=7):
    """Gamma + the zone-boundary tie set + a random cloud + a 40-pt path."""
    rng = np.random.default_rng(rng_seed)
    path = np.vstack([np.linspace([0, 0, 0], [0.5, 0, 0], 20),
                      np.linspace([0.5, 0, 0], [0.5, 0.5, 0], 20)])
    zb = np.array([[0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0],
                   [0.5, 0.5, 0.5]])
    return np.vstack([np.zeros((1, 3)), zb,
                      rng.uniform(-0.5, 0.5, (125, 3)), path])


@pytest.fixture(scope="module", params=["rock3d", "hex2d"])
def toy(request, tmp_path_factory):
    """(name, A1 Phonopy object, A2 NumpyEngine, dataset dir)."""
    from phonon_core.engine import load_engine
    d = str(tmp_path_factory.mktemp(f"a2toy_{request.param}"))
    yaml = _build_phonopy_toy(d, request.param, seed=42 if
                              request.param == "rock3d" else 43)
    return request.param, load_engine(yaml), load_engine_np(yaml), d


def _a1_modes(a1, qs):
    from phonon_core.engine import evaluate_dq
    f0, v0 = evaluate_dq(a1, qs[0])
    f = np.empty((len(qs),) + f0.shape)
    v = np.empty((len(qs),) + v0.shape, dtype=np.complex128)
    f[0], v[0] = f0, v0
    for k in range(1, len(qs)):
        f[k], v[k] = evaluate_dq(a1, qs[k])
    return f, v


@tier2
def test_oracle_toys(toy):
    """THE ORACLE: A2 == A1 on 170 q (path + cloud + Gamma + tie-carrying
    zone boundary) at 1e-8 THz / 1e-10 overlap / 1e-8 projector; ties are
    genuinely present; the signed convention is exercised by REAL negative
    frequencies and compared WITH sign."""
    name, a1, a2, _d = toy
    assert int(a2.counts.max()) > 1          # guaranteed multiplicity ties
    qs = _toy_qs()
    f1, v1 = _a1_modes(a1, qs)
    f2, v2 = evaluate_dq_np_batch(a2, qs)
    df, ov, pd = mode_agreement(f1, v1, f2, v2)
    print(f"\n[a2-oracle:{name}] {len(qs)} q: max|dv| = {df:.3e} THz, "
          f"overlap deficit = {1 - ov:.3e}, proj dev = {pd:.3e}, "
          f"min freq = {f1.min():.2f} THz")
    assert df < FREQ_TOL
    assert ov > 1 - OV_TOL
    assert pd < PROJ_TOL
    # signed-sqrt convention on real negative frequencies, matched WITH sign
    assert (f1 < -0.5).any()
    neg = f1 < -0.5
    assert np.max(np.abs(f2[neg] - f1[neg])) < FREQ_TOL


def _flip_gauge(eng):
    """Test-only hook for negctl (a): subtract (tau_j - tau_i) from every
    shortest vector -> the LATTICE gauge (phases lose the atom offsets)."""
    tau = np.asarray(eng.material.fractional_coords)
    pj = np.repeat(eng.sp2p[:, None], eng.n_p, axis=1)
    pi = np.repeat(np.arange(eng.n_p)[None, :], eng.n_s, axis=0)
    dtau = np.repeat((tau[pj] - tau[pi]).reshape(-1, 3),
                     eng.counts.reshape(-1), axis=0)
    return dataclasses.replace(eng, svecs=eng.svecs - dtau)


@tier2
def test_negctl_gauge_flip_bites(toy):
    """(a) THE PER-ATOM-PHASE TRAP: the lattice-gauge engine keeps the
    frequencies (similarity transform — asserted!) while the eigenvector
    overlap COLLAPSES at generic q. A frequency-only battery would pass a
    gauge-broken engine; this is why eigenvectors are in the oracle."""
    name, a1, a2, _d = toy
    q = np.array([0.1234, 0.2345, 0.0567])
    from phonon_core.engine import evaluate_dq
    f1, v1 = evaluate_dq(a1, q)
    flipped = _flip_gauge(a2)
    ff, vf = evaluate_dq_np(flipped, q)
    f2, v2 = evaluate_dq_np(a2, q)
    assert np.max(np.abs(ff - f2)) < 1e-8            # gauge-invariant freqs
    _df, ov_clean, _pd = mode_agreement(f1[None], v1[None], f2[None], v2[None])
    _df, ov_flip, _pd = mode_agreement(f1[None], v1[None], ff[None], vf[None])
    print(f"\n[a2-negctl-a:{name}] overlap clean = 1-{1 - ov_clean:.1e}, "
          f"flipped = {ov_flip:.4f} -> the gauge trap BITES")
    assert ov_clean > 1 - OV_TOL                     # ...clean passes
    assert ov_flip < 0.999                           # ...flipped collapses
    with pytest.raises(AssertionError):              # fail-then-pass, in-suite
        assert ov_flip > 1 - OV_TOL


def _truncate_ties(eng):
    """Test-only hook for negctl (b): keep only the FIRST shortest vector."""
    firsts = eng.svecs[eng.offsets.reshape(-1)]
    return dataclasses.replace(
        eng, svecs=firsts, counts=np.ones_like(eng.counts),
        offsets=np.arange(eng.counts.size).reshape(eng.counts.shape))


@tier2
def test_negctl_tie_truncation_bites(toy):
    """(b) tie semantics: truncation blows up |dnu| at INCOMMENSURATE q
    (including the hex toy's zone boundary, incommensurate with 3x3x1) and
    is provably INVISIBLE at supercell-commensurate q (tie partners differ
    by supercell lattice vectors -> equal phases) — both directions pinned."""
    name, a1, a2, _d = toy
    from phonon_core.engine import evaluate_dq
    trunc = _truncate_ties(a2)

    q_gen = np.array([0.1234, 0.2345, 0.0567])       # incommensurate, generic
    f1, _ = evaluate_dq(a1, q_gen)
    ft, _ = evaluate_dq_np(trunc, q_gen)
    blow_gen = float(np.max(np.abs(ft - f1)))

    # commensurate zone-boundary for the 2x2x2 rock toy; NOT commensurate for
    # the 3x3x1 hex toy (1/2 is not a multiple of 1/3)
    q_zb = np.array([0.5, 0.0, 0.0])
    f1z, _ = evaluate_dq(a1, q_zb)
    ftz, _ = evaluate_dq_np(trunc, q_zb)
    dzb = float(np.max(np.abs(ftz - f1z)))
    print(f"\n[a2-negctl-b:{name}] truncation: generic q |dv| = "
          f"{blow_gen:.3f} THz (BITES); zb (1/2,0,0) |dv| = {dzb:.3e} THz")
    assert blow_gen > 0.1                            # the control bites...
    with pytest.raises(AssertionError):              # ...fail-then-pass
        assert blow_gen < FREQ_TOL
    if name == "rock3d":
        assert dzb < 1e-8       # commensurate: equal phases, invisible
    else:
        assert dzb > 0.1        # hex zone boundary IS incommensurate: bites


@tier2
def test_negctl_corrupt_fc_on_disk_bites(toy, tmp_path):
    """(c) one corrupted FC block on disk -> the oracle fails loudly."""
    name, a1, _a2, d = toy
    corr = tmp_path / "corr"
    shutil.copytree(d, corr)
    lines = (corr / "FORCE_CONSTANTS").read_text().splitlines(keepends=True)
    for k in (2, 3, 4):                              # first block's 3 rows
        lines[k] = "  ".join(f"{float(x) * 1.5:.15f}"
                             for x in lines[k].split()) + "\n"
    (corr / "FORCE_CONSTANTS").write_text("".join(lines))
    a2c = load_engine_np(str(corr / "phonopy.yaml"))
    q = np.array([0.1234, 0.2345, 0.0567])
    from phonon_core.engine import evaluate_dq
    f1, _ = evaluate_dq(a1, q)
    fc_, _ = evaluate_dq_np(a2c, q)
    dev = float(np.max(np.abs(fc_ - f1)))
    print(f"\n[a2-negctl-c:{name}] corrupted FC block: |dv| = {dev:.3f} THz "
          f"-> oracle BITES")
    assert dev > 0.05                                # the corruption is seen
    with pytest.raises(AssertionError):              # fail-then-pass
        assert dev < FREQ_TOL


# ===========================================================================
# Tier 3 — the real dataset (A2 -> A1 -> band.hdf5 transitively closed)
# ===========================================================================
@tier3
@pytest.mark.skipif(not _HAVE_H5PY, reason="h5py not present")
def test_dataset_a2_vs_band_file():
    """A2 alone (NO phonopy needed) vs the dataset's band.hdf5 at every path
    q — the file was computed by the phonopy CLI, so this transitively
    closes A2 -> A1 -> file.

    METRIC DOCTRINE (0.6.87): compared as |dnu| above NU_FLOOR and |dlambda|
    below it. SnSe2's Gamma acoustic modes sit at nu ~ 3e-07 THz, where
    dnu = dlambda / 2 nu turns 1.8e-14 THz^2 of EIGENSOLVER noise into 3.1e-08
    THz of apparent frequency error — a raw |dnu| < 1e-8 pin there measures the
    BLAS, not the engine, and is not portable across LAPACK builds (it passed
    under the 0.6.86 venv's OpenBLAS and fails under conda's, on the SHIPPED
    0.6.86 engine, byte-for-byte). Away from zero the engines agree to 6e-12
    THz; the eigenvectors agree to 3e-16 either way."""
    from phonon_core import load_band_hdf5
    band = load_band_hdf5(os.path.join(_DS, "band.hdf5"))
    a2 = load_engine_np(os.path.join(_DS, "phonopy.yaml"))
    qs = np.asarray(band.q_points)
    f2, v2 = evaluate_dq_np_batch(a2, qs)
    f_ref = np.asarray(band.frequencies)
    _df, ov, pd = mode_agreement(f_ref, np.asarray(band.eigenvectors), f2, v2)
    d_nu, d_lam = doctrine_agreement(f_ref, f2)          # METRIC DOCTRINE
    print(f"\n[a2-dataset] vs band.hdf5 FILE over {len(qs)} q: "
          f"max|dv| (|v| > {NU_FLOOR} THz) = {d_nu:.3e} THz, "
          f"max|dlambda| (near zero) = {d_lam:.3e} THz^2, "
          f"overlap deficit = {1 - ov:.3e}, proj dev = {pd:.3e}")
    assert d_nu < FREQ_TOL and d_lam < LAM_TOL
    assert ov > 1 - OV_TOL and pd < PROJ_TOL


@tier3
@tier2
def test_dataset_a2_vs_a1_path_and_cloud():
    """A2 vs A1 on the SnSe2 data_task quadruple: all 909 band-path q plus a
    125-point random in-BZ cloud, under the METRIC DOCTRINE (see
    test_dataset_a2_vs_band_file: the Gamma acoustic zeros make a raw |dnu|
    tolerance a BLAS thermometer)."""
    pytest.importorskip("h5py")
    from phonon_core import load_band_hdf5
    from phonon_core.engine import load_engine
    band = load_band_hdf5(os.path.join(_DS, "band.hdf5"))
    a1 = load_engine(os.path.join(_DS, "phonopy.yaml"))
    a2 = load_engine_np(os.path.join(_DS, "phonopy.yaml"))
    rng = np.random.default_rng(7)
    qs = np.vstack([np.asarray(band.q_points),
                    rng.uniform(-0.5, 0.5, (125, 3))])
    f1, v1 = _a1_modes(a1, qs)
    f2, v2 = evaluate_dq_np_batch(a2, qs)
    _df, ov, pd = mode_agreement(f1, v1, f2, v2)
    d_nu, d_lam = doctrine_agreement(f1, f2)             # METRIC DOCTRINE
    print(f"\n[a2-dataset] vs A1 over {len(qs)} q (909 path + 125 cloud): "
          f"max|dv| (|v| > {NU_FLOOR} THz) = {d_nu:.3e} THz, "
          f"max|dlambda| (near zero) = {d_lam:.3e} THz^2, "
          f"overlap deficit = {1 - ov:.3e}, proj dev = {pd:.3e}")
    assert d_nu < FREQ_TOL and d_lam < LAM_TOL
    assert ov > 1 - OV_TOL and pd < PROJ_TOL


# ===========================================================================
# Tier 2 — THE PSEUDO-TRANSLATION TOY (0.6.87): the tie-tolerance regime
#
# WHY THIS TOY EXISTS. The shortest-vector tie tolerance is phonopy's symprec:
# an ABSOLUTE 1e-5 ANGSTROM. Until 0.6.87 A2 used a RELATIVE 1e-5, i.e.
# 7e-5..1.1e-4 A at 7-11 A pair distances — 7-11x looser — so A2 tied images
# phonopy does not. The tie decision is INVISIBLE AT GAMMA (every phase is 1;
# averaging 1's gives 1) and only bites at generic q.
#
# NEITHER pre-existing toy can reach that failure mode, and that is the point:
#   - random-position P1 toys have min image gaps ~1e-2 A — THREE DECADES
#     above the tolerance window; every tie is decided identically by any
#     tolerance between 1e-12 and 1e-3.
#   - high-symmetry cells (rocksalt, hex) have EXACT ties (gap ~1e-16 A) —
#     likewise decided identically by any tolerance.
# A convention that only manifests in a narrow structural regime needs a toy
# CONSTRUCTED to sit in that regime. So: displace one sublattice by
# (1/n + delta) along a, with delta chosen so competing image distances differ
# by ~2e-5 A — ABOVE phonopy's absolute 1e-5 A (phonopy does NOT tie them) but
# BELOW a 1e-5 RELATIVE tolerance at that distance (the old A2 DID tie them).
# The gap is ASSERTED to sit in (1e-5, 1e-4) A so a future refactor cannot
# silently make this toy toothless.
# ===========================================================================
PSEUDO_A = 6.0                       # |a| in Angstrom
#: fractional displacement of sublattice 1 off the exact 1/2 pseudo-translation.
#: The competing images of an affected pair differ in |v| by 2*delta*|a|, so
#: delta = 1e-5 / |a| puts the gap at 2e-5 A — dead centre of the window.
PSEUDO_DELTA = 1e-5 / PSEUDO_A
PSEUDO_GAP_LO = 1e-5                 # > phonopy symprec: phonopy does NOT tie
PSEUDO_GAP_HI = 1e-4                 # < 1e-5 RELATIVE at these distances


def _build_pseudo_toy(dirpath, seed=91):
    """Triclinic P1 3-atom toy with a near-1/2 pseudo-translation along a.

    Supercell 3x1x1 (9 atoms). FC: seeded random, pair-symmetrized
    (Phi(i,j) = Phi(j,i)^T) and ASR-corrected on the diagonal, written COMPACT
    (n_p, n_s, 3, 3) via p2s_map — the same FORCE_CONSTANTS spelling phonopy
    itself emits for a compact FC, so the parser path is exercised too.
    """
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.file_IO import write_FORCE_CONSTANTS

    cell = PhonopyAtoms(
        symbols=["Na", "Cl", "K"],
        cell=np.array([[PSEUDO_A, 0.0, 0.0],
                       [0.7, 5.3, 0.0],
                       [0.4, 0.9, 6.7]]),
        scaled_positions=[[0.0, 0.0, 0.0],
                          [0.5 + PSEUDO_DELTA, 0.0, 0.0],   # <- the toy
                          [0.23, 0.61, 0.44]],
        masses=[22.99, 35.45, 39.10])
    ph = Phonopy(cell, supercell_matrix=np.diag([3, 1, 1]),
                 primitive_matrix=np.eye(3), log_level=0)
    n_s = len(ph.supercell)
    rng = np.random.default_rng(seed)
    fc = rng.uniform(-1.0, 1.0, (n_s, n_s, 3, 3))
    fc = 0.5 * (fc + fc.transpose(1, 0, 3, 2))             # pair symmetry
    idx = np.arange(n_s)
    fc[idx, idx] = 0.0
    fc[idx, idx] = -fc.sum(axis=1)                          # ASR
    p2s = ph.primitive.p2s_map
    ph.force_constants = fc
    ph.save(os.path.join(dirpath, "phonopy.yaml"),
            settings={"force_constants": False})
    write_FORCE_CONSTANTS(fc[p2s], p2s_map=p2s,
                          filename=os.path.join(dirpath, "FORCE_CONSTANTS"))
    return os.path.join(dirpath, "phonopy.yaml")


def _image_gaps(yaml_path):
    """Every pair's (d_min, second-shortest image distance), in Angstrom, over
    the [-1..1]^3 supercell-translation shell.

    Measured from PHONOPY's own supercell — deliberately NOT from the engine:
    this is a statement about the toy's GEOMETRY, and it must stay true even if
    A2's tie rule (the convention under test) is wrong."""
    import phonopy
    ph = phonopy.load(yaml_path, produce_fc=False, log_level=0)
    slat = np.asarray(ph.supercell.cell, dtype=np.float64)      # rows
    r_s = np.asarray(ph.supercell.positions, dtype=np.float64)  # Cartesian
    r_p = r_s[np.asarray(ph.primitive.p2s_map)]
    rng3 = np.arange(-1, 2)
    L = np.stack(np.meshgrid(rng3, rng3, rng3, indexing="ij"),
                 axis=-1).reshape(-1, 3).astype(np.float64) @ slat
    d = np.linalg.norm(r_s[:, None, None, :] - r_p[None, :, None, :]
                       + L[None, None, :, :], axis=-1)     # (N_s, N_p, K)
    ds = np.sort(d, axis=-1)
    return ds[..., 0], ds[..., 1]


def _phonopy_multiplicities(yaml_path):
    """A1's OWN shortest-vector multiplicities — the oracle's tie decision."""
    import phonopy
    ph = phonopy.load(yaml_path, produce_fc=False, log_level=0)
    dm = ph.dynamical_matrix
    if dm is None:                                          # pragma: no cover
        ph.force_constants = np.zeros((len(ph.primitive),
                                       len(ph.supercell), 3, 3))
        dm = ph.dynamical_matrix
    _svecs, multi = dm.primitive.get_smallest_vectors()
    multi = np.asarray(multi)
    # phonopy >= 4 returns the COMPACT (N_s, N_p, 2) form: [multiplicity,
    # offset-into-svecs]. Older shapes are (N_s, N_p) multiplicities already.
    return multi[..., 0] if multi.ndim == 3 else multi


@pytest.fixture(scope="module")
def pseudo_toy(tmp_path_factory):
    """(A1 Phonopy object, A2 NumpyEngine, yaml path) for the pseudo-translation
    toy — the ONLY fixture in the suite whose image gaps land in the
    tie-tolerance window."""
    from phonon_core.engine import load_engine
    d = str(tmp_path_factory.mktemp("a2_pseudo"))
    yaml = _build_pseudo_toy(d)
    return load_engine(yaml), load_engine_np(yaml), yaml


@tier2
def test_pseudo_toy_sits_in_the_tie_window(pseudo_toy):
    """THE SHARPNESS ASSERTION. A toy that does not SIT in the tolerance window
    cannot exercise the tolerance — and would go on passing forever while
    testing nothing. Pin the geometry, not the outcome:

      1e-5 A  <  min nonzero image gap  <  1e-4 A

    Lower bound: > phonopy's absolute symprec (1e-5 A) — so phonopy does NOT
    tie these images (multiplicity 1, asserted).
    Upper bound: < a 1e-5 RELATIVE tolerance at these ~10 A distances (which
    is 1e-4 A) — so the PRE-0.6.87 A2 DID tie them. The bug lives strictly
    inside this band; the toy is inside it too.
    """
    _a1, a2, yaml = pseudo_toy
    dmin, d2nd = _image_gaps(yaml)
    gap = d2nd - dmin
    nz = gap[gap > 1e-12]                    # ignore the exact/self ties
    assert nz.size > 0
    gmin = float(nz.min())
    in_window = int(((gap > PSEUDO_GAP_LO) & (gap < PSEUDO_GAP_HI)).sum())
    print(f"\n[a2-pseudo] min nonzero image gap = {gmin:.3e} A "
          f"(window {PSEUDO_GAP_LO:.0e} .. {PSEUDO_GAP_HI:.0e} A), "
          f"pairs in window = {in_window}")
    assert PSEUDO_GAP_LO < gmin < PSEUDO_GAP_HI, (
        "the pseudo-translation toy has drifted OUT of the tie-tolerance "
        "window — it can no longer catch a relative-vs-absolute tie bug")
    assert in_window > 0

    # the oracle's ruling on those images, and A2's, must AGREE: no tie.
    multi_a1 = _phonopy_multiplicities(yaml)
    print(f"[a2-pseudo] phonopy max multiplicity = {int(multi_a1.max())}, "
          f"A2 max multiplicity = {int(a2.counts.max())}")
    assert int(multi_a1.max()) == 1          # phonopy does NOT tie them
    assert int(a2.counts.max()) == 1         # neither does A2 (post-0.6.87)
    assert np.array_equal(np.asarray(a2.counts).reshape(-1),
                          multi_a1.reshape(-1).astype(a2.counts.dtype))


@tier2
def test_pseudo_toy_oracle(pseudo_toy):
    """A2 == A1 on the pseudo-translation toy at Gamma, the zone boundary, and
    a random q cloud. WITH the pre-0.6.87 relative tie rule this FAILS by
    ~1e-1 THz (see the negative control below) — and it fails ONLY away from
    Gamma, which is exactly why the bug survived the 0.6.86 battery."""
    a1, a2, _yaml = pseudo_toy
    qs = _toy_qs()
    f1, v1 = _a1_modes(a1, qs)
    f2, v2 = evaluate_dq_np_batch(a2, qs)
    df, ov, pd = mode_agreement(f1, v1, f2, v2)
    print(f"\n[a2-pseudo] vs A1 over {len(qs)} q: max|dv| = {df:.3e} THz, "
          f"overlap deficit = {1 - ov:.3e}, proj dev = {pd:.3e}")
    assert df < FREQ_TOL
    assert ov > 1 - OV_TOL
    assert pd < PROJ_TOL


@tier2
def test_negctl_relative_tie_tolerance_bites(pseudo_toy, monkeypatch):
    """(d) THE 0.6.87 NEGATIVE CONTROL — restore the PRE-0.6.87 RELATIVE tie
    rule (d <= d_min * (1 + 1e-5)) and the pseudo-toy oracle must FAIL.

    SIGNATURE (measured, and NOT the one a gauge flip gives): BOTH the
    frequencies AND the eigenvectors move. A tie error changes D(q) ITSELF —
    it averages phases over images that are not degenerate — so the SPECTRUM
    moves. Only a GAUGE flip (negctl (a)) is a similarity transform, and only
    that leaves the frequencies invariant. This is precisely the signature the
    field bug showed (0.116 THz AND overlap deficit 0.43), and asserting it
    here keeps the two failure modes distinguishable.

    Gamma is asserted CLEAN under the broken rule: every phase there is 1, so
    averaging over wrongly-tied images still gives 1. That invisibility is the
    whole reason the bug shipped.
    """
    a1, _a2_good, yaml = pseudo_toy
    real_shell = dq_numpy._shortest_table_shell

    def relative_shell(r_s, r_p, slat, shell, symprec=dq_numpy.DEFAULT_SYMPREC):
        dmin, _mask, cand = real_shell(r_s, r_p, slat, shell, symprec)
        d = np.linalg.norm(cand, axis=-1)
        bad = d <= dmin[..., None] * (1.0 + 1e-5) + 1e-12   # the PRE-0.6.87 rule
        return dmin, bad, cand

    monkeypatch.setattr(dq_numpy, "_shortest_table_shell", relative_shell)
    a2_bad = load_engine_np(yaml)
    monkeypatch.undo()

    assert int(a2_bad.counts.max()) > 1, (
        "the relative rule must TIE images the absolute rule does not — "
        "otherwise this control has no teeth")
    n_tied = int((np.asarray(a2_bad.counts) > 1).sum())

    # (i) INVISIBLE AT GAMMA — the trap that let the bug ship.
    qg = np.zeros((1, 3))
    fg1, vg1 = _a1_modes(a1, qg)
    fg2, vg2 = evaluate_dq_np_batch(a2_bad, qg)
    dfg, ovg, pdg = mode_agreement(fg1, vg1, fg2, vg2)
    assert dfg < FREQ_TOL and ovg > 1 - OV_TOL and pdg < PROJ_TOL

    # (ii) LOUD at generic q — frequencies AND eigenvectors both move.
    qs = _toy_qs()
    f1, v1 = _a1_modes(a1, qs)
    f2, v2 = evaluate_dq_np_batch(a2_bad, qs)
    df, ov, pd = mode_agreement(f1, v1, f2, v2)
    print(f"\n[negctl-d relative-tie] {n_tied} wrongly-tied pairs; at Gamma "
          f"max|dv| = {dfg:.3e} THz (INVISIBLE); over {len(qs)} q "
          f"max|dv| = {df:.3e} THz, overlap deficit = {1 - ov:.3e}")
    assert df > 1e-3          # spectrum moves: a tie error changes D itself
    assert ov < 1 - 1e-3      # ... and so do the eigenvectors
