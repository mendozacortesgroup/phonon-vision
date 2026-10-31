"""Physics tests for band_connect.

Built on the house rule: every claim is checked against something the
implementation had no access to — analytic crossing lines, an adiabatic
two-level model, unitary-invariance mathematics — never against a re-run of
the same arithmetic.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phonon_core.band_connect import (  # noqa: E402
    apply_permutations, connect_bands, overlap_matrix,
)

WEB = os.path.join(os.path.dirname(__file__), "..", "web")


# ------------------------------------------------------------ constructions

def crossing_toy(nq=41):
    """Two bands with CONSTANT orthogonal eigenvectors whose analytic
    frequencies cross: f_A = 1 + 2t, f_B = 3 - 2t over t in [0, 1] (crossing
    at t = 0.5). The input arrays are what an eigensolver would return —
    sorted by frequency, so the mode ORDER swaps at the crossing. The test
    knows the straight lines; the module does not."""
    t = np.linspace(0.0, 1.0, nq)
    fA, fB = 1.0 + 2.0 * t, 3.0 - 2.0 * t
    eA = np.zeros((2, 3)); eA[0, 0] = 1.0          # atom 0 moves x
    eB = np.zeros((2, 3)); eB[1, 1] = 1.0          # atom 1 moves y
    freq = np.empty((nq, 2)); er = np.empty((nq, 2, 2, 3))
    for i in range(nq):
        pairs = sorted([(fA[i], eA), (fB[i], eB)], key=lambda p: p[0])
        for b, (f, e) in enumerate(pairs):
            freq[i, b] = f; er[i, b] = e
    return t, freq, er, np.zeros_like(er), (fA, fB)


def avoided_toy(nq=81, gap=0.15):
    """Adiabatic two-level model h(t) = [[t, g], [g, -t]] + offset: the
    eigenvectors rotate continuously and the sorted frequencies never cross,
    so following must be the IDENTITY everywhere."""
    t = np.linspace(-1.0, 1.0, nq)
    freq = np.empty((nq, 2)); er = np.zeros((nq, 2, 1, 3))
    for i, x in enumerate(t):
        h = np.array([[x, gap], [gap, -x]])
        w, v = np.linalg.eigh(h)
        freq[i] = 5.0 + w                           # positive frequencies
        er[i, 0, 0, :2] = v[:, 0]
        er[i, 1, 0, :2] = v[:, 1]
    return freq, er, np.zeros_like(er)


def random_unitary_modes(nq, nb, na, seed):
    rng = np.random.default_rng(seed)
    er = np.empty((nq, nb, na, 3)); ei = np.empty((nq, nb, na, 3))
    for i in range(nq):
        z = rng.normal(size=(3 * na, nb)) + 1j * rng.normal(size=(3 * na, nb))
        q, _ = np.linalg.qr(z)
        modes = q.T.reshape(nb, na, 3)
        er[i], ei[i] = modes.real, modes.imag
    return er, ei


# --------------------------------------------------------------- the physics

def test_true_crossing_is_undone():
    """The one thing band-following exists for: after connection the two
    bands are the analytic straight lines, not the sorted kinked ones."""
    t, freq, er, ei, (fA, fB) = crossing_toy()
    perms, info = connect_bands(er, ei)
    connected = apply_permutations(freq, perms)
    # connected band 0 starts as A (lower at t=0) and must STAY A through
    # the crossing; likewise band 1 stays B. Analytic, exact.
    assert np.allclose(connected[:, 0], fA, atol=1e-12)
    assert np.allclose(connected[:, 1], fB, atol=1e-12)
    # the sorted input really did kink (the test would otherwise test nothing)
    assert not np.allclose(freq[:, 0], fA)
    # constant orthogonal eigenvectors: every connection is perfect
    assert info["min_overlap"] == pytest.approx(1.0)
    assert info["n_reorder"] == 1                   # exactly one swap event


def test_avoided_crossing_is_not_swapped():
    """Adiabatic bands rotate continuously; following must keep the sorted
    order (identity permutations) even though the eigenvectors at the two
    path ends are completely exchanged."""
    freq, er, ei = avoided_toy()
    perms, info = connect_bands(er, ei)
    assert np.array_equal(perms, np.tile(np.arange(2), (freq.shape[0], 1)))
    assert info["n_reorder"] == 0
    # the ends really are exchanged in character (else this tests nothing):
    # the lower band's eigenvector at t=-1 vs t=+1 are nearly orthogonal
    end_overlap = overlap_matrix(er[0] + 1j * ei[0], er[-1] + 1j * ei[-1])
    assert end_overlap[0, 0] < 0.05


def test_gauge_invariance():
    """Multiplying any mode by any phase must change nothing — the overlap
    is |.|^2. This is what makes the connection convention-proof."""
    t, freq, er, ei, _ = crossing_toy()
    rng = np.random.default_rng(7)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=er.shape[:2]))
    eig = (er + 1j * ei) * phases[..., None, None]
    perms_g, _ = connect_bands(eig.real, eig.imag)
    perms_0, _ = connect_bands(er, ei)
    assert np.array_equal(perms_g, perms_0)


def test_permutations_are_valid_and_deterministic():
    er, ei = random_unitary_modes(nq=12, nb=6, na=4, seed=3)
    perms1, _ = connect_bands(er, ei)
    perms2, _ = connect_bands(er, ei)
    assert np.array_equal(perms1, perms2), "same input must give same output"
    for row in perms1:
        assert sorted(row.tolist()) == list(range(6)), "each row is a permutation"


def test_no_connection_across_a_path_break():
    """A crossing placed exactly at a segment junction must NOT be followed:
    each segment restarts from sorted order (the disconnected-path lesson —
    a junction is not a neighbouring q-point)."""
    t, freq, er, ei, _ = crossing_toy(nq=40)
    # split at the crossing: two segments of 20
    perms, info = connect_bands(er, ei, segments=[20, 20])
    assert perms[20].tolist() == [0, 1], "segment start must be identity"
    assert info["segment_starts"] == [0, 20]
    # within-segment: the crossing sits at the junction, so NO reorder events
    assert info["n_reorder"] == 0
    with pytest.raises(ValueError):
        connect_bands(er, ei, segments=[19, 20])    # does not partition Nq


def test_degenerate_subspace_yields_a_valid_assignment():
    """Inside an exact degeneracy any partner choice is physical; the
    contract is only that the assignment stays a permutation and the
    connected FREQUENCIES are unaffected (they are equal)."""
    nq, na = 9, 2
    er = np.zeros((nq, 2, na, 3)); ei = np.zeros_like(er)
    rng = np.random.default_rng(11)
    for i in range(nq):
        th = rng.uniform(0, 2 * np.pi)              # arbitrary mixing per point
        c, s = np.cos(th), np.sin(th)
        er[i, 0, 0, 0], er[i, 0, 1, 1] = c, s
        er[i, 1, 0, 0], er[i, 1, 1, 1] = -s, c
    freq = np.full((nq, 2), 3.7)                    # exactly degenerate
    perms, _ = connect_bands(er, ei)
    for row in perms:
        assert sorted(row.tolist()) == [0, 1]
    assert np.allclose(apply_permutations(freq, perms), 3.7)


def test_apply_permutations_reorders_any_rank():
    t, freq, er, ei, (fA, fB) = crossing_toy()
    perms, _ = connect_bands(er, ei)
    # eigenvectors follow too: connected band 0 keeps mode A's pattern
    er_c = apply_permutations(er, perms)
    assert np.allclose(er_c[:, 0, 0, 0], 1.0)       # atom 0 x, everywhere
    with pytest.raises(ValueError):
        apply_permutations(freq[:, :1], perms)      # shape mismatch is loud


def test_reversed_labels_negative_control():
    """Teeth: flip the mode CONTENT mid-path in a toy whose frequencies never
    cross — following must detect and undo it (non-identity permutation),
    proving the overlaps, not the frequencies, drive the connection.
    (A first version of this test flipped modes at a crossing toy's own swap
    point, which UNDID the sorted-order swap and correctly produced identity
    — the control was testing nothing. Constant frequencies remove the
    ambiguity.)"""
    nq = 10
    eA = np.zeros((2, 3)); eA[0, 0] = 1.0
    eB = np.zeros((2, 3)); eB[1, 1] = 1.0
    er = np.zeros((nq, 2, 2, 3))
    er[:, 0], er[:, 1] = eA, eB
    er[5:] = er[5:, ::-1]                           # content swap, no crossing
    perms, info = connect_bands(er, np.zeros_like(er))
    assert perms[5].tolist() == [1, 0], "the swap must be detected"
    assert np.all(perms[5:] == [1, 0]) and np.all(perms[:5] == [0, 1])
    assert info["n_reorder"] == 1


# ------------------------------------------------------------- real payload

def test_real_snse2_band_connectivity():
    path = os.path.join(WEB, "navigator_data_snse2.js")
    if not os.path.exists(path):
        pytest.skip("bundled SnSe2 payload not present")
    with open(path) as fh:
        d = json.loads(fh.read().split("=", 1)[1].strip().rstrip(";"))
    band = d["band"]
    er = np.array(band["eigenvectors_real"]); ei = np.array(band["eigenvectors_imag"])
    freq = np.array(band["frequency"])
    perms, info = connect_bands(er, ei)
    nq, nb = freq.shape
    for row in perms:
        assert sorted(row.tolist()) == list(range(nb))
    # A blanket overlap floor is WRONG on real data: inside a degenerate
    # subspace the gauge mixes freely between path points, so a 2-fold pair
    # legitimately splits to ~1/2 and a 3-fold to ~1/3 (SnSe2 measures
    # 0.368 — between the two). The discriminating physical pin: every WEAK
    # connection must be explained by near-degeneracy. A weak overlap
    # between well-separated bands would be a genuine following failure.
    matched = info["matched_overlap"]
    for i in range(1, nq):
        for b_ in range(nb):
            if matched[i, b_] >= 0.6:
                continue
            jraw = perms[i, b_]
            gaps = np.abs(freq[i] - freq[i, jraw])
            gaps[jraw] = np.inf
            assert gaps.min() < 0.25, (
                f"weak overlap {matched[i, b_]:.3f} at q {i}, band {b_} "
                f"without a nearby band (min gap {gaps.min():.3f} THz)")
    # catastrophic coarseness still guarded, at the 3-fold-degeneracy floor
    assert info["min_overlap"] > 0.25, f"weakest {info['min_overlap']}"
    # determinism on real data
    perms2, _ = connect_bands(er, ei)
    assert np.array_equal(perms, perms2)
    # the relabelling must not manufacture frequency discontinuities:
    # the largest per-step frequency jump may not grow under connection
    jump_sorted = np.abs(np.diff(freq, axis=0)).max()
    jump_conn = np.abs(np.diff(apply_permutations(freq, perms), axis=0)).max()
    assert jump_conn <= jump_sorted + 1e-9


# ------------------------------------------------------- exporter integration

def test_exporter_bakes_connectivity_opt_in(tmp_path):
    """--band-connect bakes a presence-keyed band.connect block computed from
    the payload's own arrays; WITHOUT the flag the payload is unchanged (the
    key-set pins elsewhere in the suite stay valid)."""
    import subprocess
    band = os.path.join(os.path.dirname(__file__), "fixtures", "phonopy",
                        "SnSe2_band.hdf5")
    mesh = os.path.join(os.path.dirname(__file__), "fixtures", "navigator",
                        "SnSe2_mesh_coarse.hdf5")
    if not (os.path.exists(band) and os.path.exists(mesh)):
        pytest.skip("SnSe2 fixtures not present")
    tool = os.path.join(os.path.dirname(__file__), "..", "tools",
                        "pv_navigator_export.py")
    env = dict(os.environ,
               PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"))

    def bake(extra):
        out = tmp_path / f"p{len(extra)}.js"
        r = subprocess.run(
            [sys.executable, tool, band, mesh, "-o", str(out),
             "--group-a", "Sn", "--dim", "2"] + extra,
            capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr[-800:]
        return json.loads(out.read_text()), r.stderr

    plain, _ = bake([])
    assert "connect" not in plain["band"], "must stay opt-in"

    baked, err = bake(["--band-connect"])
    conn = baked["band"]["connect"]
    freq = np.array(baked["band"]["frequency"])
    perms = np.array(conn["perm"])
    assert perms.shape == freq.shape
    for row in perms:
        assert sorted(row.tolist()) == list(range(freq.shape[1]))
    # the baked table must be EXACTLY what the library computes from the same
    # payload — the exporter may not have its own opinion
    from phonon_core.band_connect import connect_bands
    seg = [len(range(int(a2), int(b2) + 1))
           for a2, b2 in zip(*(lambda st: (st["segment_starts"],
                                           st["segment_starts"][1:] + [len(freq)]))(conn))]
    lib_perms, lib_info = connect_bands(
        np.array(baked["band"]["eigenvectors_real"]),
        np.array(baked["band"]["eigenvectors_imag"]),
        segments=[e - s for s, e in zip(conn["segment_starts"],
                                        conn["segment_starts"][1:] + [len(freq)])])
    assert np.array_equal(perms, lib_perms)
    assert conn["min_overlap"] == pytest.approx(lib_info["min_overlap"])
    assert "band connectivity" in err
