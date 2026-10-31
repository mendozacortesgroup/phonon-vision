"""Units for tests/_degeneracy.py + the gauge-invariance physics documented as
tests (1d-portability, 0.6.18). Pure numpy; no fixtures, no phonopy.

The B4 invariance tests are a REGRESSION against the failure mode that broke
the suite on MKL: the old per-mode comparisons were measuring the LAPACK gauge
— an arbitrary U(k) basis choice inside each degenerate subspace — not the
physics. Here a synthetic exactly-degenerate doublet is rotated by a random
U(2) and the per-mode quantities change while the cluster-summed spectral
weight is invariant, which is precisely why the comparators were rewritten at
cluster level.
"""
import numpy as np
import pytest

from _degeneracy import degenerate_clusters


# ---------------------------------------------------------------------------
# degenerate_clusters units
# ---------------------------------------------------------------------------
def test_clusters_isolated_and_exact():
    f = np.array([0.0, 1.0, 1.0, 2.0])
    assert degenerate_clusters(f) == [[0], [1, 2], [3]]


def test_clusters_near_degenerate_below_atol_merge():
    f = np.array([0.0, 1.0, 1.0 + 5e-5, 2.0])         # gap 5e-5 < atol 1e-4
    assert degenerate_clusters(f) == [[0], [1, 2], [3]]
    # and a gap just above atol stays split
    f2 = np.array([0.0, 1.0, 1.0 + 5e-4, 2.0])
    assert degenerate_clusters(f2) == [[0], [1], [2], [3]]


def test_clusters_greedy_chain_spans_more_than_atol():
    # a ladder of sub-threshold gaps is ONE mixing block even though its total
    # span (2.4e-4) exceeds atol — greedy chaining is intentional
    f = np.array([1.0, 1.0 + 8e-5, 1.0 + 1.6e-4, 1.0 + 2.4e-4])
    assert degenerate_clusters(f) == [[0, 1, 2, 3]]


def test_clusters_rtol_scales_with_frequency():
    # at f=1000, rtol 1e-6 contributes 1e-3 to the threshold
    f = np.array([1000.0, 1000.0 + 9e-4])
    assert degenerate_clusters(f) == [[0, 1]]
    assert degenerate_clusters(f, rtol=0) == [[0], [1]]


def test_clusters_edge_cases():
    assert degenerate_clusters(np.array([])) == []
    assert degenerate_clusters(np.array([3.0])) == [[0]]
    with pytest.raises(ValueError):
        degenerate_clusters(np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# B4: the gauge physics, documented as tests
# ---------------------------------------------------------------------------
def _doublet_and_rotation(seed=7, n=12):
    """A synthetic 2-fold degenerate eigen-doublet (orthonormal complex
    columns C, n x 2) and a random U(2) acting WITHIN the subspace."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
    C, _ = np.linalg.qr(A)
    U, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    return C, C @ U


def test_per_mode_spectral_weight_is_gauge_but_cluster_sum_is_invariant():
    """Spectral weight w_n = |<g|c_n>|^2 onto a fixed vector g: the per-mode
    values rotate with the intra-subspace U(2); the cluster sum is
    Tr[P g g^dag] and is invariant to machine precision."""
    C, C2 = _doublet_and_rotation()
    rng = np.random.default_rng(11)
    g = rng.normal(size=C.shape[0]) + 1j * rng.normal(size=C.shape[0])
    g /= np.linalg.norm(g)
    w1 = np.abs(g.conj() @ C) ** 2
    w2 = np.abs(g.conj() @ C2) ** 2
    assert np.max(np.abs(w1 - w2)) > 1e-3, "rotation should move per-mode weight"
    np.testing.assert_allclose(w1.sum(), w2.sum(), rtol=0, atol=1e-12)


def test_per_mode_ipr_changes_under_intra_subspace_rotation():
    """IPR_n = sum_a |c_an|^4 is global-phase invariant but NOT invariant under
    mixing of degenerate partners — the regression that explains why the old
    per-mode atol-1e-10 comparison was measuring gauge, not physics."""
    C, C2 = _doublet_and_rotation()
    ipr1 = (np.abs(C) ** 4).sum(axis=0)
    ipr2 = (np.abs(C2) ** 4).sum(axis=0)
    assert np.max(np.abs(ipr1 - ipr2)) > 1e-3
