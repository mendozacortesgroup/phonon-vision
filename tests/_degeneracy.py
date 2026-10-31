"""Degenerate-cluster identification for gauge-invariant test comparisons.

Within a degenerate eigenvalue subspace LAPACK returns an ARBITRARY orthonormal
basis — any U(k) rotation of it is an equally valid eigenbasis — and WHICH
basis you get depends on the BLAS/LAPACK build (reduction order, blocking,
FMA). Quantities that are not invariant under that rotation (per-mode spectral
weight, per-mode IPR) therefore differ legitimately between machines and can
only be compared after summing or masking over each cluster; the cluster-summed
spectral weight Tr[P_cluster rho] IS invariant.

Near-degeneracies are clustered too: first-order mixing scales like eps/gap
(eps = backend rounding differences), so a sub-1e-6-THz gap amplifies O(1e-16)
arithmetic noise into O(1) eigenvector rotations. The default atol = 1e-4 THz
was calibrated on the committed W-Sn unfolding fixtures: the per-row gap
spectrum has 10706/27195 gaps below 1e-6 (true degeneracies), only +183 more
up to 1e-4 (the near-degenerate tail), then +~2000 by 1e-3 — i.e. 1e-4 sits in
a plateau of the gap distribution, comfortably above gauge-unstable gaps and
below the genuine level spacing. rtol = 1e-6 adds ~6e-6 THz at 6 THz,
negligible against atol; it is kept for scale safety on higher-frequency rows.
"""
import numpy as np


def degenerate_clusters(freqs, atol=1e-4, rtol=1e-6):
    """Greedy adjacent clustering of a SORTED-ascending 1-D frequency row.

    Modes i and i+1 share a cluster iff f[i+1] - f[i] <= atol + rtol*|f[i]|.
    Returns a list of index lists that covers range(len(freqs)) in order.
    (Greedy chaining is intentional: a ladder of sub-threshold gaps is ONE
    gauge-mixing block even if its total span exceeds atol.)
    """
    f = np.asarray(freqs, dtype=float)
    if f.ndim != 1:
        raise ValueError("degenerate_clusters expects a 1-D frequency row")
    if f.size == 0:
        return []
    clusters = [[0]]
    for i in range(f.size - 1):
        if f[i + 1] - f[i] <= atol + rtol * abs(f[i]):
            clusters[-1].append(i + 1)
        else:
            clusters.append([i + 1])
    return clusters
