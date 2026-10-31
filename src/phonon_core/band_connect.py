"""Band-following: connect branches along a q-path by eigenvector overlap.

The problem
-----------
An eigensolver returns modes sorted by frequency, so the "branch index" is a
frequency rank, not a physical identity. Wherever two bands cross, the ranks
swap while the physical bands sail through — and every consumer that treats
the index as an identity (fatband colouring by branch, "step to the next
q-point on this branch", the trajectory export's ``branch=`` metadata)
silently jumps from one physical band to another at the crossing.

The standard repair is maximum-overlap connection: at successive path points
:math:`q_i, q_{i+1}` the physical band that was :math:`e_b(q_i)` is the mode
at :math:`q_{i+1}` with the largest squared eigenvector overlap
:math:`|\\langle e_b(q_i)|e_j(q_{i+1})\\rangle|^2`. For mass-weighted,
orthonormal eigenvectors the overlap matrix is doubly stochastic, and with
path sampling fine enough that bands rotate little between neighbouring
points it is close to a permutation matrix — so a greedy best-first
assignment resolves it. Squared moduli make the connection GAUGE-INVARIANT:
the arbitrary per-mode phase (and the arbitrary unitary mixing inside a
degenerate subspace) cannot affect it, which also means that *within* an
exactly degenerate subspace the choice of partner is genuinely arbitrary —
any valid assignment is equally physical there.

Two hard rules, both inherited from field lessons:

* **Never connect across a path discontinuity.** A disconnected path's
  junction is not a neighbouring q-point; eigenvectors there share no
  continuity. Each continuous segment restarts from the sorted order.
* The connection is a **relabelling only**: frequencies and eigenvectors are
  untouched; the output is a permutation table consumers may apply. Nothing
  downstream is forced through it.

Pure numpy. The reference for the browser side, when the viewer learns to
consume the baked table (forward work).
"""

from __future__ import annotations

import numpy as np

__all__ = ["overlap_matrix", "connect_bands", "apply_permutations"]


def overlap_matrix(e_prev, e_next):
    """Squared-modulus overlap matrix between two eigenvector sets.

    Parameters
    ----------
    e_prev, e_next : (Nb, Na, 3) complex — one mode per row, the engine's
        mass-weighted convention (unit norm per mode).

    Returns
    -------
    (Nb, Nb) float: ``M[b, j] = |<e_prev_b | e_next_j>|^2``. Rows/columns sum
    to ~1 for orthonormal inputs; deviations are an honest diagnostic, so the
    inputs are deliberately NOT renormalized here.
    """
    a = np.asarray(e_prev).reshape(e_prev.shape[0], -1)
    b = np.asarray(e_next).reshape(e_next.shape[0], -1)
    inner = a.conj() @ b.T
    return np.abs(inner) ** 2


def _assign(M):
    """Greedy best-first assignment on an overlap matrix.

    Pairs are taken in descending overlap; each row and column is used once.
    For near-permutation matrices (the fine-sampling regime this module is
    for) this is exact; in a degenerate subspace any choice is equally valid
    (see module docstring). ``kind="stable"`` makes ties deterministic —
    identical inputs always yield the identical permutation, which the tests
    pin.

    Returns perm with ``perm[b] = j``: connected band b continues as raw
    (frequency-sorted) index j.
    """
    nb = M.shape[0]
    order = np.argsort(M, axis=None, kind="stable")[::-1]
    perm = np.full(nb, -1, dtype=np.intp)
    row_used = np.zeros(nb, dtype=bool)
    col_used = np.zeros(nb, dtype=bool)
    placed = 0
    for flat in order:
        i, j = divmod(int(flat), nb)
        if row_used[i] or col_used[j]:
            continue
        perm[i] = j
        row_used[i] = True
        col_used[j] = True
        placed += 1
        if placed == nb:
            break
    return perm


def connect_bands(eig_real, eig_imag, *, segments=None):
    """Follow bands along a path by maximum eigenvector overlap.

    Parameters
    ----------
    eig_real, eig_imag : (Nq, Nb, Na, 3) float — the payload's split
        eigenvector arrays (the band block's own storage shape).
    segments : optional sequence of int — q-point count per CONTINUOUS
        path segment, summing to Nq (phonopy's ``segment_nqpoint``). Each
        segment restarts from sorted order; no connection is made across a
        boundary. Omitted = one continuous segment.

    Returns
    -------
    perms : (Nq, Nb) intp — ``perms[i, b]`` is the raw (frequency-sorted)
        index at point i of connected band b. Row 0 of each segment is the
        identity: connected labels are DEFINED as the sorted order at the
        segment start.
    info : dict —
        ``matched_overlap`` (Nq, Nb): the |<..>|^2 each connection achieved
        (1.0 at segment starts by convention);
        ``n_reorder``: number of steps whose permutation differs from the
        previous point's (crossing count, roughly);
        ``min_overlap``: the weakest connection made anywhere (a small value
        means the sampling is too coarse for following to be trustworthy —
        report it, never hide it);
        ``segment_starts``: the q indices where relabelling reset.
    """
    er = np.asarray(eig_real, dtype=np.float64)
    ei = np.asarray(eig_imag, dtype=np.float64)
    if er.shape != ei.shape or er.ndim != 4:
        raise ValueError(f"eigenvectors must be (Nq, Nb, Na, 3); got "
                         f"real {er.shape}, imag {ei.shape}")
    nq, nb = er.shape[:2]
    eig = er + 1j * ei

    if segments is None:
        segments = [nq]
    segments = [int(s) for s in segments]
    if sum(segments) != nq or any(s < 1 for s in segments):
        raise ValueError(f"segments {segments} do not partition Nq={nq}")

    perms = np.empty((nq, nb), dtype=np.intp)
    matched = np.ones((nq, nb), dtype=np.float64)
    starts = []
    n_reorder = 0
    identity = np.arange(nb, dtype=np.intp)

    i = 0
    for seg_len in segments:
        starts.append(i)
        perms[i] = identity                       # sorted order defines labels
        prev = eig[i]                             # connected-order eigvecs
        for k in range(1, seg_len):
            j = i + k
            M = overlap_matrix(prev, eig[j])
            p = _assign(M)
            perms[j] = p
            matched[j] = M[identity, p]
            if not np.array_equal(p, perms[j - 1]):
                n_reorder += 1
            prev = eig[j][p]                      # chain from the CONNECTED order
        i += seg_len

    interior = np.ones(nq, dtype=bool)
    interior[np.asarray(starts, dtype=int)] = False
    info = {
        "matched_overlap": matched,
        "n_reorder": int(n_reorder),
        "min_overlap": float(matched[interior].min()) if interior.any() else 1.0,
        "segment_starts": starts,
    }
    return perms, info


def apply_permutations(per_band, perms):
    """Reorder a per-band array into connected-band order.

    ``per_band``: (Nq, Nb, ...) — frequencies, characters, or eigenvectors.
    Returns the array with ``out[i, b] = per_band[i, perms[i, b]]``.
    """
    arr = np.asarray(per_band)
    p = np.asarray(perms, dtype=np.intp)
    if arr.shape[:2] != p.shape:
        raise ValueError(f"array {arr.shape} does not match perms {p.shape}")
    idx = np.expand_dims(p, axis=tuple(range(2, arr.ndim)))
    return np.take_along_axis(arr, np.broadcast_to(idx, arr.shape).copy(), axis=1)
