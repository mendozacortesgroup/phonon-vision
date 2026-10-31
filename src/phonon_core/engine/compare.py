"""phonon_core.engine.compare — engine-vs-reference mode agreement (0.6.80).

The degeneracy-aware comparison Stage 1 established in tests/test_engine_dq.py,
factored into the package so RUNTIME consumers can gate on it — first consumer:
tools/pv_navigator_export.py's engine mode, which REFUSES to emit a payload
when the FORCE_CONSTANTS do not reproduce the band file (the FC<->band
provenance mismatch the Level-2 arc exists to catch). Pure numpy, no phonopy
import — callers bring their own engine output.

Conventions (identical to the Stage-1 battery):
- frequencies are compared absolutely (signed THz, phonopy signed-sqrt);
- NON-degenerate modes compare by phase-invariant overlap |<v_e|v_ref>|
  (a global phase per mode is arbitrary);
- DEGENERATE subspaces (|df| < deg_tol) compare by their PROJECTORS
  sum_b |v_b><v_b| — any basis of a degenerate eigenspace is arbitrary, so
  per-vector overlap is meaningless there (measured 0.23 at Gamma on SnSe2
  with a perfectly faithful engine; the projector is basis-invariant).

tests/test_engine_dq.py keeps its own local copies (Stage-1 scope, frozen at
0.6.78); this module is the shared runtime home going forward.
"""
from __future__ import annotations

import numpy as np

DEG_TOL = 1e-4   # THz — modes closer than this form one degenerate subspace


def deg_groups(freqs, tol: float = DEG_TOL):
    """Chain consecutive modes with |df| < tol into one degenerate subspace
    (freqs ascending, as phonopy and the reference files store them)."""
    groups, start = [], 0
    n = len(freqs)
    for i in range(1, n + 1):
        if i == n or abs(freqs[i] - freqs[i - 1]) > tol:
            groups.append(list(range(start, i)))
            start = i
    return groups


def mode_agreement(f_ref, v_ref, f_eng, v_eng, *, deg_tol: float = DEG_TOL):
    """Worst-case agreement between two (N_q, N_b[, N_a, 3]) mode sets.

    Parameters
    ----------
    f_ref, f_eng : (N_q, N_b) float — signed THz, ascending per q.
    v_ref, v_eng : (N_q, N_b, N_a, 3) complex — parser-layout eigenvectors.

    Returns
    -------
    (max_df, min_overlap, max_proj_dev) : floats
        max |f_eng - f_ref| over all q/branches; the minimum phase-invariant
        overlap over all NON-degenerate modes (1.0 when every mode is
        degenerate); the maximum |P_eng - P_ref| projector deviation over all
        DEGENERATE subspaces (0.0 when none exist).
    """
    f_ref = np.asarray(f_ref, dtype=np.float64)
    f_eng = np.asarray(f_eng, dtype=np.float64)
    max_df = float(np.max(np.abs(f_eng - f_ref)))
    min_ov, max_pd = 1.0, 0.0
    n_q = f_ref.shape[0]
    for qi in range(n_q):
        A = np.asarray(v_ref[qi]).reshape(f_ref.shape[1], -1)
        B = np.asarray(v_eng[qi]).reshape(f_eng.shape[1], -1)
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
        B = B / np.linalg.norm(B, axis=1, keepdims=True)
        for grp in deg_groups(f_ref[qi], tol=deg_tol):
            if len(grp) == 1:
                b = grp[0]
                min_ov = min(min_ov, float(abs(np.vdot(B[b], A[b]))))
            else:
                P_ref = sum(np.outer(A[b], A[b].conj()) for b in grp)
                P_eng = sum(np.outer(B[b], B[b].conj()) for b in grp)
                max_pd = max(max_pd, float(np.max(np.abs(P_eng - P_ref))))
    return max_df, min_ov, max_pd
