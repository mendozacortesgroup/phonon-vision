"""Round-trip-through-conformance: the real correctness gate for export.

The existing json_v1 round-trip proves the *bytes* survive write/read. This
proves the *physics* survives: write the document, read it back, and run the
0.2.0 conformance gate (``conformance.check_against_reference``) on a fixed set
of modes spanning Γ AND finite q. Tier 3 at finite q is the assertion that a
per-atom phase error — invisible to a byte-level round-trip's value comparison
when the corruption is applied before writing — cannot slip through.

This module consumes the frozen json_v1 layer and the conformance checker; it
does not modify either.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np

from phonon_core.types import BandPath, MeshGrid
from phonon_core.io.json_v1 import PhononDocument, write_json_v1, read_json_v1
from phonon_core import conformance


@dataclass
class ModeRef:
    """A selected (block, q, band) reference into a document."""
    block: str          # "band" or "mesh"
    q_index: int
    band_index: int
    q_point: np.ndarray
    frequency: float
    eigenvector: np.ndarray   # (N_atoms, 3) complex
    is_finite_q: bool


def _block_modes(block, name: str, *, max_finite: int) -> list[ModeRef]:
    """Pick Γ + up to ``max_finite`` finite-q q-points, ALL bands at each."""
    if block is None or block.eigenvectors is None:
        return []
    q = np.asarray(block.q_points, dtype=np.float64)
    qn = np.linalg.norm(q, axis=1)
    n_q = len(q)
    n_b = block.frequencies.shape[1]

    chosen_q: list[int] = []
    gamma = np.flatnonzero(qn < 1e-8)
    if gamma.size:
        chosen_q.append(int(gamma[0]))

    finite = np.flatnonzero(qn > 1e-6)
    if finite.size:
        if isinstance(block, BandPath):
            # segment boundaries (M, K, …) are the physically meaningful
            # finite-q points; add a generic interior point too.
            ends = np.cumsum(block.segment_nqpoint) - 1
            cands = [int(e) for e in ends if qn[int(e)] > 1e-6]
            cands.append(int(finite[len(finite) // 2]))   # generic interior
        else:
            # mesh: spread a few finite-q indices across the grid
            cands = [int(finite[k * len(finite) // (max_finite + 1)])
                     for k in range(1, max_finite + 1)]
        # dedupe, keep order, cap
        seen: set[int] = set()
        for c in cands:
            if c not in seen and qn[c] > 1e-6:
                seen.add(c)
                chosen_q.append(c)
                if len(seen) >= max_finite:
                    break

    out: list[ModeRef] = []
    for qi in chosen_q:
        for bi in range(n_b):
            out.append(ModeRef(
                block=name, q_index=qi, band_index=bi,
                q_point=q[qi], frequency=float(block.frequencies[qi, bi]),
                eigenvector=np.asarray(block.eigenvectors[qi, bi],
                                       dtype=np.complex128),
                is_finite_q=bool(qn[qi] > 1e-6),
            ))
    return out


def select_reference_modes(document: PhononDocument, *,
                           max_finite: int = 3) -> list[ModeRef]:
    """Modes spanning Γ and finite q, from the band block if present, plus the
    mesh block if present. Deterministic; used by the round-trip gate."""
    modes = _block_modes(document.band, "band", max_finite=max_finite)
    modes += _block_modes(document.mesh, "mesh", max_finite=max_finite)
    if not modes:
        raise ValueError("document has no eigenvector-bearing band or mesh block")
    return modes


def _stacks(modes: list[ModeRef]):
    eig = np.array([m.eigenvector for m in modes], dtype=np.complex128)
    q = np.array([m.q_point for m in modes], dtype=np.float64)
    f = np.array([m.frequency for m in modes], dtype=np.float64)
    return eig, q, f


def roundtrip_conformance(
    document: PhononDocument,
    *,
    reference: Optional[PhononDocument] = None,
    max_finite: int = 3,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> conformance.ConformanceResult:
    """Write ``document`` to JSON v1, read it back, and run the three-tier
    conformance gate on a Γ+finite-q mode set.

    By default the reread eigenvectors are compared against ``document``'s own
    in-memory eigenvectors — i.e. "did write→read preserve the physics?". Pass
    ``reference`` to compare the reread document against a *different* (e.g.
    pristine) document; this is how a phase-corrupted export is caught against
    the truth (see tests).

    Returns the :class:`ConformanceResult` (``.passed``, three ``TierResult``s,
    ``.finite_q_mask``). Mode selection is shared with the web bundle so the gate
    covers the same Γ+finite-q span.
    """
    ref_doc = reference if reference is not None else document
    ref_modes = select_reference_modes(ref_doc, max_finite=max_finite)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "doc.json")
        write_json_v1(document, path)
        reread = read_json_v1(path)

    cand_modes = select_reference_modes(reread, max_finite=max_finite)

    # selection is deterministic, so candidate/reference align mode-for-mode
    if len(cand_modes) != len(ref_modes):
        raise ValueError("mode selection mismatch between reread and reference")
    masses = ref_doc.material.masses

    cand_e, _, cand_f = _stacks(cand_modes)
    ref_e, ref_q, ref_f = _stacks(ref_modes)

    return conformance.check_against_reference(
        cand_e, ref_e, masses, ref_q, atol=atol, rtol=rtol,
        candidate_freqs=cand_f, reference_freqs=ref_f,
    )
