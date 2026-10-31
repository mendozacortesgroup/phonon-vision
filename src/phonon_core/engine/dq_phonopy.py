"""phonon_core.engine.dq_phonopy — isolated D(q) engine via phonopy's API (A1).

Stage 1 of the Level-2 arc: evaluate the dynamical matrix at ARBITRARY
fractional q and return frequencies + eigenvectors in the tool's own layout.
Until now this package only PARSED phonopy outputs (band/mesh files sampled on
a fixed path/grid); this module adds the capability to compute at any q —
the foundation for band-within-mesh (Level-2) work.

A1 deliberately drives phonopy's own API (``phonopy.load`` +
``Phonopy.run_qpoints``) rather than re-implementing D(q); the numpy
re-implementation is A2/Stage 4, for which THIS module is the oracle. Because
A1 *is* phonopy, path/grid reproduction of phonopy's own band.hdf5 / mesh.yaml
is near-exact by construction — the value of this stage is the arbitrary-q
capability, the correct extraction/shaping into the tool's conventions, and
the sanity battery (Hermiticity / acoustic sum rule / continuity) pinned by
tests/test_engine_dq.py.

Conventions (matching the parsers, single source: src/phonon_core/types.py):

- ``freq_THz`` is SIGNED THz — phonopy's signed-sqrt convention
  (Togo & Tanaka, Scr. Mater. 108, 1 (2015)): omega = sign(lambda) *
  sqrt(|lambda|) * factor, so an imaginary (unstable) mode is NEGATIVE.
  Identical to how BandPath.frequencies stores band.hdf5.
- ``eigvec`` is (N_b, N_a, 3) complex per q — mass-weighted, phonopy-normalized
  dynamical-matrix eigenvectors (Maradudin & Vosko 1968), reshaped from
  phonopy's (3N_a rows, band columns) matrix EXACTLY like
  parsers/band_hdf5.py does (transpose columns->bands, split rows into
  (atom, xyz)). No renormalization, no phase fixing — verbatim phonopy.
- DIMENSION-AGNOSTIC: one code path for every material; nothing here inspects
  or branches on dimensionality.

NAC: these datasets carry no BORN file; ``load_engine`` passes a BORN file
through to phonopy when one is present next to phonopy.yaml (or given
explicitly) and adds NO non-analytic-correction logic beyond that pass-through.

phonopy is an OPTIONAL dependency (extra: ``phonon-core[engine]``); it is
imported lazily inside the functions so importing this module stays free of
the dependency (the bonds.py/motion.py discipline in the Blender sibling).
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np


def load_engine(phonopy_yaml: str, force_constants: Optional[str] = None,
                born: Optional[str] = None):
    """Build a ``Phonopy`` object for D(q) evaluation.

    Parameters
    ----------
    phonopy_yaml : str
        Path to the material's ``phonopy.yaml`` (carries the unit cell,
        supercell matrix, calculator/unit conventions — everything
        ``phonopy.load`` needs to rebuild the calculation).
    force_constants : str, optional
        Path to ``FORCE_CONSTANTS``. Default: a file named ``FORCE_CONSTANTS``
        next to ``phonopy_yaml`` when present (the layout of every dataset
        this stage was proven against).
    born : str, optional
        Path to a ``BORN`` file (non-analytic correction). Default: a file
        named ``BORN`` next to ``phonopy_yaml`` when present; otherwise no
        NAC. Pass-through only — no NAC logic lives here.

    Notes
    -----
    ``symmetrize_fc=False`` on purpose: the reference band.hdf5 / mesh.yaml
    these engines are validated against were produced by the phonopy CLI
    without ``FC_SYMMETRY``, so the raw FORCE_CONSTANTS must be used verbatim
    (empirically the setting does not move these datasets' frequencies, but
    determinism beats luck).
    """
    import phonopy   # optional extra: pip install phonon-core[engine]

    phonopy_yaml = os.fspath(phonopy_yaml)
    here = os.path.dirname(os.path.abspath(phonopy_yaml))
    if force_constants is None:
        cand = os.path.join(here, "FORCE_CONSTANTS")
        force_constants = cand if os.path.exists(cand) else None
    if born is None:
        cand = os.path.join(here, "BORN")
        born = cand if os.path.exists(cand) else None
    return phonopy.load(
        phonopy_yaml=phonopy_yaml,
        force_constants_filename=force_constants,
        born_filename=born,
        symmetrize_fc=False,
        produce_fc=False,
        log_level=0,
    )


def evaluate_dq(ph, q_frac) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate D(q) at one arbitrary fractional q-point.

    Parameters
    ----------
    ph : Phonopy
        The object from :func:`load_engine`.
    q_frac : (3,) array-like
        Fractional reciprocal coordinate (primitive-cell basis — the same
        basis band.hdf5 ``path`` and mesh.yaml ``q-position`` use).

    Returns
    -------
    freq_THz : (N_b,) float64
        SIGNED frequencies in THz, ascending (phonopy's signed-sqrt
        convention; imaginary = negative).
    eigvec : (N_b, N_a, 3) complex128
        Mass-weighted, phonopy-normalized eigenvectors in the SAME layout as
        ``BandPath.eigenvectors`` / ``MeshGrid.eigenvectors`` per q.
    """
    q = np.asarray(q_frac, dtype=np.float64).reshape(3)
    ph.run_qpoints([q], with_eigenvectors=True)
    out = ph.get_qpoints_dict()
    freq = np.asarray(out["frequencies"][0], dtype=np.float64)       # (3N,)
    cols = np.asarray(out["eigenvectors"][0], dtype=np.complex128)   # (3N, 3N)
    n_b = freq.shape[0]
    n_a = n_b // 3
    # columns are bands; rows encode atom*3+cartesian — the identical
    # transpose+reshape parsers/band_hdf5.py applies to the hdf5 matrix.
    eigvec = cols.T.reshape(n_b, n_a, 3)
    return freq, eigvec


def dynamical_matrix_at(ph, q_frac) -> np.ndarray:
    """The raw (3N, 3N) complex dynamical matrix D(q) phonopy diagonalizes.

    Exposed for the sanity battery (Hermiticity; eigenvalue->frequency
    consistency via ``ph.unit_conversion_factor``) and as the object A2's
    numpy re-implementation must reproduce. Not consumed by anything else.
    """
    q = np.asarray(q_frac, dtype=np.float64).reshape(3)
    dm = ph.dynamical_matrix
    dm.run(q)
    return np.asarray(dm.dynamical_matrix, dtype=np.complex128)
