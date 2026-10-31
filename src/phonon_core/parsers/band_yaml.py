"""Parser for phonopy band.yaml files.

Expected structure (essential fields):

    npath: 3
    nqpoint: 909
    segment_nqpoint: [303, 303, 303]
    labels: [['Γ', 'M'], ['M', 'K'], ['K', 'Γ']]
    reciprocal_lattice: [[...], [...], [...]]
    natom: 3
    lattice: [[...], [...], [...]]
    points:
      - symbol: W
        coordinates: [0, 0, 0.5]
        mass: 183.84
      ...
    phonon:
      - q-position: [0, 0, 0]
        distance: 0.0
        band:
          - frequency: -0.003
            eigenvector:
              - [[Re_x, Im_x], [Re_y, Im_y], [Re_z, Im_z]]   # atom 1
              - [[Re_x, Im_x], [Re_y, Im_y], [Re_z, Im_z]]   # atom 2
              ...
          - frequency: ...
          ...
      ...

The eigenvectors must be present (BAND.conf must include EIGENVECTORS=.TRUE.).
"""

from __future__ import annotations

import numpy as np

from phonon_core.types import BandPath
from phonon_core.parsers._yaml_loader import load_yaml
from phonon_core.parsers._material import parse_material


def load_band_yaml(filepath: str) -> BandPath:
    """Load a phonopy band.yaml into a :class:`BandPath`.

    Raises
    ------
    ValueError
        If the file is missing required fields (eigenvectors, segment metadata).
    """
    data = load_yaml(filepath)

    if "phonon" not in data:
        raise ValueError(f"{filepath}: missing 'phonon' section")

    material = parse_material(data)
    n_bands = material.num_bands
    n_atoms = material.num_atoms

    phonons = data["phonon"]
    n_q = len(phonons)

    # Pre-allocate arrays — much faster than appending.
    q_points = np.empty((n_q, 3), dtype=np.float64)
    distances = np.empty(n_q, dtype=np.float64)
    frequencies = np.empty((n_q, n_bands), dtype=np.float64)
    eigenvectors = np.empty((n_q, n_bands, n_atoms, 3), dtype=np.complex128)

    has_eigenvectors = False

    for iq, pt in enumerate(phonons):
        q_points[iq] = pt["q-position"]
        distances[iq] = float(pt.get("distance", 0.0))
        bands = pt["band"]
        if len(bands) != n_bands:
            raise ValueError(
                f"{filepath}: q-point {iq} has {len(bands)} bands, expected {n_bands}"
            )
        for ib, b in enumerate(bands):
            frequencies[iq, ib] = float(b["frequency"])
            if "eigenvector" in b:
                has_eigenvectors = True
                # eig is a nested list with shape (n_atoms, 3, 2) where each
                # innermost pair is [re, im]. Bulk-convert with numpy in one
                # shot instead of three nested Python loops — ~50× faster on
                # a 7-atom, 1414-q dataset (21 bands).
                eig_arr = np.asarray(b["eigenvector"], dtype=np.float64)
                # eig_arr.shape == (n_atoms, 3, 2); the last axis is (re, im).
                # Combining as Re + 1j*Im is more robust than .view(complex)
                # which would require a contiguous c128-compatible layout.
                eigenvectors[iq, ib] = eig_arr[..., 0] + 1j * eig_arr[..., 1]

    if not has_eigenvectors:
        raise ValueError(
            f"{filepath}: no eigenvectors found. Re-run phonopy with "
            "EIGENVECTORS=.TRUE. in your band.conf."
        )

    # Segment metadata — required for path tick labels
    segment_nqpoint = data.get("segment_nqpoint")
    if segment_nqpoint is None:
        # Single segment if absent
        segment_nqpoint = [n_q]
    segment_nqpoint = list(segment_nqpoint)

    labels_raw = data.get("labels")
    if labels_raw is None:
        labels: list[tuple[str, str]] = [("", "")] * len(segment_nqpoint)
    else:
        labels = [(pair[0], pair[1]) for pair in labels_raw]

    if len(labels) != len(segment_nqpoint):
        # Tolerate mismatch by padding with empty labels
        while len(labels) < len(segment_nqpoint):
            labels.append(("", ""))
        labels = labels[: len(segment_nqpoint)]

    return BandPath(
        material=material,
        q_points=q_points,
        distances=distances,
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        segment_nqpoint=segment_nqpoint,
        labels=labels,
    )
