"""Parser for phonopy mesh.yaml files.

For 3D surface plots over the BZ, you must run phonopy with both
``MESH_SYMMETRY = .FALSE.`` and ``EIGENVECTORS = .TRUE.`` so that:

1. Every q in the full grid is present (the surface plot needs the full grid).
2. Per-q eigenvectors are available for atomic projection.

Expected structure:

    mesh: [91, 91, 1]
    nqpoint: 8281
    reciprocal_lattice: ...
    natom: 3
    lattice: ...
    points: ...
    phonon:
      - q-position: [...]
        distance_from_gamma: 0.0
        weight: 1
        band:
          - frequency: ...
            eigenvector: [...]   # optional, present iff EIGENVECTORS=.TRUE.
          ...
      ...
"""

from __future__ import annotations

import numpy as np

from phonon_core.types import MeshGrid
from phonon_core.parsers._yaml_loader import load_yaml
from phonon_core.parsers._material import parse_material


def load_mesh_yaml(filepath: str, *, eigenvectors: bool = True) -> MeshGrid:
    """Load a phonopy mesh.yaml into a :class:`MeshGrid`.

    Parameters
    ----------
    filepath : str
    eigenvectors : bool
        If True (default), parse eigenvectors when present.
        Set False to skip eigenvector parsing if you only need the dispersion
        surface (saves time and memory on large meshes).
    """
    data = load_yaml(filepath)

    if "phonon" not in data:
        raise ValueError(f"{filepath}: missing 'phonon' section")

    material = parse_material(data)
    n_bands = material.num_bands
    n_atoms = material.num_atoms

    if "mesh" in data:
        mesh = tuple(int(m) for m in data["mesh"])  # type: ignore[assignment]
    else:
        raise ValueError(f"{filepath}: missing 'mesh' size field")

    phonons = data["phonon"]
    n_q = len(phonons)

    q_points = np.empty((n_q, 3), dtype=np.float64)
    frequencies = np.empty((n_q, n_bands), dtype=np.float64)
    weights = np.empty(n_q, dtype=np.int64)

    # Decide once whether eigenvectors should be loaded
    first_band = phonons[0]["band"][0]
    has_eig_in_file = "eigenvector" in first_band
    load_eig = eigenvectors and has_eig_in_file

    if load_eig:
        eigvecs = np.empty((n_q, n_bands, n_atoms, 3), dtype=np.complex128)
    else:
        eigvecs = None

    for iq, pt in enumerate(phonons):
        q_points[iq] = pt["q-position"]
        weights[iq] = int(pt.get("weight", 1))
        bands = pt["band"]
        if len(bands) != n_bands:
            raise ValueError(
                f"{filepath}: q-point {iq} has {len(bands)} bands, expected {n_bands}"
            )
        for ib, b in enumerate(bands):
            frequencies[iq, ib] = float(b["frequency"])
            if load_eig:
                # Bulk numpy convert; same speedup as band_yaml.
                eig_arr = np.asarray(b["eigenvector"], dtype=np.float64)
                eigvecs[iq, ib] = eig_arr[..., 0] + 1j * eig_arr[..., 1]

    return MeshGrid(
        material=material,
        mesh=mesh,  # type: ignore[arg-type]
        q_points=q_points,
        frequencies=frequencies,
        eigenvectors=eigvecs,
        weights=weights,
    )
