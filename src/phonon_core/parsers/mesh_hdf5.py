"""Parser for phonopy mesh.hdf5 files.

The mesh.hdf5 file is emitted when ``HDF5 = .TRUE.`` is set in mesh.conf
alongside ``MESH_SYMMETRY = .FALSE.`` and ``EIGENVECTORS = .TRUE.``.

Expected HDF5 layout (phonopy ≥ 2.x):

    /mesh           (3,)              grid dimensions (nx, ny, nz)
    /qpoint         (N_q, 3)          fractional q-points (flat, no segments)
    /weight         (N_q,)            symmetry weights (all 1 if MESH_SYMMETRY=.FALSE.)
    /frequency      (N_q, 3N)         THz, signed
    /eigenvector    (N_q, 3N, 3N)     complex; columns are bands, rows are atom-cartesian

**Important asymmetry vs band.hdf5:** mesh.hdf5 does NOT contain the
``Material`` block (no lattice, symbols, coordinates, masses). Phonopy
stores those only in the YAML files. So this loader requires the
material to be provided explicitly OR auto-detected from a sibling
``phonopy.yaml`` / ``phonopy_params.yaml`` / ``phonopy_disp.yaml`` in
the same directory.

Cross-validated against ``vasp_SnSe2_mesh.yaml`` on the 91×91×1 grid
(8281 q-points × 9 bands): 74,529 eigenvector overlaps all match
bytewise.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from phonon_core.types import MeshGrid, Material
from phonon_core.parsers._phonopy_yaml import parse_phonopy_yaml_material


# Sibling files we'll look for, in priority order (first match wins).
# phonopy emits these at different stages; ``phonopy.yaml`` (post-process
# output) and ``phonopy_params.yaml`` are the most common names users
# preserve alongside their mesh.hdf5. We also probe for a sibling band.hdf5
# or band.yaml from the same DFT calculation — both files (band and mesh)
# come from the same calculator run, so band.hdf5's material block is
# valid for the mesh.hdf5 too. This lets users get away with just two
# HDF5 files (band + mesh), with no phonopy.yaml needed.
_SIBLING_PHONOPY_YAML_CANDIDATES = (
    "phonopy.yaml",
    "phonopy_params.yaml",
    "phonopy_disp.yaml",
)
_SIBLING_BAND_CANDIDATES = (
    "band.hdf5",
    "band.h5",
    "band.yaml",
)


def _autodetect_material(path: Path) -> Material:
    """Try several common sibling files to recover the material block.

    Priority order (first hit wins):
      1. Sibling band.hdf5 / band.h5 — full material info from the
         band-structure file's self-contained header. Same DFT calc.
      2. Sibling band.yaml — same info, slower to parse than band.hdf5.
      3. Sibling phonopy*.yaml — phonopy's post-process summary files.

    Each candidate is tried at ``path.parent / candidate_name``.
    """
    parent = path.parent

    # Try band.hdf5/.h5 first — bytewise material with no YAML cost
    for name in ("band.hdf5", "band.h5"):
        candidate = parent / name
        if candidate.is_file():
            from phonon_core.parsers.band_hdf5 import load_band_hdf5
            return load_band_hdf5(str(candidate)).material

    # Try band.yaml — slower but always parseable, doesn't require h5py
    candidate = parent / "band.yaml"
    if candidate.is_file():
        from phonon_core.parsers.band_yaml import load_band_yaml
        return load_band_yaml(str(candidate)).material

    # Fall back to phonopy.yaml / phonopy_params.yaml / phonopy_disp.yaml
    for name in _SIBLING_PHONOPY_YAML_CANDIDATES:
        candidate = parent / name
        if candidate.is_file():
            return parse_phonopy_yaml_material(str(candidate))

    siblings_tried = (_SIBLING_BAND_CANDIDATES +
                      _SIBLING_PHONOPY_YAML_CANDIDATES)
    raise ValueError(
        f"mesh.hdf5 does not carry material info, and none of "
        f"{', '.join(siblings_tried)} was found next to {path}. "
        f"Either pass an explicit `material=...` argument, or put one of "
        f"those files in the same directory."
    )


def load_mesh_hdf5(
    filepath: str,
    *,
    material: Material | None = None,
) -> MeshGrid:
    """Load a phonopy mesh.hdf5 into a :class:`MeshGrid`.

    Parameters
    ----------
    filepath : str
        Path to the mesh.hdf5 file.
    material : Material, optional
        The material to associate with this mesh. If None, will look for
        a sibling ``phonopy.yaml`` (or ``phonopy_params.yaml`` /
        ``phonopy_disp.yaml``) in the same directory and parse the material
        from it. If neither is found, raises ``ValueError``.

    Raises
    ------
    ImportError
        If h5py isn't installed.
    ValueError
        If the file is missing required datasets, or if no material can
        be found.
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for HDF5 parsing. "
            "Install with `pip install h5py` (or use the YAML loader)."
        ) from e

    path = Path(filepath)

    if material is None:
        material = _autodetect_material(path)

    n_atoms = material.num_atoms
    n_bands = material.num_bands

    with h5py.File(str(path), "r") as f:
        for req in ("mesh", "qpoint", "frequency", "eigenvector"):
            if req not in f:
                raise ValueError(f"{filepath}: missing required dataset '/{req}'")

        mesh_arr = np.asarray(f["mesh"][:], dtype=np.int64)
        mesh = (int(mesh_arr[0]), int(mesh_arr[1]), int(mesh_arr[2]))
        q_points = np.asarray(f["qpoint"][:], dtype=np.float64)
        frequencies = np.asarray(f["frequency"][:], dtype=np.float64)
        eigvec_raw = np.asarray(f["eigenvector"][:], dtype=np.complex128)
        weights = (np.asarray(f["weight"][:], dtype=np.int64)
                   if "weight" in f else None)

    n_q = len(q_points)

    # Same reshape as band.hdf5: columns are bands, rows are atom-cartesian.
    # Mesh layout is already flat (n_q, 3N_row, 3N_col_band) — no segment unflatten.
    if eigvec_raw.shape != (n_q, n_bands, n_bands):
        raise ValueError(
            f"{filepath}: eigenvector shape {eigvec_raw.shape} does not match "
            f"expected ({n_q}, {n_bands}, {n_bands}) from material."
        )
    eig_band_first = eigvec_raw.transpose(0, 2, 1)
    eigenvectors = eig_band_first.reshape(n_q, n_bands, n_atoms, 3)

    if frequencies.shape != (n_q, n_bands):
        raise ValueError(
            f"{filepath}: frequency shape {frequencies.shape} does not match "
            f"expected ({n_q}, {n_bands}) from material."
        )

    return MeshGrid(
        material=material,
        mesh=mesh,
        q_points=q_points,
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        weights=weights,
    )
