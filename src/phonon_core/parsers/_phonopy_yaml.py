"""Parser for the ``Material`` block of a phonopy ``phonopy.yaml`` /
``phonopy_params.yaml`` / ``phonopy_disp.yaml`` file.

These files are written by phonopy at calculation setup (``phonopy_disp.yaml``)
or post-process (``phonopy_params.yaml``). They contain the unit cell,
primitive cell, supercell, and force constants but in a different schema
than ``band.yaml`` / ``mesh.yaml``: the atom block lives under
``unit_cell`` (or ``primitive_cell``), not at the top level.

We only need the material info — lattice, atom positions, masses,
symbols — for the case where the user is loading a ``mesh.hdf5`` that
does NOT carry the material block itself (phonopy stores material info
only in the YAML headers, not in mesh.hdf5).

Reference: phonopy docs › Input files › phonopy.yaml.
"""

from __future__ import annotations

import numpy as np

from phonon_core.types import Material
from phonon_core.parsers._yaml_loader import load_yaml


def _try_block(data: dict, key: str) -> dict | None:
    """Return data[key] if it exists and looks like a cell block (has 'lattice')."""
    block = data.get(key)
    if isinstance(block, dict) and "lattice" in block and "points" in block:
        return block
    return None


def parse_phonopy_yaml_material(filepath: str) -> Material:
    """Parse the material from a ``phonopy*.yaml`` file.

    Priority of cell to read:
      1. ``primitive_cell`` — what band.yaml/mesh.yaml q-points refer to
      2. ``unit_cell`` — fallback if no separate primitive block
      3. top-level ``lattice`` + ``points`` (band.yaml/mesh.yaml style)

    Raises ValueError if none of these can be found.
    """
    data = load_yaml(filepath)

    block = (_try_block(data, "primitive_cell")
             or _try_block(data, "unit_cell"))

    if block is None:
        # Could be a band.yaml-style file that just has lattice+points at root.
        if "lattice" in data and "points" in data:
            block = data
        else:
            raise ValueError(
                f"{filepath}: could not find 'primitive_cell', 'unit_cell', "
                "or top-level lattice/points block. Is this a phonopy YAML?"
            )

    lattice = np.asarray(block["lattice"], dtype=np.float64)

    # reciprocal_lattice may be at the top level (phonopy.yaml) or absent
    if "reciprocal_lattice" in data:
        rec_lat = np.asarray(data["reciprocal_lattice"], dtype=np.float64)
    elif "reciprocal_lattice" in block:
        rec_lat = np.asarray(block["reciprocal_lattice"], dtype=np.float64)
    else:
        # Derive from lattice. phonopy convention drops 2π, so
        # rec = inverse(lattice).T (rows are b₁, b₂, b₃).
        rec_lat = np.linalg.inv(lattice).T

    points = block["points"]
    symbols: list[str] = []
    masses: list[float] = []
    frac_coords: list[list[float]] = []
    for p in points:
        symbols.append(p["symbol"])
        masses.append(float(p["mass"]))
        frac_coords.append(list(p["coordinates"]))

    return Material(
        lattice=lattice,
        reciprocal_lattice=rec_lat,
        symbols=symbols,
        masses=np.asarray(masses, dtype=np.float64),
        fractional_coords=np.asarray(frac_coords, dtype=np.float64),
    )
