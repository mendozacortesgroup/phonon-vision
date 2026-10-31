"""Parse the Material section common to band.yaml and mesh.yaml.

Both files start with a header containing:
- ``reciprocal_lattice``
- ``natom``, ``lattice``, ``points`` (with symbol/coordinates/mass per atom)
"""

import numpy as np

from phonon_core.types import Material


def parse_material(data: dict) -> Material:
    """Build a Material from a parsed phonopy YAML dict."""
    lattice = np.array(data["lattice"], dtype=np.float64)
    rec_lat = np.array(data["reciprocal_lattice"], dtype=np.float64)

    points = data["points"]
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
