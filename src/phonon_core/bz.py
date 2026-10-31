"""First Brillouin zone geometry.

The first BZ is the Wigner-Seitz cell of the reciprocal lattice: the set of
points closer to the origin than to any other reciprocal-lattice point. For a 2D
lattice it is the Voronoi cell of the origin among the reciprocal-lattice points,
a convex polygon (a hexagon for a hexagonal lattice, a rectangle for a
rectangular one, ...).

This helper is the single source of the first-BZ polygon for the whole project —
the navigator's BZ prism, the surface mask, and any future plot all consume it,
so the zone they draw and the zone they clip to are guaranteed identical.
"""
from __future__ import annotations

import numpy as np

__all__ = ["first_brillouin_zone_2d"]


def first_brillouin_zone_2d(reciprocal_lattice, *, shell: int = 2) -> np.ndarray:
    """Ordered 2D vertices of the first Brillouin zone (Wigner-Seitz cell).

    General for any 2D lattice via the Voronoi/Wigner-Seitz construction.

    Parameters
    ----------
    reciprocal_lattice : (3,3) or (2,2) array
        Reciprocal lattice with rows = reciprocal vectors. For a 3x3 input the
        in-plane block ``[:2, :2]`` is used (the standard 2D-material setup with
        b1, b2 in the xy-plane and b3 = c* along z); the z components of b1, b2
        are assumed zero.
    shell : int
        Half-width of the reciprocal-lattice point shell used to build the
        Voronoi diagram. 2 (a 5x5 neighborhood) is sufficient for any 2D lattice;
        increase only for extremely oblique cells.

    Returns
    -------
    vertices : (n_vertices, 2) float
        The polygon vertices in Cartesian reciprocal units, ordered
        counter-clockwise (by polar angle about the origin). Same units as
        ``reciprocal_lattice``.
    """
    try:
        from scipy.spatial import Voronoi
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch
        raise ModuleNotFoundError(
            "scipy is required for first-Brillouin-zone construction "
            "(scipy.spatial.Voronoi). Install the extra:\n"
            "    pip install 'phonon_core[export]'\n"
            "or:  pip install scipy\n"
            f"(import failed: {e})"
        ) from e

    recip = np.asarray(reciprocal_lattice, dtype=float)
    if recip.shape == (3, 3):
        b = recip[:2, :2]
    elif recip.shape == (2, 2):
        b = recip
    else:
        raise ValueError("reciprocal_lattice must be (3,3) or (2,2)")

    pts = []
    origin_index = 0
    k = 0
    for i in range(-shell, shell + 1):
        for j in range(-shell, shell + 1):
            pts.append(i * b[0] + j * b[1])
            if i == 0 and j == 0:
                origin_index = k
            k += 1
    pts = np.array(pts)

    vor = Voronoi(pts)
    region = vor.regions[vor.point_region[origin_index]]
    if len(region) == 0 or -1 in region:
        raise ValueError("origin Voronoi region is unbounded; increase `shell`")
    verts = vor.vertices[region]
    ang = np.arctan2(verts[:, 1], verts[:, 0])
    return verts[np.argsort(ang)]
