"""Unified supercell expansion.

This module is the single source of truth for turning a primitive
:class:`~phonon_core.types.Material` into a concrete list of atom slots for an
N x M x K supercell. Both the Blender add-on and (later) the web viewer call
:func:`expand_cell`; the boundary behaviours are *modes* of one function rather
than parallel code paths.

Every returned slot carries its **integer lattice vector** ``n_int`` so the
downstream Bloch-phase displacement machinery keeps working unchanged:

    R_n = n_int . (a1, a2, a3)            (Maradudin & Vosko 1968; Dove sec. 4.2)

Expansion is purely geometric — it never touches eigenvectors or masses. The
mass-weighting (1/sqrt(m_s); Togo & Tanaka 2015) lives in
:mod:`phonon_core.kinematics` and is intentionally out of scope here.

Modes
-----
"strict"
    Only atoms whose cell index ``n`` lies in [0,nx) x [0,ny) x [0,nz).
"halo"
    The strict supercell PLUS periodic-image atoms that have at least one bond
    reaching back into the supercell (or to a previously-added image). This is
    the iterative fixed-point introduced in add-on 0.5.0, lifted here verbatim.
"window"
    Reserved for VESTA-style fractional windowing (0.6.1). Raises
    NotImplementedError for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ExpandedAtoms:
    """The atoms of an expanded cell.

    Three parallel arrays, one row per atom slot:

    j : (N,) int
        Index of the atom within the primitive unit cell (0 .. n_atoms-1).
        Selects the species, mass, and eigenvector component for the slot.
    n_int : (N, 3) int
        Integer lattice vector of the slot's cell. Used directly for the
        Bloch phase R_n = n_int . (a1, a2, a3). For "strict" every row lies in
        [0,nx) x [0,ny) x [0,nz); "halo" image rows may carry components in
        {-1, ..., nx} etc. (one-cell halo).
    r_cart : (N, 3) float
        Equilibrium Cartesian position in Angstrom,
        r_cart = (fractional_coords[j] + n_int) @ lattice.
    """

    j: np.ndarray
    n_int: np.ndarray
    r_cart: np.ndarray

    def __post_init__(self) -> None:
        self.j = np.asarray(self.j, dtype=np.int64).reshape(-1)
        self.n_int = np.asarray(self.n_int, dtype=np.int64).reshape(-1, 3)
        self.r_cart = np.asarray(self.r_cart, dtype=np.float64).reshape(-1, 3)
        n = len(self.j)
        if self.n_int.shape != (n, 3):
            raise ValueError(f"n_int must be ({n},3); got {self.n_int.shape}")
        if self.r_cart.shape != (n, 3):
            raise ValueError(f"r_cart must be ({n},3); got {self.r_cart.shape}")

    def __len__(self) -> int:
        return len(self.j)

    # -- numpy <-> plain-python round-trip helpers --------------------------
    # Scaffolding for tests and the future web viewer ONLY. This is NOT the
    # JSON v1 schema (which is separate and not yet locked); do not couple it
    # to any schema.
    def to_dict(self) -> dict:
        return {
            "j": self.j.tolist(),
            "n_int": self.n_int.tolist(),
            "r_cart": self.r_cart.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExpandedAtoms":
        return cls(j=d["j"], n_int=d["n_int"], r_cart=d["r_cart"])


# ---------------------------------------------------------------------------
# strict
# ---------------------------------------------------------------------------
def _expand_strict(material, nx: int, ny: int, nz: int) -> ExpandedAtoms:
    """Enumerate every atom of the plain N x M x K supercell.

    Iteration order is (iz, iy, ix, s) — preserved exactly from the add-on's
    generate_supercell_slots so downstream slot indexing is unchanged.
    """
    n_atoms = material.num_atoms
    cart_uc = material.cartesian_coords
    lat = material.lattice

    n_int, j, pos = [], [], []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                R = ix * lat[0] + iy * lat[1] + iz * lat[2]
                for s in range(n_atoms):
                    n_int.append([ix, iy, iz])
                    j.append(s)
                    pos.append(cart_uc[s] + R)
    return ExpandedAtoms(
        j=np.asarray(j, dtype=np.int64),
        n_int=np.asarray(n_int, dtype=np.int64),
        r_cart=np.asarray(pos, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# halo
# ---------------------------------------------------------------------------
def _image_atom_slots(material, base_n_int, base_j, base_pos,
                      nx: int, ny: int, nz: int,
                      bond_tolerance: float,
                      radius_fn: Callable[[str], float],
                      *, max_iterations: int = 2):
    """Iterative fixed-point image-atom search (add-on 0.5.0 algorithm).

    Ported verbatim from generate_image_atom_slots. The SOURCE set translated
    by +/-1 supercell each pass is fixed at the real atoms (base_pos); only the
    BOND-PARTNER set (all_pos) grows, which is what keeps 3D bulk from running
    away. Image cells are capped to a one-cell halo [-1,nx] x [-1,ny] x [-1,nz].
    Returns (image_n_int, image_j, image_pos) — image rows only.
    """
    lat = material.lattice
    species_radii = {sp: float(radius_fn(sp)) for sp in set(material.symbols)}

    sc_a = nx * lat[0]
    sc_b = ny * lat[1]
    sc_c = nz * lat[2]

    all_pos = np.asarray(base_pos, dtype=np.float64).copy()
    all_atom_idx = np.asarray(base_j, dtype=np.int64).copy()
    all_uc_idx = np.asarray(base_n_int, dtype=np.int64).copy()
    all_radii = np.array([species_radii[material.symbols[int(a)]]
                          for a in all_atom_idx], dtype=np.float64)

    src_pos = np.asarray(base_pos, dtype=np.float64)
    src_atom_idx = np.asarray(base_j, dtype=np.int64)
    src_uc_idx = np.asarray(base_n_int, dtype=np.int64)
    src_radii = np.array([species_radii[material.symbols[int(a)]]
                          for a in src_atom_idx], dtype=np.float64)

    image_uc_acc: list[tuple[int, int, int]] = []
    image_atom_acc: list[int] = []
    image_pos_acc: list[np.ndarray] = []

    seen_keys: set[tuple[int, int, int, int]] = set()
    for i in range(len(all_pos)):
        seen_keys.add((int(all_uc_idx[i, 0]), int(all_uc_idx[i, 1]),
                       int(all_uc_idx[i, 2]), int(all_atom_idx[i])))

    triplets = [(ix, iy, iz)
                for ix in (-1, 0, 1)
                for iy in (-1, 0, 1)
                for iz in (-1, 0, 1)
                if (ix, iy, iz) != (0, 0, 0)]

    for _iteration in range(max_iterations):
        max_cutoff = 2.0 * float(all_radii.max()) * bond_tolerance
        box_min = all_pos.min(axis=0) - max_cutoff
        box_max = all_pos.max(axis=0) + max_cutoff

        new_pos: list[np.ndarray] = []
        new_atom_idx: list[int] = []
        new_uc_idx: list[tuple[int, int, int]] = []

        for ix, iy, iz in triplets:
            T = ix * sc_a + iy * sc_b + iz * sc_c
            translated = src_pos + T
            near_mask = ((translated > box_min).all(axis=1) &
                         (translated < box_max).all(axis=1))
            if not np.any(near_mask):
                continue
            candidate_slots = np.flatnonzero(near_mask)
            for c in candidate_slots:
                c = int(c)
                src_uc = src_uc_idx[c]
                cand_uc = (int(src_uc[0]) + ix * nx,
                           int(src_uc[1]) + iy * ny,
                           int(src_uc[2]) + iz * nz)
                if not (-1 <= cand_uc[0] <= nx and
                        -1 <= cand_uc[1] <= ny and
                        -1 <= cand_uc[2] <= nz):
                    continue
                cand_atom = int(src_atom_idx[c])
                cand_key = (cand_uc[0], cand_uc[1], cand_uc[2], cand_atom)
                if cand_key in seen_keys:
                    continue
                cand_radius = src_radii[c]
                d = all_pos - translated[c]
                dist = np.linalg.norm(d, axis=1)
                cutoff = (cand_radius + all_radii) * bond_tolerance
                within = (dist > 0.5) & (dist < cutoff)
                if not np.any(within):
                    continue
                seen_keys.add(cand_key)
                new_pos.append(translated[c])
                new_atom_idx.append(cand_atom)
                new_uc_idx.append(cand_uc)
                image_uc_acc.append(cand_uc)
                image_atom_acc.append(cand_atom)
                image_pos_acc.append(translated[c])

        if not new_pos:
            break

        new_pos_arr = np.asarray(new_pos, dtype=np.float64)
        new_atom_idx_arr = np.asarray(new_atom_idx, dtype=np.int64)
        new_uc_idx_arr = np.asarray(new_uc_idx, dtype=np.int64)
        new_radii = np.array([species_radii[material.symbols[int(a)]]
                              for a in new_atom_idx_arr], dtype=np.float64)
        all_pos = np.vstack([all_pos, new_pos_arr])
        all_atom_idx = np.concatenate([all_atom_idx, new_atom_idx_arr])
        all_uc_idx = np.vstack([all_uc_idx, new_uc_idx_arr])
        all_radii = np.concatenate([all_radii, new_radii])

    if not image_uc_acc:
        return (np.zeros((0, 3), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, 3), dtype=np.float64))
    return (np.asarray(image_uc_acc, dtype=np.int64),
            np.asarray(image_atom_acc, dtype=np.int64),
            np.asarray(image_pos_acc, dtype=np.float64))


def _expand_halo(material, nx: int, ny: int, nz: int,
                 bond_tolerance: float,
                 radius_fn: Callable[[str], float],
                 max_iterations: int) -> ExpandedAtoms:
    base = _expand_strict(material, nx, ny, nz)
    img_uc, img_j, img_pos = _image_atom_slots(
        material, base.n_int, base.j, base.r_cart,
        nx, ny, nz, bond_tolerance, radius_fn,
        max_iterations=max_iterations,
    )
    if len(img_j):
        return ExpandedAtoms(
            j=np.concatenate([base.j, img_j], axis=0),
            n_int=np.concatenate([base.n_int, img_uc], axis=0),
            r_cart=np.concatenate([base.r_cart, img_pos], axis=0),
        )
    return base


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def expand_cell(material, nx: int, ny: int, nz: int, *,
                mode: str, **kwargs) -> ExpandedAtoms:
    """Expand ``material`` to an N x M x K supercell under ``mode``.

    Parameters
    ----------
    material : Material
    nx, ny, nz : int
        Supercell repetitions along a1, a2, a3.
    mode : {"strict", "halo", "window"}
        See module docstring.

    Keyword args (halo only)
    ------------------------
    bond_tolerance : float, default 1.2
        Bond cutoff factor: a pair bonds if dist < (R_i + R_j) * bond_tolerance.
        Same meaning and default as the add-on's Bond tolerance slider.
    radius_fn : callable(symbol) -> float, default phonon_core.radii.covalent_radius
        Covalent-radius lookup. Injectable so the web viewer can supply its own;
        defaults to the exact callable the add-on uses today.
    max_iterations : int, default 2
        Fixed-point pass cap (pass 1 = direct PBC partners, pass 2 = completes
        image-atom coordination; further passes only add corner-halo clutter).

    Returns
    -------
    ExpandedAtoms
    """
    if mode == "strict":
        return _expand_strict(material, nx, ny, nz)

    if mode == "halo":
        from phonon_core.radii import covalent_radius
        bond_tolerance = float(kwargs.pop("bond_tolerance", 1.2))
        radius_fn = kwargs.pop("radius_fn", covalent_radius)
        max_iterations = int(kwargs.pop("max_iterations", 2))
        if kwargs:
            raise TypeError(f"unexpected kwargs for halo mode: {sorted(kwargs)}")
        return _expand_halo(material, nx, ny, nz,
                            bond_tolerance, radius_fn, max_iterations)

    if mode == "window":
        raise NotImplementedError(
            "expand_cell(mode='window') (VESTA-style fractional windowing) "
            "is planned for 0.6.1."
        )

    raise ValueError(
        f"unknown mode {mode!r}; expected 'strict', 'halo', or 'window'"
    )
