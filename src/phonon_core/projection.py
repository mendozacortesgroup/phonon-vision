"""Per-mode projection metrics derived from eigenvectors.

Two distinct quantities live here:

1. **Per-species weights** — how much of the mode amplitude is on each
   chemical species. Used to color band/surface plots by atomic character.

2. **Inverse participation ratio (IPR)** — how spatially localized the mode
   is. Used in defect analysis to distinguish bulk-like (delocalized) modes
   from defect-localized modes in a supercell calculation.

Both quantities are independent of how the eigenvector's global phase is
chosen, so canonicalization (kinematics.canonicalize_eigenvector_phase) does
not change them.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from phonon_core.types import Material, BandPath, MeshGrid


# ============================================================
# Per-atom and per-species squared amplitudes
# ============================================================
def per_atom_amplitudes(eigenvector: np.ndarray) -> np.ndarray:
    """Per-atom squared amplitude ``|e_s|² = Σ_d |e_{s,d}|²``.

    Returns a (N_atoms,) real array. Sums to 1 for normalized eigenvectors.
    """
    return np.sum(np.abs(eigenvector) ** 2, axis=1)


def species_weights(
    eigenvector: np.ndarray,
    symbols: Sequence[str],
) -> dict[str, float]:
    """Fractional weight of each species in this mode.

    Returns a dict {species: weight} that sums to 1 (modulo numerical noise).
    Order of entries follows first appearance in ``symbols``.
    """
    per_atom = per_atom_amplitudes(eigenvector)
    total = float(per_atom.sum())
    if total < 1e-15:
        # Degenerate / null mode
        unique = []
        seen: set[str] = set()
        for s in symbols:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return {s: 0.0 for s in unique}

    weights: dict[str, float] = {}
    for i, sp in enumerate(symbols):
        weights[sp] = weights.get(sp, 0.0) + float(per_atom[i])
    return {sp: w / total for sp, w in weights.items()}


def species_weights_array(
    eigenvectors: np.ndarray,
    symbols: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Vectorized species weights for a stack of eigenvectors.

    Parameters
    ----------
    eigenvectors : (..., N_atoms, 3) complex
        Any leading shape; last two dims are atoms × Cartesian.
    symbols : sequence of str
        Per-atom species labels.

    Returns
    -------
    weights : (..., N_species) float
        Normalized species weights along the last axis.
    species_order : list[str]
        Species symbols in column order (first-appearance order in ``symbols``).
    """
    per_atom = np.sum(np.abs(eigenvectors) ** 2, axis=-1)  # (..., N_atoms)

    species_order: list[str] = []
    seen: set[str] = set()
    for s in symbols:
        if s not in seen:
            seen.add(s)
            species_order.append(s)

    out = np.zeros(per_atom.shape[:-1] + (len(species_order),), dtype=np.float64)
    for j, sp in enumerate(species_order):
        idx = [i for i, s in enumerate(symbols) if s == sp]
        out[..., j] = per_atom[..., idx].sum(axis=-1)

    total = out.sum(axis=-1, keepdims=True)
    # Avoid div-by-zero for null modes
    safe = np.where(total > 1e-15, total, 1.0)
    return out / safe, species_order


# ============================================================
# IPR — inverse participation ratio (defect localization metric)
# ============================================================
def inverse_participation_ratio(eigenvector: np.ndarray) -> float:
    """IPR of one mode: ``IPR = Σ_s |e_s|⁴ / (Σ_s |e_s|²)²``.

    Bounds
    ------
    - ``IPR = 1 / N_atoms``: perfectly delocalized (all atoms move equally).
    - ``IPR = 1``: perfectly localized on a single atom.

    Interpretation
    --------------
    For a defective supercell of ``N`` atoms:

    - Bulk-like modes have IPR ≈ 1/N (small).
    - Defect-localized modes have IPR ≫ 1/N, often O(0.1-1).

    A common convenience metric is the "effective number of participating atoms"
    ``N_eff = 1 / IPR``, which ranges from 1 (fully localized) to N
    (fully delocalized).
    """
    per_atom = per_atom_amplitudes(eigenvector)
    norm = float(per_atom.sum())
    if norm < 1e-15:
        # Null eigenvector — treat as delocalized
        return 1.0 / max(len(per_atom), 1)
    return float(np.sum(per_atom ** 2) / norm ** 2)


def inverse_participation_ratio_array(eigenvectors: np.ndarray) -> np.ndarray:
    """Vectorized IPR over a stack of eigenvectors.

    Parameters
    ----------
    eigenvectors : (..., N_atoms, 3) complex

    Returns
    -------
    (...,) float
    """
    per_atom = np.sum(np.abs(eigenvectors) ** 2, axis=-1)  # (..., N_atoms)
    norm = per_atom.sum(axis=-1)                            # (...,)
    safe = np.where(norm > 1e-15, norm, 1.0)
    return np.sum(per_atom ** 2, axis=-1) / safe ** 2


# ============================================================
# Localization on a spatial region (e.g. atoms near a defect site)
# ============================================================
def localization_on_atoms(
    eigenvector: np.ndarray,
    atom_indices: Sequence[int],
) -> float:
    """Fraction of squared amplitude carried by a chosen atom subset.

    Returns a value in ``[0, 1]``: 0 = no participation, 1 = entirely on the
    subset.

    For defect analysis, ``atom_indices`` would be "atoms within R Å of the
    defect site". The result tells you what fraction of the mode is "near
    the defect".
    """
    per_atom = per_atom_amplitudes(eigenvector)
    total = float(per_atom.sum())
    if total < 1e-15:
        return 0.0
    return float(per_atom[list(atom_indices)].sum() / total)


# ============================================================
# Bulk operators across full band paths or meshes
# ============================================================
def band_path_species_weights(band: BandPath) -> tuple[np.ndarray, list[str]]:
    """Compute species weights for every (q, band) in a BandPath.

    Returns (weights, species_order) where weights is
    (N_q, N_bands, N_species).
    """
    return species_weights_array(band.eigenvectors, band.material.symbols)


def mesh_species_weights(mesh: MeshGrid) -> tuple[np.ndarray, list[str]]:
    """Compute species weights for every (q, band) on a MeshGrid."""
    if mesh.eigenvectors is None:
        raise ValueError("MeshGrid has no eigenvectors")
    return species_weights_array(mesh.eigenvectors, mesh.material.symbols)


def band_path_ipr(band: BandPath) -> np.ndarray:
    """IPR for every (q, band) of a BandPath, shape (N_q, N_bands)."""
    return inverse_participation_ratio_array(band.eigenvectors)


def mesh_ipr(mesh: MeshGrid) -> np.ndarray:
    """IPR for every (q, band) of a MeshGrid, shape (N_q, N_bands)."""
    if mesh.eigenvectors is None:
        raise ValueError("MeshGrid has no eigenvectors")
    return inverse_participation_ratio_array(mesh.eigenvectors)
