"""Phonon kinematics: displacement math, Bloch phases, canonical phase.

The displacement of atom ``s`` in unit cell ``ℓ`` (whose origin sits at
the lattice vector ``R_ℓ``), at global phase ``φ``, is::

    u_{ℓs}(φ) = A · Re[ e_s(q) · exp(i (q·R_ℓ + φ)) ] / sqrt(m_s)

When ``φ = -ωt`` this is the Bloch wave at time t. We pass ``φ`` directly
so the same machinery handles both animation (sweep φ) and frozen snapshots
(set φ to a fixed value).

Imaginary frequencies
---------------------
phonopy emits negative frequencies for soft / unstable modes (the dynamical
matrix has negative eigenvalues). These represent real physics — symmetry-
breaking instabilities are how phonopy tells you a structure isn't dynamically
stable. We visualize them at ``|ω|`` so the user can see the symmetry of the
mode. The ``PhononMode.is_imaginary`` flag is the right signal for the UI to
mark them clearly (typically in red).
"""

from __future__ import annotations

import numpy as np

from phonon_core.types import Material, PhononMode


# ============================================================
# Bloch phase computation
# ============================================================
def compute_bloch_phases(
    q_point: np.ndarray,
    uc_indices: np.ndarray,
) -> np.ndarray:
    """Per-slot Bloch phase ``2π · q · n`` where ``n`` is the integer UC index.

    Parameters
    ----------
    q_point : (3,) float
        Wavevector in fractional reciprocal-lattice coordinates.
    uc_indices : (N_slots, 3) int
        For each replicated atom slot, the integer (n1, n2, n3) of its unit cell.

    Returns
    -------
    (N_slots,) float
        Bloch phases in radians.

    Notes
    -----
    Because q is in fractional reciprocal coords and n is in lattice basis,
    ``q · R = 2π · q_frac · n`` exactly. No reciprocal-lattice metric needed.
    """
    return 2.0 * np.pi * (uc_indices.astype(np.float64) @ np.asarray(q_point, dtype=np.float64))


# ============================================================
# Canonical phase convention
# ============================================================
def canonicalize_eigenvector_phase(eigenvector: np.ndarray) -> np.ndarray:
    """Rotate a complex eigenvector by a global phase ``e^{iα}`` so that
    snapshots at ``φ = 0`` are reproducible across diagonalization runs.

    Convention
    ----------
    Pick α such that the largest-magnitude Cartesian component (over all atoms
    and directions) has zero imaginary part and a positive real part after
    rephasing. After this, ``φ = 0`` corresponds to "the mode at maximum
    extension along its principal direction".

    A pure rephasing does not change the physical mode — animations that sweep
    all φ in [0, 2π) trace the same trajectory. What it does fix is the
    arbitrary phase that numerical diagonalization leaves in the eigenvector,
    which would otherwise make instantaneous figures non-reproducible.

    For degenerate modes (multiple bands with identical frequency), the
    eigenvectors are only defined up to a unitary mixing within the degenerate
    subspace, which this single-mode rephasing cannot resolve. That's a
    fundamental ambiguity, not a bug; document it in any per-mode figure.
    """
    eig = np.asarray(eigenvector, dtype=np.complex128)
    flat = eig.reshape(-1)
    idx = int(np.argmax(np.abs(flat)))
    pivot = flat[idx]
    mag = abs(pivot)
    if mag < 1e-12:
        # Degenerate / zero eigenvector — nothing meaningful to rotate.
        return eig.copy()
    # Want: pivot · e^{-iα} = +|pivot|. So e^{-iα} = conj(pivot)/|pivot|.
    rotation = np.conj(pivot) / mag
    return eig * rotation


def canonicalize_mode(mode: PhononMode) -> PhononMode:
    """Return a copy of ``mode`` with its eigenvector phase-canonicalized."""
    return PhononMode(
        q_point=mode.q_point,
        band_index=mode.band_index,
        frequency=mode.frequency,
        eigenvector=canonicalize_eigenvector_phase(mode.eigenvector),
    )


# ============================================================
# Displacement computation
# ============================================================
def compute_displacements_at_phase(
    mode: PhononMode,
    material: Material,
    amplitude: float,
    phase: float,
    *,
    atom_indices: np.ndarray | None = None,
    bloch_phases: np.ndarray | None = None,
    mass_weighted: bool = True,
) -> np.ndarray:
    """Cartesian displacement vectors at a single global phase.

    For the single primitive cell::

        u_s = A · Re[ e_s · exp(i φ) ] / sqrt(m_s)

    For a supercell of replicated UCs (Bloch wave)::

        u_{ℓs} = A · Re[ e_s · exp(i (q·R_ℓ + φ)) ] / sqrt(m_s)

    The supercell case is selected by passing both ``atom_indices`` (which UC
    atom each slot represents) and ``bloch_phases`` (precomputed q·R_ℓ).

    Parameters
    ----------
    mode : PhononMode
    material : Material
        Used for atomic masses.
    amplitude : float
        Overall amplitude scale; physically the mass-weighted normal-mode
        coordinate amplitude. The user typically chooses this to make
        sub-Ångström motion visible (factor ~10).
    phase : float
        Global phase φ in radians. For animation, sweep ``[0, 2π)``.
    atom_indices : (N_slots,) int, optional
        For supercell mode: index of the primitive-cell atom each slot
        represents. If omitted, falls back to single-UC mode with N_atoms slots.
    bloch_phases : (N_slots,) float, optional
        For supercell mode: precomputed Bloch phase q·R_ℓ for each slot.
        Required iff ``atom_indices`` is given.
    mass_weighted : bool
        If True (default), divide by sqrt(m_s). Most phonopy eigenvectors are
        already mass-weighted-normalized, so the trajectory in Cartesian space
        does include 1/sqrt(m). Disable if you have raw Cartesian-amplitude
        eigenvectors.

    Returns
    -------
    (N_slots, 3) float
        Real-valued Cartesian displacement for each slot.
    """
    eig = mode.eigenvector  # (N_atoms, 3) complex

    if atom_indices is None:
        # Single primitive cell
        # Re[e * exp(iφ)] for each (atom, direction)
        phase_factor = np.exp(1j * phase)
        disp_complex = eig * phase_factor              # (N_atoms, 3)
        disp = np.real(disp_complex)
        if mass_weighted:
            disp = disp / np.sqrt(material.masses)[:, np.newaxis]
        return amplitude * disp

    # Supercell with Bloch phases
    if bloch_phases is None:
        raise ValueError("bloch_phases is required when atom_indices is given")

    atom_indices = np.asarray(atom_indices, dtype=np.int64)
    bloch_phases = np.asarray(bloch_phases, dtype=np.float64)

    if atom_indices.shape != bloch_phases.shape:
        raise ValueError(
            f"atom_indices {atom_indices.shape} and bloch_phases "
            f"{bloch_phases.shape} must match"
        )

    # Broadcast eigenvectors to each slot
    slot_eig = eig[atom_indices]                         # (N_slots, 3) complex

    total_phase = bloch_phases + phase                   # (N_slots,)
    phase_factor = np.exp(1j * total_phase)[:, np.newaxis]  # (N_slots, 1)

    disp = np.real(slot_eig * phase_factor)              # (N_slots, 3)

    if mass_weighted:
        slot_masses = material.masses[atom_indices]
        disp = disp / np.sqrt(slot_masses)[:, np.newaxis]

    return amplitude * disp


# ============================================================
# Trajectory (full cycle) — useful for caching frames in advance
# ============================================================
def compute_trajectory(
    mode: PhononMode,
    material: Material,
    amplitude: float,
    n_frames: int,
    *,
    atom_indices: np.ndarray | None = None,
    bloch_phases: np.ndarray | None = None,
    mass_weighted: bool = True,
    endpoint: bool = False,
) -> np.ndarray:
    """Compute displacements at every frame of one closed loop.

    Returns shape ``(n_frames, N_slots, 3)``. With ``endpoint=False`` (default)
    the phases span ``[0, 2π)`` so the loop ``frame[n-1] → frame[0]`` is
    seamless — useful for GIF / WebM export.
    """
    phases = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=endpoint)
    if atom_indices is not None:
        n_slots = len(atom_indices)
    else:
        n_slots = material.num_atoms
    out = np.empty((n_frames, n_slots, 3), dtype=np.float64)
    for i, phi in enumerate(phases):
        out[i] = compute_displacements_at_phase(
            mode, material, amplitude, float(phi),
            atom_indices=atom_indices,
            bloch_phases=bloch_phases,
            mass_weighted=mass_weighted,
        )
    return out
