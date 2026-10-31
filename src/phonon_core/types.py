"""Typed data containers for phonon calculations.

The conventions used throughout this package match phonopy's:

- Real-space lattice vectors are stored row-wise in a (3,3) matrix:
  cartesian = fractional @ lattice
- Reciprocal lattice vectors are stored *without* the 2π factor (phonopy convention).
- q-points are in fractional reciprocal-lattice coordinates.
- Frequencies are in THz (signed; negative = imaginary, indicating a soft mode).
- Eigenvectors are complex, shape (N_atoms, 3), in Cartesian coordinates.
- Mass is in atomic mass units (AMU).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Material:
    """The unperturbed crystal at equilibrium.

    Attributes
    ----------
    lattice : (3, 3) float
        Real-space lattice vectors stored row-wise (a1, a2, a3) in Å.
    reciprocal_lattice : (3, 3) float
        Reciprocal lattice vectors row-wise, *without* 2π factor (phonopy convention).
    symbols : list[str]
        Chemical symbol of each atom in the primitive cell.
    masses : (N_atoms,) float
        Atomic masses in AMU.
    fractional_coords : (N_atoms, 3) float
        Fractional coordinates of each atom within the primitive cell.
    """

    lattice: np.ndarray
    reciprocal_lattice: np.ndarray
    symbols: list[str]
    masses: np.ndarray
    fractional_coords: np.ndarray

    def __post_init__(self) -> None:
        self.lattice = np.asarray(self.lattice, dtype=np.float64)
        self.reciprocal_lattice = np.asarray(self.reciprocal_lattice, dtype=np.float64)
        self.masses = np.asarray(self.masses, dtype=np.float64)
        self.fractional_coords = np.asarray(self.fractional_coords, dtype=np.float64)

        # Wrap fractional coordinates into the canonical [0, 1) cell:
        #   r := r - floor(r)
        # Phonopy/VASP inputs are normally already wrapped, so this is a no-op
        # on canonical fixtures (positions_were_wrapped stays False). It exists
        # to surface non-canonical input at load time and to give expansion a
        # single, well-defined origin cell. Expansion (phonon_core.expand) and
        # the Bloch phase both assume n_int is the *integer* cell offset from
        # this canonical cell, so wrapping here keeps n_int meaningful.
        wrapped = self.fractional_coords - np.floor(self.fractional_coords)
        self.positions_were_wrapped: bool = bool(
            np.any(wrapped != self.fractional_coords)
        )
        self.fractional_coords = wrapped

        if self.lattice.shape != (3, 3):
            raise ValueError(f"lattice must be (3,3), got {self.lattice.shape}")
        if self.reciprocal_lattice.shape != (3, 3):
            raise ValueError(f"reciprocal_lattice must be (3,3), got {self.reciprocal_lattice.shape}")
        if len(self.symbols) != len(self.masses):
            raise ValueError(
                f"symbols/masses length mismatch: {len(self.symbols)} vs {len(self.masses)}"
            )
        if self.fractional_coords.shape != (len(self.symbols), 3):
            raise ValueError(
                f"fractional_coords must be (N_atoms,3); got {self.fractional_coords.shape}"
            )

    @property
    def num_atoms(self) -> int:
        return len(self.symbols)

    @property
    def num_bands(self) -> int:
        """Number of phonon branches = 3 × N_atoms."""
        return 3 * self.num_atoms

    @property
    def cartesian_coords(self) -> np.ndarray:
        """(N_atoms, 3) atom positions in Cartesian coordinates [Å]."""
        return self.fractional_coords @ self.lattice

    @property
    def unique_species(self) -> list[str]:
        """List of distinct chemical symbols in order of first appearance."""
        seen: set[str] = set()
        result: list[str] = []
        for s in self.symbols:
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result

    def atom_indices_by_species(self) -> dict[str, list[int]]:
        """Map each species symbol → list of atom indices belonging to it."""
        groups: dict[str, list[int]] = {}
        for i, s in enumerate(self.symbols):
            groups.setdefault(s, []).append(i)
        return groups


@dataclass
class PhononMode:
    """A single eigenstate of the dynamical matrix at one (q, band).

    Attributes
    ----------
    q_point : (3,) float
        Wavevector in fractional reciprocal-lattice coordinates.
    band_index : int
        0-based band index (0 .. 3·N_atoms - 1).
    frequency : float
        Phonon frequency in THz. Negative values indicate imaginary frequencies
        (soft modes / dynamical instabilities); use ``is_imaginary`` to check.
    eigenvector : (N_atoms, 3) complex128
        Polarization vector: complex displacement direction per atom, normalized
        such that Σ_s |e_s|² = 1.
    """

    q_point: np.ndarray
    band_index: int
    frequency: float
    eigenvector: np.ndarray

    def __post_init__(self) -> None:
        self.q_point = np.asarray(self.q_point, dtype=np.float64)
        self.eigenvector = np.asarray(self.eigenvector, dtype=np.complex128)
        if self.q_point.shape != (3,):
            raise ValueError(f"q_point must be shape (3,); got {self.q_point.shape}")
        if self.eigenvector.ndim != 2 or self.eigenvector.shape[1] != 3:
            raise ValueError(
                f"eigenvector must be (N_atoms,3); got {self.eigenvector.shape}"
            )

    @property
    def is_imaginary(self) -> bool:
        """True if the frequency is negative (soft / unstable mode)."""
        return self.frequency < 0.0

    @property
    def angular_frequency(self) -> float:
        """ω = 2π·|ν| in rad/ps (since ν is in THz = 1/ps)."""
        return 2.0 * np.pi * abs(self.frequency)

    @property
    def period(self) -> float:
        """Oscillation period T = 1/|ν| in picoseconds. Infinite for ν=0."""
        if abs(self.frequency) < 1e-15:
            return float("inf")
        return 1.0 / abs(self.frequency)


@dataclass
class BandPath:
    """1D path through q-space (output of phonopy --band).

    Q-points are arranged along high-symmetry segments. Use ``segment_nqpoint``
    and ``labels`` to find segment boundaries for plotting tick labels.

    Attributes
    ----------
    material : Material
    q_points : (N_q, 3) float
        Fractional q-points along the path.
    distances : (N_q,) float
        Cumulative path distance from the start, in Å⁻¹.
        Use these as the x-axis when plotting dispersion.
    frequencies : (N_q, N_bands) float
        Frequencies in THz.
    eigenvectors : (N_q, N_bands, N_atoms, 3) complex
        Per-mode polarization vectors.
    segment_nqpoint : list[int]
        Number of q-points in each segment of the path. Sum equals N_q.
    labels : list[(str, str)]
        For each segment, the (start_label, end_label) pair.
        E.g. [('Γ', 'M'), ('M', 'K'), ('K', 'Γ')].
    """

    material: Material
    q_points: np.ndarray
    distances: np.ndarray
    frequencies: np.ndarray
    eigenvectors: np.ndarray
    segment_nqpoint: list[int]
    labels: list[tuple[str, str]]

    def __post_init__(self) -> None:
        self.q_points = np.asarray(self.q_points, dtype=np.float64)
        self.distances = np.asarray(self.distances, dtype=np.float64)
        self.frequencies = np.asarray(self.frequencies, dtype=np.float64)
        self.eigenvectors = np.asarray(self.eigenvectors, dtype=np.complex128)

        n_q = len(self.q_points)
        n_b = self.material.num_bands
        n_a = self.material.num_atoms
        if self.frequencies.shape != (n_q, n_b):
            raise ValueError(f"frequencies must be ({n_q}, {n_b}); got {self.frequencies.shape}")
        if self.eigenvectors.shape != (n_q, n_b, n_a, 3):
            raise ValueError(
                f"eigenvectors must be ({n_q}, {n_b}, {n_a}, 3); got {self.eigenvectors.shape}"
            )
        if sum(self.segment_nqpoint) != n_q:
            raise ValueError(
                f"segment_nqpoint sum {sum(self.segment_nqpoint)} != n_q {n_q}"
            )

    @property
    def num_qpoints(self) -> int:
        return len(self.q_points)

    def get_mode(self, q_index: int, band_index: int) -> PhononMode:
        """Extract a single mode by (q_index, band_index)."""
        return PhononMode(
            q_point=self.q_points[q_index],
            band_index=band_index,
            frequency=float(self.frequencies[q_index, band_index]),
            eigenvector=self.eigenvectors[q_index, band_index],
        )

    def tick_locations(self) -> tuple[list[float], list[str]]:
        """Return (positions, labels) for x-axis high-symmetry ticks.

        Returns
        -------
        positions : list[float]
            Path distances at which to draw vertical lines and labels.
        labels : list[str]
            Labels for each position; consecutive identical labels are merged
            (e.g., ['Γ', 'M', 'K', 'Γ'] rather than ['Γ', 'M', 'M', 'K', 'K', 'Γ']).
        """
        # First label is the start of the first segment
        positions: list[float] = [float(self.distances[0])]
        labels: list[str] = [self.labels[0][0]]
        cum = 0
        for i, n in enumerate(self.segment_nqpoint):
            cum += n
            end_label = self.labels[i][1]
            # The end of segment i and start of segment i+1 are the same q-point,
            # so we add only the end label here.
            positions.append(float(self.distances[cum - 1]))
            # If the end label of segment i matches the start label of segment i+1,
            # merge them. Otherwise show both joined.
            if i + 1 < len(self.labels):
                next_start = self.labels[i + 1][0]
                if next_start != end_label:
                    labels.append(f"{end_label}|{next_start}")
                else:
                    labels.append(end_label)
            else:
                labels.append(end_label)
        return positions, labels


@dataclass
class MeshGrid:
    """3D q-grid (output of phonopy --mesh, MESH_SYMMETRY=.FALSE.).

    The q-points form a regular grid on the BZ. Use ``mesh`` for the grid
    dimensions [nx, ny, nz]. For 2D materials, nz is typically 1.

    Attributes
    ----------
    material : Material
    mesh : (int, int, int)
        Grid dimensions (nx, ny, nz).
    q_points : (N_q, 3) float
        Fractional q-points; N_q = nx·ny·nz when MESH_SYMMETRY=.FALSE.
    frequencies : (N_q, N_bands) float
    eigenvectors : (N_q, N_bands, N_atoms, 3) complex or None
        Eigenvectors are optional in mesh.yaml; None if not present.
    weights : (N_q,) int or None
        Symmetry weights. All 1 when MESH_SYMMETRY=.FALSE.
    """

    material: Material
    mesh: tuple[int, int, int]
    q_points: np.ndarray
    frequencies: np.ndarray
    eigenvectors: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.q_points = np.asarray(self.q_points, dtype=np.float64)
        self.frequencies = np.asarray(self.frequencies, dtype=np.float64)
        if self.eigenvectors is not None:
            self.eigenvectors = np.asarray(self.eigenvectors, dtype=np.complex128)
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=np.int64)

        nx, ny, nz = self.mesh
        n_q = len(self.q_points)
        if n_q != nx * ny * nz:
            # Allowed when MESH_SYMMETRY=.TRUE.; we just warn via a flag.
            pass

        n_b = self.material.num_bands
        if self.frequencies.shape != (n_q, n_b):
            raise ValueError(f"frequencies must be ({n_q}, {n_b}); got {self.frequencies.shape}")
        if self.eigenvectors is not None:
            n_a = self.material.num_atoms
            if self.eigenvectors.shape != (n_q, n_b, n_a, 3):
                raise ValueError(
                    f"eigenvectors must be ({n_q}, {n_b}, {n_a}, 3); got {self.eigenvectors.shape}"
                )

    @property
    def num_qpoints(self) -> int:
        return len(self.q_points)

    @property
    def has_full_grid(self) -> bool:
        """True if the grid is unfolded (MESH_SYMMETRY=.FALSE., one q per cell)."""
        nx, ny, nz = self.mesh
        return self.num_qpoints == nx * ny * nz

    def reshape_grid(self, array: np.ndarray) -> np.ndarray:
        """Reshape a (N_q, ...) array into (nz, ny, nx, ...) grid layout.

        Phonopy emits q-points in nested order with the *first* fractional
        coordinate varying fastest. We use (nz, ny, nx, ...) = ('slow', ..., 'fast')
        which lets you do ``array[iz, iy, ix]`` naturally.
        """
        if not self.has_full_grid:
            raise ValueError(
                "Cannot reshape to grid: this MeshGrid is symmetry-reduced. "
                "Re-run phonopy with MESH_SYMMETRY=.FALSE."
            )
        nx, ny, nz = self.mesh
        return array.reshape(nz, ny, nx, *array.shape[1:])

    def get_mode(self, q_index: int, band_index: int) -> PhononMode:
        """Extract a single mode."""
        if self.eigenvectors is None:
            raise ValueError("This MeshGrid has no eigenvectors. Re-run with EIGENVECTORS=.TRUE.")
        return PhononMode(
            q_point=self.q_points[q_index],
            band_index=band_index,
            frequency=float(self.frequencies[q_index, band_index]),
            eigenvector=self.eigenvectors[q_index, band_index],
        )
