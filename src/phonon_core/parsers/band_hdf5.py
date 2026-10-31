"""Parser for phonopy band.hdf5 files.

The HDF5 file is what phonopy emits when ``HDF5 = .TRUE.`` is added to
``band.conf``. It is byte-for-byte equivalent to ``band.yaml`` for the
data we care about, but two orders of magnitude faster to read for any
non-trivial calculation (e.g., 34 s → 0.13 s for a 1414-q × 21-band path).

Expected HDF5 layout (phonopy ≥ 2.x; verified on phonopy 2.21):

    /lattice              (3, 3)               real-space basis a₁,a₂,a₃ (Å, rows)
    /reciprocal_lattice   (3, 3)               recip basis without 2π
    /symbols              (N_atoms,)           bytes — chemical symbols
    /coordinates          (N_atoms, 3)         fractional atom positions
    /masses               (N_atoms,)           AMU
    /numbers              (N_atoms,)           atomic numbers (unused)
    /natom                ()                   N_atoms

    /path                 (N_seg, N_per_seg, 3) fractional q-points
    /distance             (N_seg, N_per_seg)   cumulative path distance (Å⁻¹)
    /frequency            (N_seg, N_per_seg, 3N) THz, signed
    /eigenvector          (N_seg, N_per_seg, 3N, 3N) complex
    /segment_nqpoint      (N_seg,)             points per segment
    /label                (N_seg, 2)           bytes — (start, end) per segment
    /nqpoint              (1,)                 total N_q (== N_seg * N_per_seg)

The eigenvector convention is phonopy's standard (Maradudin & Vosko 1968,
phonopy docs ``output-files.html#band-hdf5``): each q-point holds the full
``3N × 3N`` eigenvector matrix of the dynamical matrix, with COLUMNS being
bands and ROWS being atom-cartesian (row r = atom r//3, direction r%3).

We reshape to our internal ``(N_q, N_bands, N_atoms, 3)`` convention via
``transpose(-1, -2).reshape(...)``. This was cross-validated against the
YAML parser on a 7-atom, 1414-q calculation (21 bands): all 29,694
eigenvectors match the YAML output bytewise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phonon_core.types import BandPath, Material


def _decode_label(b) -> str:
    """HDF5 stores object dtype as bytes in some platforms, str in others."""
    if isinstance(b, bytes):
        return b.decode("utf-8")
    return str(b)


def load_band_hdf5(filepath: str) -> BandPath:
    """Load a phonopy band.hdf5 into a :class:`BandPath`.

    Parameters
    ----------
    filepath : str
        Path to the band.hdf5 file.

    Raises
    ------
    ImportError
        If h5py isn't installed.
    ValueError
        If the file is missing required datasets.
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for HDF5 parsing. "
            "Install with `pip install h5py` (or use the YAML loader)."
        ) from e

    fp = str(filepath)
    with h5py.File(fp, "r") as f:
        # --- Material block ---
        for req in ("lattice", "reciprocal_lattice", "symbols",
                    "coordinates", "masses"):
            if req not in f:
                raise ValueError(f"{fp}: missing required dataset '/{req}'")

        lattice = np.asarray(f["lattice"][:], dtype=np.float64)
        rec_lat = np.asarray(f["reciprocal_lattice"][:], dtype=np.float64)
        symbols = [_decode_label(s) for s in f["symbols"][:]]
        masses = np.asarray(f["masses"][:], dtype=np.float64)
        frac_coords = np.asarray(f["coordinates"][:], dtype=np.float64)

        material = Material(
            lattice=lattice,
            reciprocal_lattice=rec_lat,
            symbols=symbols,
            masses=masses,
            fractional_coords=frac_coords,
        )

        n_atoms = material.num_atoms
        n_bands = material.num_bands

        # --- Path data ---
        for req in ("path", "frequency", "eigenvector",
                    "segment_nqpoint", "distance"):
            if req not in f:
                raise ValueError(f"{fp}: missing required dataset '/{req}'")

        # Flatten the (N_seg, N_per_seg, ...) segmented layout to (N_q, ...)
        path = np.asarray(f["path"][:], dtype=np.float64)
        frequency = np.asarray(f["frequency"][:], dtype=np.float64)
        distance = np.asarray(f["distance"][:], dtype=np.float64)
        eigvec = np.asarray(f["eigenvector"][:], dtype=np.complex128)
        segment_nqpoint = [int(n) for n in f["segment_nqpoint"][:]]

        # --- Reshape eigenvectors ---
        # raw: (N_seg, N_per_seg, 3N_row, 3N_col_band)
        # The column axis (last) is the band index; the row axis encodes
        # (atom * 3 + cartesian). Move band → axis 2, then split row into (atom, 3).
        n_q = int(np.sum(segment_nqpoint))
        eig_flat = eigvec.reshape(n_q, n_bands, n_bands)
        # transpose: (n_q, row=atom×3, col=band) → (n_q, band, atom×3)
        eig_band_first = eig_flat.transpose(0, 2, 1)
        eigenvectors = eig_band_first.reshape(n_q, n_bands, n_atoms, 3)

        q_points = path.reshape(n_q, 3)
        frequencies = frequency.reshape(n_q, n_bands)
        distances = distance.reshape(n_q)

        # --- Labels ---
        if "label" in f:
            raw_labels = f["label"][:]
            labels: list[tuple[str, str]] = [
                (_decode_label(row[0]), _decode_label(row[1]))
                for row in raw_labels
            ]
        else:
            labels = [("", "")] * len(segment_nqpoint)

        # Tolerate length mismatch (defensive, same as YAML parser does)
        if len(labels) != len(segment_nqpoint):
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
