#!/usr/bin/env python3
"""Generate the golden kinematics fixtures — the frozen contract that any future
importer or the JS web-kinematics engine must reproduce.

From a trusted phonopy ``band.hdf5`` it selects a FIXED, documented set of modes
that span Γ AND finite q (a Γ acoustic, a Γ optical, an M-point, a K-point, and
one generic low-symmetry q), and for each runs phonon_core's EXISTING kinematics
at a fixed amplitude / phase / small supercell, freezing:

  tests/golden_kinematics.npz   displacement arrays + raw/canonical eigenvectors
                                + per-slot atom-index/Bloch arrays + material
  tests/golden_kinematics.json  conventions block + exact input spec per mode
                                + phonon_core version + source provenance

The finite-q modes are non-negotiable: they are the only ones that expose a
per-atom phase-convention error (invisible at Γ and to any |e|-based metric).

Regenerate (needs the band.hdf5; default path is the sandbox upload):

    HDF5_USE_FILE_LOCKING=FALSE python tools/gen_golden.py \
        --band /path/to/2d_WSe2_band.hdf5

This reads phonon_core but does NOT modify it; the engine is the reference.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "src"))

from phonon_core import __version__, load_band_hdf5
from phonon_core.kinematics import (
    canonicalize_mode, compute_bloch_phases, compute_displacements_at_phase,
    compute_trajectory,
)
from phonon_core.expand import expand_cell

# --- the FIXED mode set (WSe2 band path Γ(0)-M(201)-K(403)-Γ(605), 3 atoms) ---
#     label -> (q_index, band_index). band_index chosen for a clear, mostly
#     non-degenerate representative; the finite-q trio (M, K, generic) is what
#     guards the phase convention. 'generic' is the low-symmetry self-test base.
MODES = {
    "gamma_acoustic": (0, 2),     # acoustic branch at Γ (≈0 THz)
    "gamma_optical":  (0, 8),     # highest optical branch at Γ
    "m_point":        (201, 3),   # M = (1/2, 0, 0)
    "k_point":        (403, 3),   # K = (1/3, 1/3, 0)
    "generic":        (100, 4),   # generic low-symmetry q on Γ–M
}

# fixed kinematics inputs (documented in the sidecar)
AMPLITUDE = 1.5
PHI = 0.7                # a non-trivial fixed global phase
SUPERCELL = (2, 2, 1)    # small; 2D material → tile in-plane
N_FRAMES = 16            # trajectory frames over [0, 2π)
MASS_WEIGHTED = True

CONVENTIONS = {
    "displacement_formula": "u_{ls}(phi) = A * Re[ e_s * exp(i (q.R_l + phi)) ] / sqrt(m_s)",
    "mass_weighting": "displacement divided by sqrt(m_s); masses in AMU",
    "canonical_phase": ("rotate the eigenvector by a global e^{i alpha} so the "
                        "largest-magnitude Cartesian component is real and positive"),
    "eigenvector_normalization": "sum_s |e_s|^2 = 1 (mass-weighted normal-mode convention)",
    "q_convention": "fractional reciprocal-lattice coordinates; q.R = 2*pi * q_frac . n_int",
    "bloch_phase": "2*pi * q_frac . n_int (n_int = integer cell index)",
    "frequency_units": "THz, signed (negative = imaginary / soft mode)",
    "phase_grid": "phi_k = linspace(0, 2*pi, n_frames, endpoint=False); frame 0 is phi=0",
    "snapshot_phase_phi": "disp_phi arrays are at the single fixed phase PHI (radians)",
    "eigenvector_layout": "(N_atoms, 3) complex128, Cartesian, per primitive-cell atom",
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description="freeze golden kinematics fixtures")
    p.add_argument("--band", required=True,
                   help="trusted phonopy band.hdf5 to sample modes from")
    p.add_argument("--out-dir", default=os.path.join(_HERE, os.pardir, "tests"))
    args = p.parse_args(argv)
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    if not os.path.isfile(args.band):
        print(f"ERROR: band file not found: {args.band}", file=sys.stderr)
        return 2

    band = load_band_hdf5(args.band)
    mat = band.material
    ea = expand_cell(mat, *SUPERCELL, mode="strict")  # j, n_int per slot

    npz = {
        "_material_lattice": mat.lattice,
        "_material_reciprocal": mat.reciprocal_lattice,
        "_material_masses": mat.masses,
        "_material_fractional": mat.fractional_coords,
        "_slot_j": ea.j.astype(np.int64),
        "_slot_n_int": ea.n_int.astype(np.int64),
    }
    sidecar_modes = {}

    for name, (qi, bi) in MODES.items():
        mode = band.get_mode(qi, bi)               # raw eigenvector
        canon = canonicalize_mode(mode)            # canonical-phase rephasing
        bloch = compute_bloch_phases(canon.q_point, ea.n_int)

        disp_phi = compute_displacements_at_phase(
            canon, mat, AMPLITUDE, PHI,
            atom_indices=ea.j, bloch_phases=bloch, mass_weighted=MASS_WEIGHTED)
        disp_traj = compute_trajectory(
            canon, mat, AMPLITUDE, N_FRAMES,
            atom_indices=ea.j, bloch_phases=bloch, mass_weighted=MASS_WEIGHTED)

        npz[f"{name}__eigvec_raw"] = mode.eigenvector.astype(np.complex128)
        npz[f"{name}__eigvec_canon"] = canon.eigenvector.astype(np.complex128)
        npz[f"{name}__bloch"] = bloch.astype(np.float64)
        npz[f"{name}__disp_phi"] = disp_phi.astype(np.float64)
        npz[f"{name}__disp_traj"] = disp_traj.astype(np.float64)

        sidecar_modes[name] = {
            "q_index": int(qi),
            "band_index": int(bi),
            "q_point": [float(x) for x in mode.q_point],
            "frequency_THz": float(mode.frequency),
            "is_imaginary": bool(mode.is_imaginary),
            "is_finite_q": bool(np.linalg.norm(mode.q_point) > 1e-8),
            "n_atoms": int(mat.num_atoms),
            "n_slots": int(len(ea)),
        }
        print(f"  {name:14s} q={qi:4d} b={bi}  "
              f"q={np.round(mode.q_point,4).tolist()}  "
              f"f={mode.frequency:+.4f} THz  "
              f"{'finite-q' if sidecar_modes[name]['is_finite_q'] else 'Γ'}")

    out_dir = os.path.abspath(args.out_dir)
    npz_path = os.path.join(out_dir, "golden_kinematics.npz")
    json_path = os.path.join(out_dir, "golden_kinematics.json")
    np.savez(npz_path, **npz)

    sidecar = {
        "fixture_version": 1,
        "phonon_core_version": __version__,
        "source_file": os.path.basename(args.band),
        "source_sha256": _sha256(args.band),
        "material_symbols": list(mat.symbols),
        "inputs": {
            "amplitude": AMPLITUDE,
            "phi": PHI,
            "supercell": list(SUPERCELL),
            "n_frames": N_FRAMES,
            "mass_weighted": MASS_WEIGHTED,
        },
        "conventions": CONVENTIONS,
        "modes": sidecar_modes,
    }
    with open(json_path, "w") as fh:
        json.dump(sidecar, fh, indent=2)

    print(f"\nWrote {npz_path}")
    print(f"Wrote {json_path}")
    print(f"phonon_core {__version__}; {len(MODES)} modes; supercell {SUPERCELL}; "
          f"{len(ea)} slots; finite-q modes: "
          f"{[n for n,m in sidecar_modes.items() if m['is_finite_q']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
