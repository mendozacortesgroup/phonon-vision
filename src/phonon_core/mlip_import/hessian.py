"""Convert an MLIP supercell Hessian into the phonopy triple.

Scope
-----
The input is the full Cartesian second-derivative matrix of a SUPERCELL whose
structure an existing ``phonopy.yaml`` already describes — the common case in
practice: the MLIP is evaluated on exactly the supercell a DFT calculation
used, so the two force-constant sets are directly comparable on the same
footing. The converter writes ``phonopy.yaml`` (copied verbatim — the
structure is by definition unchanged) + ``FORCE_CONSTANTS`` (full phonopy
text format), which every engine in this package consumes.

Pure numpy — deliberately NO phonopy import: phonopy's full-format
FORCE_CONSTANTS *is* the supercell Hessian in eV/Å², so the conversion is a
validation + layout problem, not a physics transform.

Conventions (each checked or loudly refused, none silently assumed)
-------------------------------------------------------------------
* **Units are eV/Å²** — the ASE-calculator world's natural Hessian units.
  The ``units`` parameter exists so the requirement is visible at the call
  site; anything else is refused (converting is the caller's job, and a
  silent wrong-unit table is the failure mode this refuses to allow).
* **Atom order must be the phonopy.yaml supercell order.** A permutation can
  be supplied (``atom_order[k]`` = which Hessian row/column holds supercell
  atom k). The trap this guards is a field lesson: ONE swapped supercell
  pair moved frequencies by 0.54 THz at generic q while staying invisible at
  Γ — the classic q-dependent convention error. The negative control in the
  test-suite reproduces exactly that signature.
* **Symmetry**: a Hessian is symmetric up to the MLIP's own noise. The
  asymmetry ‖H − Hᵀ‖∞ is measured; below ``asym_tol`` it is symmetrized and
  REPORTED, above it the conversion refuses (that much asymmetry means the
  input is not a second-derivative matrix).
* **Translation consistency is EXPECTED, not enforced.** A Hessian from a
  periodic supercell calculation satisfies Φ[Ti,Tj] = Φ[i,j] for supercell
  translations up to solver noise, and the engines rely on it: they read
  only the primitive rows and hermitize, so for a strongly non-TI input the
  Γ translation modes do not close and no sum-rule correction can help
  (the test-suite documents the mechanism). Garbage that is not even
  approximately TI is out of scope by construction.
* **Acoustic sum rule**: MLIP Hessians usually violate the ASR at the
  potential's noise level, which shows up as spurious Γ acoustic
  frequencies. ``enforce_asr=True`` applies a symmetric projection that
  zeroes row AND column sums while preserving pair symmetry (the textbook
  diagonal-only correction breaks it — see the inline note); the drift is
  measured and reported either way. (An old bundled dataset's broken sum
  rule once read as a −0.129 THz "instability" that was pure artifact.)
"""

from __future__ import annotations

import os
import shutil

import numpy as np

from phonon_core.parsers._yaml_loader import load_yaml

__all__ = ["convert_hessian"]

OUT_YAML = "phonopy.yaml"
OUT_FC = "FORCE_CONSTANTS"


def _supercell_natom(phonopy_yaml: str) -> int:
    data = load_yaml(phonopy_yaml)
    sc = data.get("supercell")
    if not (isinstance(sc, dict) and "points" in sc):
        raise ValueError(f"{phonopy_yaml}: no 'supercell' block — regenerate "
                         "with phonopy >= 2.x (Phonopy.save writes it)")
    return len(sc["points"])


def _write_force_constants_full(path: str, fc: np.ndarray) -> None:
    """Full phonopy text format: header 'N N', then per (i, j) a '<i> <j>'
    line and a 3x3 block, 1-based indices, high precision."""
    n = fc.shape[0]
    with open(path, "w") as fh:
        fh.write(f"{n} {n}\n")
        for i in range(n):
            for j in range(n):
                fh.write(f"{i + 1:4d} {j + 1:4d}\n")
                for a in range(3):
                    fh.write("  %21.15f %21.15f %21.15f\n"
                             % tuple(fc[i, j, a]))


def convert_hessian(hessian, phonopy_yaml, outdir, *, units="eV/A^2",
                    atom_order=None, asym_tol=1e-3, enforce_asr=False):
    """Write the phonopy triple for an MLIP supercell Hessian.

    Parameters
    ----------
    hessian : (3N, 3N) or (N, N, 3, 3) array — Cartesian second derivatives
        of the supercell energy, eV/Å², in the phonopy.yaml supercell atom
        order (or ``atom_order``'s).
    phonopy_yaml : path — the structure authority; copied verbatim.
    outdir : existing directory to write ``phonopy.yaml`` + ``FORCE_CONSTANTS``.
    units : must be ``"eV/A^2"`` (refused otherwise — see module docstring).
    atom_order : optional (N,) int — ``atom_order[k]`` is the Hessian
        row/column holding supercell atom k of the phonopy.yaml order.
    asym_tol : max allowed ‖H − Hᵀ‖∞ (eV/Å²) before refusal.
    enforce_asr : apply the diagonal acoustic-sum-rule correction.

    Returns
    -------
    info : dict — ``n_supercell_atoms``, ``max_asymmetry``,
        ``asr_drift_before`` / ``asr_drift_after`` (max |Σ_j Φ_ij| component,
        eV/Å²), ``asr_enforced``, ``atom_order_applied``.
    """
    if units != "eV/A^2":
        raise ValueError(f"units must be 'eV/A^2' (got {units!r}); convert "
                         "before calling — a silent unit guess is refused")
    if not os.path.isdir(outdir):
        raise ValueError(f"outdir does not exist: {outdir}")

    n = _supercell_natom(phonopy_yaml)

    H = np.asarray(hessian, dtype=np.float64)
    if H.shape == (3 * n, 3 * n):
        fc = H.reshape(n, 3, n, 3).transpose(0, 2, 1, 3).copy()
    elif H.shape == (n, n, 3, 3):
        fc = H.copy()
    else:
        raise ValueError(f"hessian shape {H.shape} matches neither "
                         f"({3 * n}, {3 * n}) nor ({n}, {n}, 3, 3) for the "
                         f"{n}-atom supercell of {phonopy_yaml}")

    order_applied = False
    if atom_order is not None:
        order = np.asarray(atom_order, dtype=np.intp)
        if sorted(order.tolist()) != list(range(n)):
            raise ValueError("atom_order must be a permutation of the "
                             f"{n} supercell atoms")
        fc = fc[np.ix_(order, order)]
        order_applied = True

    # symmetry: measure, then symmetrize or refuse
    asym = float(np.abs(fc - fc.transpose(1, 0, 3, 2)).max())
    if asym > asym_tol:
        raise ValueError(
            f"Hessian asymmetry {asym:.3e} eV/A^2 exceeds asym_tol="
            f"{asym_tol:.1e} — this is not a second-derivative matrix "
            "(wrong displacement pairing, or unconverged autograd)")
    fc = 0.5 * (fc + fc.transpose(1, 0, 3, 2))

    # acoustic sum rule: report always, enforce on request.
    #
    # TWO wrong repairs are on record here, both found by the test-suite:
    # (i) the textbook diagonal correction (Φ_ii -= Σ_j Φ_ij) zeroes ROW
    # sums but breaks pair symmetry Φ_ij^ab = Φ_ji^ba, so COLUMN sums go
    # nonzero and the engine's hermitization (D+D†)/2 re-imports the
    # violation — D(Γ) loses its translation null vectors (measured 0.069
    # eV/Å² leakage, fake Γ "acoustics" at 1-3 THz); (ii) alternating that
    # correction with re-symmetrization does NOT converge — the antisym
    # diagonal residue is a 2-CYCLE, flipping sign each pass (measured:
    # rows 9e-16, asym stuck at 5.12 for 200 passes).
    #
    # The correct repair is the symmetric PROJECTION
    #     Φ'_ij = Φ_ij − r_i/N − c_j/N + T/N²
    # (r_i row sums, c_j column sums, T the grand sum): rows AND columns are
    # exactly zero in one shot, and for a pair-symmetric input (c_j = r_jᵀ,
    # T = Tᵀ) the result is pair-symmetric — verified algebraically and
    # asserted below.
    drift_before = float(np.abs(fc.sum(axis=1)).max())
    if enforce_asr:
        r = fc.sum(axis=1)                             # (N, 3, 3) row sums
        c = fc.sum(axis=0)                             # (N, 3, 3) column sums
        t = fc.sum(axis=(0, 1))                        # (3, 3) grand sum
        fc = fc - r[:, None] / n - c[None, :] / n + t / (n * n)
        asr_asym = float(np.abs(fc - fc.transpose(1, 0, 3, 2)).max())
        col_after = float(np.abs(fc.sum(axis=0)).max())
        if max(asr_asym, col_after) > 1e-12:           # pragma: no cover
            raise RuntimeError(
                f"ASR projection self-check failed (asym {asr_asym:.2e}, "
                f"cols {col_after:.2e})")
    drift_after = float(np.abs(fc.sum(axis=1)).max())

    shutil.copyfile(phonopy_yaml, os.path.join(outdir, OUT_YAML))
    _write_force_constants_full(os.path.join(outdir, OUT_FC), fc)

    return {
        "n_supercell_atoms": int(n),
        "max_asymmetry": asym,
        "asr_drift_before": drift_before,
        "asr_drift_after": drift_after,
        "asr_enforced": bool(enforce_asr),
        "atom_order_applied": order_applied,
    }
