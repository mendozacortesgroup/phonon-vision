#!/usr/bin/env python3
"""Re-emit the frozen kinematics golden into a web-consumable JSON (for 1b).

This is a PURE RE-EMISSION from the two SHIPPED fixtures
``tests/golden_kinematics.npz`` and ``tests/golden_kinematics.json``. It does NOT
invoke ``capture_golden.py``'s regeneration path and reads NOTHING from the user
uploads area (the original ``band.yaml`` inputs are not in the repo and are not
needed here) — only the two committed fixture files above.

The npz ``disp_phi`` were computed from the CANONICALIZED mode, so we canonicalize
each raw eigenvector here with the same ``canonicalize_eigenvector_phase`` the rest
of the toolchain uses; a JS animator that consumes the canonical eigenvector then
reproduces ``disp_phi`` exactly. ``disp_phi`` itself is copied VERBATIM from the
npz — the physics is never recomputed a second way.

Run from the repo root (or anywhere):

    python3 tests/emit_web_golden.py

writes ``tests/golden_kinematics_web.json`` next to the fixtures.
"""
import json
import os

import numpy as np

from phonon_core.kinematics import canonicalize_eigenvector_phase

_HERE = os.path.dirname(os.path.abspath(__file__))
_NPZ = os.path.join(_HERE, "golden_kinematics.npz")
_JSON = os.path.join(_HERE, "golden_kinematics.json")
_OUT = os.path.join(_HERE, "golden_kinematics_web.json")

# the five fixture modes, in the canonical order
MODES = ("gamma_acoustic", "gamma_optical", "m_point", "k_point", "generic")


def emit_web_golden(npz_path=_NPZ, json_path=_JSON, out_path=_OUT):
    """Build golden_kinematics_web.json from the frozen fixtures; return its path."""
    z = np.load(npz_path)
    with open(json_path, encoding="utf-8") as f:
        g = json.load(f)

    src_modes = g["modes"]
    src_inputs = g["inputs"]
    # only the inputs the animator needs (amplitude, phi, mass_weighted)
    web_inputs = {
        "amplitude": src_inputs["amplitude"],
        "phi": src_inputs["phi"],
        "mass_weighted": src_inputs["mass_weighted"],
    }
    # supercell slot table (shared by every mode): flat atom index + integer cell
    slots = {"j": z["_slot_j"].tolist(), "n_int": z["_slot_n_int"].tolist()}
    # primitive-atom masses, SAME order as material_symbols / the eigenvector atom
    # axis, taken VERBATIM from the npz (no element->mass table). The mass-weighted
    # displacement needs these for the JS gate to recompute.
    masses = z["_material_masses"].tolist()

    out = {
        "fixture_version": g.get("fixture_version"),
        "source": "re-emitted from golden_kinematics.{npz,json}",
        "phonon_core_version": g.get("phonon_core_version"),
        "material_symbols": g.get("material_symbols"),
        "masses": masses,
        "inputs": web_inputs,
        "conventions": {
            "displacement": ("u = A*Re[e*exp(i(2pi*n.q_frac + phi))]/sqrt(m); "
                             "Dove 4.16, mass-weighted"),
            "bloch_phase": ("2*pi * (n_int . q_frac); q fractional, n integer, "
                            "no metric"),
            "eigenvector_phase": ("canonicalized (largest component real-positive "
                                  "at phi=0)"),
            "slots": ("j = flat atom-in-supercell index; n_int = integer cell "
                      "translation (3,) for that slot"),
        },
        "modes": {},
    }

    for m in MODES:
        md = src_modes[m]
        # canonicalize the RAW eigenvector (Na,3) -> the same canonical mode that
        # produced disp_phi; the JS decoder must consume this exact eigenvector.
        ev_canon = canonicalize_eigenvector_phase(z[f"{m}__eigvec_raw"])
        out["modes"][m] = {
            "q_frac": md["q_point"],
            "band_index": md["band_index"],
            "frequency_THz": md["frequency_THz"],
            "is_finite_q": md["is_finite_q"],
            "eigenvector_real": ev_canon.real.tolist(),   # (Na,3)
            "eigenvector_imag": ev_canon.imag.tolist(),    # (Na,3)
            "slots": slots,
            "inputs": web_inputs,
            "disp_phi": z[f"{m}__disp_phi"].tolist(),      # VERBATIM from the npz
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out_path


if __name__ == "__main__":
    print(emit_web_golden())
