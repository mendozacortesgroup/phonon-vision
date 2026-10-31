"""Web bundle (Deliverable 3): the single self-describing artifact a web viewer
loads.

A bundle is a directory holding

    document.json            JSON v1 document (schema-valid; the data)
    golden_kinematics.npz    displacement frames for a few bundled modes
    golden_kinematics.json   conventions + per-mode input spec + version
    bundle.json              manifest tying the above together

The golden files reuse the 0.2.0 golden-vector format (see tools/gen_golden.py),
so the *same* three-tier conformance logic validates a candidate engine. The
cross-language contract: a JS engine loads document.json, recomputes
displacements with the documented kinematics, and checks itself against the
bundled golden vectors — eigenvectors via check_against_reference (three tiers)
and displacement frames within the float floor.

Consumes the frozen json_v1 layer, the conformance checker, and the kinematics
engine; modifies none of them.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from phonon_core import __version__ as _PC_VERSION
from phonon_core import conformance
from phonon_core.expand import expand_cell
from phonon_core.kinematics import (
    canonicalize_mode, compute_bloch_phases, compute_displacements_at_phase,
    compute_trajectory,
)
from phonon_core.io.json_v1 import (
    PhononDocument, SCHEMA_VERSION, write_json_v1, read_json_v1,
)

# Fixed kinematics inputs for the bundled golden vectors — identical to the
# 0.2.0 golden fixture so the formats are interchangeable.
AMPLITUDE = 1.5
PHI = 0.7
SUPERCELL = (2, 2, 1)
N_FRAMES = 16
MASS_WEIGHTED = True

_CONVENTIONS = {
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

_BUNDLE_VERSION = 1


@dataclass
class _CuratedMode:
    name: str
    block: str
    q_index: int
    band_index: int


def _block_of(document: PhononDocument, name: str):
    return document.band if name == "band" else document.mesh


def curate_modes(document: PhononDocument) -> list[_CuratedMode]:
    """A few documented modes spanning Γ and finite q (band block preferred)."""
    name = "band" if (document.band is not None
                      and document.band.eigenvectors is not None) else "mesh"
    blk = _block_of(document, name)
    if blk is None or blk.eigenvectors is None:
        raise ValueError("document has no eigenvector-bearing block to bundle")
    q = np.asarray(blk.q_points, dtype=np.float64)
    qn = np.linalg.norm(q, axis=1)
    n_b = blk.frequencies.shape[1]

    gamma = int(np.flatnonzero(qn < 1e-8)[0]) if (qn < 1e-8).any() else int(np.argmin(qn))
    out = [
        _CuratedMode("gamma_acoustic", name, gamma, min(2, n_b - 1)),
        _CuratedMode("gamma_optical", name, gamma, n_b - 1),
    ]
    finite_idx: list[int] = []
    if hasattr(blk, "segment_nqpoint"):
        ends = (np.cumsum(blk.segment_nqpoint) - 1).tolist()
        finite_idx = [int(e) for e in ends if qn[int(e)] > 1e-6]
    fin = np.flatnonzero(qn > 1e-6)
    if fin.size:
        finite_idx.append(int(fin[len(fin) // 2]))   # generic interior
    labels = ["m_point", "k_point", "generic", "finite_3", "finite_4"]
    seen: set[int] = set()
    li = 0
    for qi in finite_idx:
        if qi in seen:
            continue
        seen.add(qi)
        out.append(_CuratedMode(labels[min(li, len(labels) - 1)], name, qi,
                                min(3 + li, n_b - 1)))
        li += 1
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_web_bundle(document: PhononDocument, out_dir: str, *,
                     indent: Optional[int] = None) -> dict:
    """Write {document.json, golden_kinematics.{npz,json}, bundle.json} into
    ``out_dir``. Returns the manifest dict."""
    os.makedirs(out_dir, exist_ok=True)
    doc_path = os.path.join(out_dir, "document.json")
    write_json_v1(document, doc_path, indent=indent)

    mat = document.material
    ea = expand_cell(mat, *SUPERCELL, mode="strict")
    curated = curate_modes(document)

    npz = {
        "_material_lattice": mat.lattice,
        "_material_reciprocal": mat.reciprocal_lattice,
        "_material_masses": mat.masses,
        "_material_fractional": mat.fractional_coords,
        "_slot_j": ea.j.astype(np.int64),
        "_slot_n_int": ea.n_int.astype(np.int64),
    }
    sidecar_modes = {}
    for cm in curated:
        blk = _block_of(document, cm.block)
        mode = blk.get_mode(cm.q_index, cm.band_index)
        canon = canonicalize_mode(mode)
        bloch = compute_bloch_phases(canon.q_point, ea.n_int)
        disp_phi = compute_displacements_at_phase(
            canon, mat, AMPLITUDE, PHI, atom_indices=ea.j,
            bloch_phases=bloch, mass_weighted=MASS_WEIGHTED)
        disp_traj = compute_trajectory(
            canon, mat, AMPLITUDE, N_FRAMES, atom_indices=ea.j,
            bloch_phases=bloch, mass_weighted=MASS_WEIGHTED)
        npz[f"{cm.name}__eigvec_raw"] = mode.eigenvector.astype(np.complex128)
        npz[f"{cm.name}__eigvec_canon"] = canon.eigenvector.astype(np.complex128)
        npz[f"{cm.name}__bloch"] = bloch.astype(np.float64)
        npz[f"{cm.name}__disp_phi"] = disp_phi.astype(np.float64)
        npz[f"{cm.name}__disp_traj"] = disp_traj.astype(np.float64)
        sidecar_modes[cm.name] = {
            "block": cm.block,
            "q_index": int(cm.q_index),
            "band_index": int(cm.band_index),
            "q_point": [float(x) for x in mode.q_point],
            "frequency_THz": float(mode.frequency),
            "is_imaginary": bool(mode.is_imaginary),
            "is_finite_q": bool(np.linalg.norm(mode.q_point) > 1e-8),
            "n_atoms": int(mat.num_atoms),
            "n_slots": int(len(ea)),
        }

    npz_path = os.path.join(out_dir, "golden_kinematics.npz")
    json_path = os.path.join(out_dir, "golden_kinematics.json")
    np.savez(npz_path, **npz)
    golden_sidecar = {
        "fixture_version": 1,
        "phonon_core_version": _PC_VERSION,
        "source_file": "document.json",
        "material_symbols": list(mat.symbols),
        "inputs": {"amplitude": AMPLITUDE, "phi": PHI, "supercell": list(SUPERCELL),
                   "n_frames": N_FRAMES, "mass_weighted": MASS_WEIGHTED},
        "conventions": _CONVENTIONS,
        "modes": sidecar_modes,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(golden_sidecar, fh, indent=2)

    manifest = {
        "bundle_version": _BUNDLE_VERSION,
        "phonon_core_version": _PC_VERSION,
        "schema_version": SCHEMA_VERSION,
        "document": "document.json",
        "document_sha256": _sha256(doc_path),
        "golden_vectors": "golden_kinematics.npz",
        "golden_sidecar": "golden_kinematics.json",
        "golden_npz_sha256": _sha256(npz_path),
        "n_bundled_modes": len(curated),
        "bundled_modes": list(sidecar_modes),
        "has_finite_q": any(m["is_finite_q"] for m in sidecar_modes.values()),
    }
    with open(os.path.join(out_dir, "bundle.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


@dataclass
class BundleCheck:
    conformance: conformance.ConformanceResult
    max_disp_dev: float          # displacement reproduction vs golden frames
    n_modes: int

    @property
    def passed(self) -> bool:
        return self.conformance.passed and self.max_disp_dev < 1e-9


def verify_web_bundle(out_dir: str, *, atol: float = 1e-8,
                      rtol: float = 1e-5) -> BundleCheck:
    """The cross-language contract, run in Python: load the bundle's document,
    recompute displacements via phonon_core kinematics, and check against the
    bundled golden vectors — eigenvectors through the three-tier gate, and
    displacement frames at the float floor."""
    with open(os.path.join(out_dir, "bundle.json")) as fh:
        manifest = json.load(fh)
    document = read_json_v1(os.path.join(out_dir, manifest["document"]))
    npz = np.load(os.path.join(out_dir, manifest["golden_vectors"]))
    with open(os.path.join(out_dir, manifest["golden_sidecar"])) as fh:
        side = json.load(fh)

    mat = document.material
    j = npz["_slot_j"]
    n_int = npz["_slot_n_int"]
    amp = side["inputs"]["amplitude"]
    phi = side["inputs"]["phi"]
    nfr = side["inputs"]["n_frames"]
    mw = side["inputs"]["mass_weighted"]

    names = list(side["modes"])
    cand_e, ref_e, qpts, cand_f, ref_f = [], [], [], [], []
    max_disp = 0.0
    for nm in names:
        meta = side["modes"][nm]
        blk = document.band if meta["block"] == "band" else document.mesh
        mode = blk.get_mode(meta["q_index"], meta["band_index"])
        # eigenvector conformance: document vs bundled golden raw eigenvector
        cand_e.append(mode.eigenvector)
        ref_e.append(npz[f"{nm}__eigvec_raw"])
        qpts.append(mode.q_point)
        cand_f.append(mode.frequency)
        ref_f.append(meta["frequency_THz"])
        # displacement reproduction: recompute from the document's eigenvector
        canon = canonicalize_mode(mode)
        bloch = compute_bloch_phases(canon.q_point, n_int)
        dphi = compute_displacements_at_phase(canon, mat, amp, phi,
                                              atom_indices=j, bloch_phases=bloch,
                                              mass_weighted=mw)
        dtraj = compute_trajectory(canon, mat, amp, nfr, atom_indices=j,
                                   bloch_phases=bloch, mass_weighted=mw)
        max_disp = max(max_disp,
                       float(np.max(np.abs(dphi - npz[f"{nm}__disp_phi"]))),
                       float(np.max(np.abs(dtraj - npz[f"{nm}__disp_traj"]))))

    res = conformance.check_against_reference(
        np.array(cand_e), np.array(ref_e), mat.masses, np.array(qpts),
        atol=atol, rtol=rtol,
        candidate_freqs=np.array(cand_f), reference_freqs=np.array(ref_f))
    return BundleCheck(conformance=res, max_disp_dev=max_disp, n_modes=len(names))
