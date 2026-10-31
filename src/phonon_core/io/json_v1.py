"""phonon-core JSON v1 exchange format — base layer (Phase A).

One JSON v1 document = one :class:`Material` + optional :class:`BandPath` +
optional :class:`MeshGrid` + metadata. This module reads and writes that
document with the stdlib ``json`` only (no heavy runtime deps), and provides a
one-way exporter to phononwebsite's JSON schema.

The wire format is specified in ``docs/json_v1_schema.md`` (v1.0 core LOCKED;
1.1.0 adds the ADDITIVE ``extensions.engine`` block — see the appendix) and
validated by ``docs/json_v1_schema.json``. Top-level shape::

    {
      "schema_version": "1.1.0",
      "generator":   {"name": "phonon-core", "version": "x.y.z"},
      "conventions": { ... 8 mandatory keys ... },
      "metadata":    { ... free-form, optional ... },
      "core": {
        "material": {lattice, symbols, masses, fractional_coords,
                     positions_were_wrapped},
        "band":  {storage, q_points, distances, segment_nqpoint, labels,
                  frequencies, eigenvectors_real, eigenvectors_imag} | null,
        "mesh":  {storage, mesh, q_points, weights, frequencies,
                  eigenvectors_real, eigenvectors_imag} | null
      },
      "extensions": {}        // or {"engine": {...}} — see EngineProvenance
    }

Conventions (Maradudin & Vosko 1968; Togo & Tanaka 2015): frequencies in THz,
eigenvectors mass-weighted (Σ|e_s|² = 1), q-points fractional, lattice rows are
a1,a2,a3. Eigenvectors are stored as a real/imag split (JSON has no complex),
shaped [n_q][n_branch][n_atom][3] — Float32Array/three.js-friendly.

Losslessness: floats are serialized via Python's round-trippable ``repr`` (the
stdlib json default), so eigenvector real/imag arrays survive write→read
**bitwise**. ``reciprocal_lattice`` is NOT a base-layer field (decision 7); it
is recomputed from ``lattice`` on read (B = inv(A)^T, phonopy no-2π convention)
and so will differ from a phonopy-original by ~1e-9 — expected, since it is
derivable and out of the v1 schema.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from phonon_core.types import Material, BandPath, MeshGrid
from phonon_core.units import THZ_TO_CM1
from phonon_core import __version__ as _PC_VERSION

SCHEMA_VERSION = "1.1.0"

# Mandatory conventions block (decision 3). Values fixed by phonon-core's data
# model except source_code/source_version (provenance, caller-supplied).
_REQUIRED_CONVENTION_KEYS = (
    "frequency_units",
    "eigenvector_normalization",
    "q_point_format",
    "lattice_vectors",
    "dynamical_matrix_fourier",
    "nac_method",
    "source_code",
    "source_version",
)


# --------------------------------------------------------------------------
# document container
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EngineProvenance:
    """``extensions.engine`` (schema 1.1.0, ADDITIVE) — evaluation provenance.

    This block is Level-2's SINGLE-SOURCE declaration: ``core.band`` and
    ``core.mesh`` marked ``evaluated=true`` were COMPUTED by the named engine
    from the force constants identified by ``fc_md5`` — not parsed from two
    unrelated phonopy output files — so band-within-mesh exactness is a
    provenance-backed claim, not an assumption. The arc: Stage 1 shipped the
    A1 engine itself (``engine/dq_phonopy.py``, 0.6.78, kind "phonopy-A1";
    A2's numpy engine will declare "numpy-A2"); THIS bump (0.6.79) defines the
    wire format; Bump 2 (0.6.80) makes the exporter emit it.

    Wire form (written iff a document carries provenance; otherwise
    ``extensions`` stays ``{}`` and every existing 1.0.0 file reads back with
    ``engine=None``)::

        "extensions": {"engine": {
            "kind": "phonopy-A1",
            "evaluated": {"band": true, "mesh": true},
            "phonopy_version": "4.3.1" | null,
            "fc_md5": "<32 lowercase hex>" | null,
            "phonopy_yaml_md5": "<32 lowercase hex>" | null }}

    Reader contract: strict on the required keys/types above; unknown extra
    keys INSIDE ``engine`` are IGNORED (semver minor-compat — a 1.2.0 file
    must not break this reader); other ``extensions`` keys keep the untouched
    v1.0 contract (ignored).
    """

    kind: str                          # e.g. "phonopy-A1" (A2 later: "numpy-A2")
    evaluated_band: bool
    evaluated_mesh: bool
    phonopy_version: Optional[str] = None
    fc_md5: Optional[str] = None       # md5 of FORCE_CONSTANTS
    phonopy_yaml_md5: Optional[str] = None


@dataclass
class PhononDocument:
    """A JSON v1 document: one material, optional band and/or mesh, metadata.

    ``band.material`` / ``mesh.material`` are the same object as ``material``;
    the material is serialized once.
    """

    material: Material
    band: Optional[BandPath] = None
    mesh: Optional[MeshGrid] = None
    metadata: dict = field(default_factory=dict)
    # Provenance read back from / written to the conventions block:
    source_code: str = "phonopy"
    source_version: Optional[str] = None
    # Evaluation provenance (extensions.engine, schema 1.1.0); None on every
    # 1.0.0 file and whenever the document was parsed, not engine-evaluated.
    engine: Optional[EngineProvenance] = None

    @classmethod
    def from_bandpath(cls, bp: BandPath, **kw) -> "PhononDocument":
        return cls(material=bp.material, band=bp, **kw)

    @classmethod
    def from_meshgrid(cls, mg: MeshGrid, **kw) -> "PhononDocument":
        return cls(material=mg.material, mesh=mg, **kw)


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------
def _material_block(mat: Material) -> dict:
    return {
        "lattice": mat.lattice.tolist(),
        "symbols": list(mat.symbols),
        "masses": mat.masses.tolist(),
        "fractional_coords": mat.fractional_coords.tolist(),
        "positions_were_wrapped": bool(getattr(mat, "positions_were_wrapped", False)),
    }


def _split_eigvecs(ev: Optional[np.ndarray]):
    """(n_q,n_branch,n_atom,3) complex -> (real_nested, imag_nested) or (None,None)."""
    if ev is None:
        return None, None
    ev = np.asarray(ev, dtype=np.complex128)
    return ev.real.tolist(), ev.imag.tolist()


def _band_block(bp: BandPath) -> dict:
    real, imag = _split_eigvecs(bp.eigenvectors)
    return {
        "storage": "inline",
        "q_points": bp.q_points.tolist(),
        "distances": bp.distances.tolist(),
        "segment_nqpoint": [int(n) for n in bp.segment_nqpoint],
        "labels": [[a, b] for (a, b) in bp.labels],
        "frequencies": bp.frequencies.tolist(),          # THz
        "eigenvectors_real": real,
        "eigenvectors_imag": imag,
    }


def _mesh_block(mg: MeshGrid) -> dict:
    real, imag = _split_eigvecs(mg.eigenvectors)
    return {
        "storage": "inline",
        "mesh": [int(x) for x in mg.mesh],
        "q_points": mg.q_points.tolist(),
        "weights": (None if mg.weights is None
                    else [int(w) for w in np.asarray(mg.weights).tolist()]),
        "frequencies": mg.frequencies.tolist(),          # THz
        "eigenvectors_real": real,
        "eigenvectors_imag": imag,
    }


def _conventions_block(doc: PhononDocument) -> dict:
    return {
        "frequency_units": "THz",
        "eigenvector_normalization": "mass_weighted",   # Togo & Tanaka 2015
        "q_point_format": "fractional",
        "lattice_vectors": "rows",                        # phonopy
        "dynamical_matrix_fourier": None,                 # reserved
        "nac_method": "none",                             # reserved
        "source_code": doc.source_code,
        "source_version": doc.source_version,
    }


def _engine_block(e: EngineProvenance) -> dict:
    """EngineProvenance -> the extensions.engine wire object (1.1.0)."""
    return {
        "kind": e.kind,
        "evaluated": {"band": bool(e.evaluated_band),
                      "mesh": bool(e.evaluated_mesh)},
        "phonopy_version": e.phonopy_version,
        "fc_md5": e.fc_md5,
        "phonopy_yaml_md5": e.phonopy_yaml_md5,
    }


def build_document_dict(doc: PhononDocument) -> dict:
    """Build the JSON-serializable dict for a document (no file I/O)."""
    core: dict = {"material": _material_block(doc.material)}
    core["band"] = _band_block(doc.band) if doc.band is not None else None
    core["mesh"] = _mesh_block(doc.mesh) if doc.mesh is not None else None
    # extensions.engine written iff the document carries provenance; a
    # provenance-free document keeps the 1.0.0 shape (extensions == {}).
    extensions: dict = {}
    if doc.engine is not None:
        extensions["engine"] = _engine_block(doc.engine)
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": {"name": "phonon-core", "version": _PC_VERSION},
        "conventions": _conventions_block(doc),
        "metadata": dict(doc.metadata),
        "core": core,
        "extensions": extensions,
    }


def write_json_v1(document: PhononDocument, path: str, *, indent: int | None = None) -> None:
    """Write ``document`` to ``path`` as JSON v1.

    ``indent=None`` (default) is the most compact and the cheapest to parse;
    pass an int for a human-readable file. Float serialization uses the stdlib
    json default (round-trippable repr) so eigenvectors round-trip bitwise.
    """
    obj = build_document_dict(document)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------
class JsonV1Error(ValueError):
    """Raised when a JSON v1 file is malformed or incompatible."""


def _check_schema_version(doc: dict) -> None:
    sv = doc.get("schema_version")
    if sv is None:
        raise JsonV1Error("missing required top-level 'schema_version'")
    if not isinstance(sv, str):
        raise JsonV1Error(f"'schema_version' must be a string; got {type(sv).__name__}")
    parts = sv.split(".")
    try:
        major = int(parts[0])
    except (ValueError, IndexError):
        raise JsonV1Error(f"malformed semver 'schema_version': {sv!r}")
    # Forward-compatible: accept any unknown MINOR/PATCH within MAJOR 1; reject
    # an unknown MAJOR (decision 2).
    if major != 1:
        raise JsonV1Error(
            f"unsupported schema major version {major} (file schema_version "
            f"{sv!r}); this reader supports 1.x. Refusing to guess at a "
            f"breaking format change."
        )


def _check_conventions(doc: dict) -> dict:
    conv = doc.get("conventions")
    if conv is None:
        raise JsonV1Error(
            "missing required 'conventions' block — a JSON v1 file must declare "
            "its units and conventions explicitly (frequency_units, "
            "eigenvector_normalization, q_point_format, lattice_vectors, "
            "dynamical_matrix_fourier, nac_method, source_code, source_version)."
        )
    if not isinstance(conv, dict):
        raise JsonV1Error("'conventions' must be an object")
    missing = [k for k in _REQUIRED_CONVENTION_KEYS if k not in conv]
    if missing:
        raise JsonV1Error(f"'conventions' block missing required keys: {missing}")
    return conv


def _eigvecs_from_split(real, imag):
    if real is None and imag is None:
        return None
    if real is None or imag is None:
        raise JsonV1Error(
            "eigenvectors must provide BOTH eigenvectors_real and "
            "eigenvectors_imag, or neither"
        )
    re = np.asarray(real, dtype=np.float64)
    im = np.asarray(imag, dtype=np.float64)
    if re.shape != im.shape:
        raise JsonV1Error(
            f"eigenvector real/imag shape mismatch: {re.shape} vs {im.shape}"
        )
    return re + 1j * im


def _check_storage(block: dict, which: str) -> None:
    storage = block.get("storage", "inline")
    if storage == "sidecar":
        raise JsonV1Error(
            f"{which}.storage='sidecar' is reserved but not implemented in v1; "
            f"only 'inline' is supported"
        )
    if storage != "inline":
        raise JsonV1Error(f"{which}.storage={storage!r} unknown; expected 'inline'")


_MD5_HEX = re.compile(r"^[0-9a-f]{32}$")   # exactly 32 lowercase hex


def _engine_from_extensions(doc: dict) -> Optional[EngineProvenance]:
    """Parse extensions.engine (schema 1.1.0) -> EngineProvenance or None.

    Absent extensions, or absent "engine" key -> None (every existing 1.0.0
    file; the v1.0 ignore-unknown-keys contract for OTHER extensions keys is
    untouched). A PRESENT engine block is validated STRICTLY on its required
    keys/types — but unknown extra keys INSIDE it are ignored, so a future
    1.2.0 file that grows the block does not break this reader (semver
    minor-compat). Malformed -> JsonV1Error naming the offending key.
    """
    ext = doc.get("extensions")
    if not isinstance(ext, dict):        # absent (or non-object: v1.0 readers
        return None                      # never looked inside — stay lenient)
    eng = ext.get("engine")
    if eng is None:
        return None
    if not isinstance(eng, dict):
        raise JsonV1Error("'extensions.engine' must be an object")
    kind = eng.get("kind")
    if not isinstance(kind, str) or not kind:
        raise JsonV1Error("'extensions.engine.kind' must be a non-empty string")
    ev = eng.get("evaluated")
    if not isinstance(ev, dict):
        raise JsonV1Error("'extensions.engine.evaluated' must be an object "
                          "with boolean 'band' and 'mesh'")
    flags = {}
    for k in ("band", "mesh"):
        v = ev.get(k)
        if not isinstance(v, bool):
            raise JsonV1Error(f"'extensions.engine.evaluated.{k}' must be a boolean")
        flags[k] = v
    pv = eng.get("phonopy_version")
    if pv is not None and not isinstance(pv, str):
        raise JsonV1Error("'extensions.engine.phonopy_version' must be a "
                          "string or null")
    md5s = {}
    for k in ("fc_md5", "phonopy_yaml_md5"):
        v = eng.get(k)
        if v is not None and not (isinstance(v, str) and _MD5_HEX.match(v)):
            raise JsonV1Error(f"'extensions.engine.{k}' must be null or "
                              f"exactly 32 lowercase hex characters")
        md5s[k] = v
    return EngineProvenance(
        kind=kind,
        evaluated_band=flags["band"],
        evaluated_mesh=flags["mesh"],
        phonopy_version=pv,
        fc_md5=md5s["fc_md5"],
        phonopy_yaml_md5=md5s["phonopy_yaml_md5"],
    )


def _material_from_block(mb: dict) -> Material:
    for key in ("lattice", "symbols", "masses", "fractional_coords"):
        if key not in mb:
            raise JsonV1Error(f"core.material missing required field {key!r}")
    lattice = np.asarray(mb["lattice"], dtype=np.float64)
    # reciprocal_lattice is NOT stored (decision 7); recompute B = inv(A)^T
    # (phonopy reciprocal without 2π).
    reciprocal = np.linalg.inv(lattice).T
    mat = Material(
        lattice=lattice,
        reciprocal_lattice=reciprocal,
        symbols=list(mb["symbols"]),
        masses=np.asarray(mb["masses"], dtype=np.float64),
        fractional_coords=np.asarray(mb["fractional_coords"], dtype=np.float64),
    )
    # Restore the stored provenance flag (Material.__post_init__ recomputes it
    # as False for already-wrapped coords; the file's value is authoritative).
    if "positions_were_wrapped" in mb:
        mat.positions_were_wrapped = bool(mb["positions_were_wrapped"])
    return mat


def read_json_v1(path: str) -> PhononDocument:
    """Read a JSON v1 file into a :class:`PhononDocument`.

    Raises :class:`JsonV1Error` on a missing/incompatible schema version, a
    missing/incomplete conventions block, a reserved 'sidecar' storage value,
    or a missing required core field.
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise JsonV1Error("top-level JSON value must be an object")

    _check_schema_version(doc)
    conv = _check_conventions(doc)

    core = doc.get("core")
    if not isinstance(core, dict) or "material" not in core:
        raise JsonV1Error("missing 'core.material'")
    material = _material_from_block(core["material"])

    band = None
    if core.get("band") is not None:
        b = core["band"]
        _check_storage(b, "core.band")
        band = BandPath(
            material=material,
            q_points=np.asarray(b["q_points"], dtype=np.float64),
            distances=np.asarray(b["distances"], dtype=np.float64),
            frequencies=np.asarray(b["frequencies"], dtype=np.float64),
            eigenvectors=_eigvecs_from_split(
                b.get("eigenvectors_real"), b.get("eigenvectors_imag")),
            segment_nqpoint=[int(n) for n in b["segment_nqpoint"]],
            labels=[(p[0], p[1]) for p in b["labels"]],
        )

    mesh = None
    if core.get("mesh") is not None:
        m = core["mesh"]
        _check_storage(m, "core.mesh")
        mesh = MeshGrid(
            material=material,
            mesh=tuple(int(x) for x in m["mesh"]),
            q_points=np.asarray(m["q_points"], dtype=np.float64),
            frequencies=np.asarray(m["frequencies"], dtype=np.float64),
            eigenvectors=_eigvecs_from_split(
                m.get("eigenvectors_real"), m.get("eigenvectors_imag")),
            weights=(None if m.get("weights") is None
                     else np.asarray(m["weights"], dtype=np.int64)),
        )

    return PhononDocument(
        material=material, band=band, mesh=mesh,
        metadata=dict(doc.get("metadata") or {}),
        source_code=conv.get("source_code", "phonopy"),
        source_version=conv.get("source_version"),
        engine=_engine_from_extensions(doc),
    )


# --------------------------------------------------------------------------
# phononwebsite exporter (one-way; their schema)
# --------------------------------------------------------------------------
# Standard element symbol -> atomic number Z (1..118), for phononwebsite's
# "atom_numbers". Reference data (IUPAC periodic table); not a v1 schema field.
_SYMBOL_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16,
    "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23,
    "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37,
    "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44,
    "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51,
    "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58,
    "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65,
    "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72,
    "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79,
    "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
    "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93,
    "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106,
    "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112,
    "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118,
}


def _hill_formula(symbols: list[str]) -> str:
    """Compact count formula, e.g. ['W','Se','Se'] -> 'WSe2'. Display only."""
    counts: dict[str, int] = {}
    order: list[str] = []
    for s in symbols:
        if s not in counts:
            order.append(s)
            counts[s] = 0
        counts[s] += 1
    return "".join(s + (str(counts[s]) if counts[s] > 1 else "") for s in order)


def _highsym_and_breaks(bp: BandPath):
    """phononwebsite highsym_qpts (list of [q_index, label]) + line_breaks
    (list of [start, end] inclusive q-index ranges), from segment_nqpoint/labels.
    """
    highsym: list[list] = []
    line_breaks: list[list[int]] = []
    cum = 0
    for i, n in enumerate(bp.segment_nqpoint):
        start = cum
        end = cum + n - 1
        line_breaks.append([start, end])
        if i == 0:
            highsym.append([start, bp.labels[0][0]])
        highsym.append([end, bp.labels[i][1]])
        cum += n
    return highsym, line_breaks


def to_phononwebsite_json(material: Material, bandpath: BandPath, *,
                          name: Optional[str] = None,
                          repetitions=(3, 3, 3)) -> dict:
    """Export a (material, bandpath) to phononwebsite's INTERNAL JSON format.

    One-way export to the format consumed by
    https://henriquemiranda.github.io/phononwebsite/ . Emits exactly the
    internal-format field set (no ``@class``, none of the
    qpoints+frequencies+eigendisplacements+structure quartet), so
    phononjson.js::getFromJson routes it to getFromInternalJson rather than the
    pymatgen or Materials-Project-OpenData branches (phononjson.js lines
    681-690). ``line_breaks`` is kept (optional; read by
    phononjson.js::getLineBreaks, line 1063).

    Conventions on the wire (verified against phonon.py::write_json, lines
    219-232): lattice in Å (the "bohr" comment there is stale — the loader and
    real data/localdb/*/data.json treat it as Å); eigenvalues in cm⁻¹
    (= THz × 33.35641); qpoints fractional; vectors [Nq][Nb][Na][3][2] with the
    last axis [Re, Im]; highsym_qpts a list of [int q_index, str label] pairs.

    MASS CONVENTION — the important one. phononwebsite's JSON render path applies
    NO mass factor: vibcrystal.js displaces atoms directly by the (per-mode
    unit-normalized) eigenvector components, and phononwebpage.js's only mass
    term (getZeroPointAmplitudeAngstrom, lines 64-85) is guarded by
    mode_amplitude_convention === 'avg-mass-normalized' and affects only the
    overall scale. Its own writer (phonon.py) stores RAW mass-weighted e_s. By
    contrast our renderer displaces by e_s/√m_s (Togo & Tanaka 2015), and
    phononwebsite's YAML path pre-scales each atom by √(M_avg/m_s)
    (phononyaml.js lines 259-292) precisely so the no-mass-factor renderer shows
    the physical motion.

    So we pre-scale exported vectors by √(M_avg/m_s) per atom — identical to
    phononwebsite's own YAML path. Because phononjson.js::normalizeEigenvectors
    (lines 281-315) unit-normalizes each mode on load, the constant √(M_avg) is
    irrelevant; this is exactly ÷√m_s up to a per-mode constant, i.e. our
    physical displacement. This is an ADAPTER presentation transform: core
    eigenvectors and kinematics are untouched.
    """
    ev = np.asarray(bandpath.eigenvectors, dtype=np.complex128)
    masses = np.asarray(material.masses, dtype=np.float64)        # (Na,)
    m_avg = float(masses.mean())
    # per-atom scale √(M_avg/m_s); broadcast over (Nq,Nb,Na,3)
    scale = np.sqrt(m_avg / masses).reshape(1, 1, -1, 1)
    ev_scaled = ev * scale
    vectors = np.stack([ev_scaled.real, ev_scaled.imag], axis=-1)  # (Nq,Nb,Na,3,2)
    highsym, line_breaks = _highsym_and_breaks(bandpath)
    return {
        "name": name if name is not None else _hill_formula(material.symbols),
        "natoms": int(material.num_atoms),
        "lattice": material.lattice.tolist(),                      # Å
        "atom_types": list(material.symbols),
        "atom_numbers": [int(_SYMBOL_TO_Z.get(s, 0)) for s in material.symbols],
        "formula": _hill_formula(material.symbols),
        "repetitions": [int(r) for r in repetitions],
        "atom_pos_car": material.cartesian_coords.tolist(),        # Å
        "atom_pos_red": material.fractional_coords.tolist(),       # fractional
        "qpoints": bandpath.q_points.tolist(),                     # fractional
        "distances": bandpath.distances.tolist(),
        "eigenvalues": (bandpath.frequencies * THZ_TO_CM1).tolist(),  # cm⁻¹
        "vectors": vectors.tolist(),                               # [Nq][Nb][Na][3][2]
        "highsym_qpts": highsym,                                   # [[int,label],...]
        "line_breaks": line_breaks,
        # Declare the convention phononwebsite's own YAML loader sets after the
        # √(M_avg/m_s) scaling (phononyaml.js lines 291-292). With our vectors
        # already pre-scaled, this names the convention explicitly and unlocks
        # phononwebpage.js::getZeroPointAmplitudeAngstrom (lines 64-85, which
        # reads mode_amplitude_convention + average_mass + eigenvalues for a
        # physical display scale). SAFE / no double-scaling: the per-atom
        # re-scaler applyModeAmplitudeConventionToEigenvectors (phononjson.js
        # line 465) additionally requires dynamical_matrix.masses (we emit none)
        # and is only ever called on the runtime-compute path
        # (computeAllRuntimeEigenvectorsWithBandConnectionAsync, line 199), never
        # on stored vectors loaded via getFromInternalJson — and we ship no
        # dynamical_matrix, so canComputeEigenvectorsOnDemand() is false.
        "mode_amplitude_convention": "avg-mass-normalized",
        "average_mass": m_avg,                                     # AMU, mean(masses)
    }


# --------------------------------------------------------------------------
# phononwebsite importer (reads their files into our dataclasses)
# --------------------------------------------------------------------------
# Atomic mass (AMU) indexed by Z. Copied verbatim from phononwebsite's
# src/atomic_data.js (`atomic_mass`), so masses used here match exactly the
# values phononwebsite itself uses to mass-scale its YAML path
# (phononyaml.js). Reference data, not a v1 schema field.
_ATOMIC_MASS_AMU = {
    1: 1.00794, 2: 4.002602, 3: 6.941, 4: 9.012182, 5: 10.811, 6: 12.0107,
    7: 14.0067, 8: 15.9994, 9: 18.9984032, 10: 20.1797, 11: 22.98976928,
    12: 24.305, 13: 26.9815386, 14: 28.0855, 15: 30.973762, 16: 32.065,
    17: 35.453, 18: 39.948, 19: 39.0983, 20: 40.078, 21: 44.955912,
    22: 47.867, 23: 50.9415, 24: 51.9961, 25: 54.938045, 26: 55.845,
    27: 58.933195, 28: 58.6934, 29: 63.546, 30: 65.38, 31: 69.723, 32: 72.64,
    33: 74.9216, 34: 78.96, 35: 79.904, 36: 83.798, 37: 85.4678, 38: 87.62,
    39: 88.90585, 40: 91.224, 41: 92.90638, 42: 95.96, 44: 101.07,
    45: 102.9055, 46: 106.42, 47: 107.8682, 48: 112.411, 49: 114.818,
    50: 118.71, 51: 121.76, 52: 127.6, 53: 126.90447, 54: 131.293,
    55: 132.9054519, 56: 137.327, 57: 138.90547, 58: 140.116, 59: 140.90765,
    60: 144.242, 62: 150.36, 63: 151.964, 64: 157.25, 65: 158.92535,
    66: 162.5, 67: 164.93032, 68: 167.259, 69: 168.93421, 70: 173.054,
    71: 174.9668, 72: 178.49, 73: 180.94788, 74: 183.84, 75: 186.207,
    76: 190.23, 77: 192.217, 78: 195.084, 79: 196.966569, 80: 200.59,
    81: 204.3833, 82: 207.2, 83: 208.9804, 90: 232.03806, 91: 231.03588,
    92: 238.02891,
}


def _detect_phononwebsite_format(data: dict) -> str:
    """Mirror phononjson.js::getFromJson routing (lines 681-690)."""
    if "@class" in data:
        return "pymatgen"
    if all(k in data for k in ("qpoints", "frequencies",
                               "eigendisplacements", "structure")):
        return "opendata"
    return "internal"


def _masses_for(atom_numbers, atom_types) -> np.ndarray:
    """Per-atom mass (AMU) from atom_numbers (preferred) or atom_types→Z."""
    if atom_numbers is not None:
        zs = [int(z) for z in atom_numbers]
    else:
        zs = [int(_SYMBOL_TO_Z.get(str(s), 0)) for s in atom_types]
    out = []
    for z, sym in zip(zs, atom_types if atom_types is not None else zs):
        m = _ATOMIC_MASS_AMU.get(z)
        if m is None:
            raise JsonV1Error(
                f"no atomic mass for Z={z} (symbol {sym!r}); cannot reconstruct "
                f"mass-weighted eigenvectors"
            )
        out.append(m)
    return np.asarray(out, dtype=np.float64)


def _segments_from_highsym(highsym, nq: int):
    """phononwebsite highsym_qpts ([[q_index,label],...]) -> our
    (segment_nqpoint, labels). phononwebsite labels points on a continuous
    path; our BandPath splits into per-segment label pairs. We use consecutive
    high-symmetry indices as half-open segment boundaries with the final
    segment taking the remainder, so Σ segment_nqpoint == nq. Tick grouping is
    therefore approximate at shared boundaries; q-points/frequencies/
    eigenvectors are exact. (This importer targets visualization, not a
    byte-exact path round-trip.)
    """
    if not highsym or len(highsym) < 2:
        lbl = str(highsym[0][1]) if highsym else ""
        return [nq], [(lbl, lbl)]
    bounds = [int(h[0]) for h in highsym]
    labs = [str(h[1]) for h in highsym]
    seg_n, seg_labels = [], []
    m = len(bounds)
    for k in range(m - 1):
        n = (bounds[k + 1] - bounds[k]) if k < m - 2 else (nq - bounds[k])
        seg_n.append(int(n))
        seg_labels.append((labs[k], labs[k + 1]))
    return seg_n, seg_labels


def from_phononwebsite_json(source) -> PhononDocument:
    """Read a phononwebsite file into our dataclasses (Material + BandPath).

    ``source`` is a path (str) or an already-parsed dict. Supports the
    phononwebsite INTERNAL format (phonon.py::write_json / data/localdb/*/
    data.json). The pymatgen (``@class``) and Materials-Project OpenData
    (qpoints+frequencies+eigendisplacements+structure) formats are detected and
    raise NotImplementedError — flagged as a follow-up — rather than being
    mis-parsed (phononjson.js supports all three; we cover internal here).

    Conversions: eigenvalues cm⁻¹ → THz (÷33.35641); vectors [Nq][Nb][Na][3][2]
    [Re,Im] → complex (Nq,Nb,Na,3); qpoints stay fractional. Masses are looked
    up from atom_numbers via phononwebsite's own atomic_mass table
    (atomic_data.js).

    IMPORTANT — convention asymmetry. Internal-format vectors are RAW
    mass-weighted e_s (phonopy convention; phonon.py stores e_s directly), so
    they are stored AS-IS as our native mass-weighted eigenvectors and our
    renderer applies the 1/√m_s itself. Do NOT feed this importer the output of
    ``to_phononwebsite_json`` — that export is pre-scaled by √(M_avg/m_s) for
    phononwebsite's no-mass-factor renderer, and re-importing it would
    double-apply the mass factor. Import targets phononwebsite-NATIVE files.
    """
    if isinstance(source, str):
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(source, dict):
        data = source
    else:
        raise TypeError("source must be a file path (str) or a parsed dict")

    fmt = _detect_phononwebsite_format(data)
    if fmt != "internal":
        raise NotImplementedError(
            f"phononwebsite '{fmt}' format import is not implemented in 0.1.6 "
            f"(internal format only); follow-up. Detected via the same routing "
            f"as phononjson.js::getFromJson."
        )

    for key in ("lattice", "atom_types", "atom_pos_red", "qpoints",
                "eigenvalues", "vectors"):
        if key not in data:
            raise JsonV1Error(f"phononwebsite internal file missing {key!r}")

    lattice = np.asarray(data["lattice"], dtype=np.float64)
    symbols = [str(s) for s in data["atom_types"]]
    masses = _masses_for(data.get("atom_numbers"), symbols)
    frac = np.asarray(data["atom_pos_red"], dtype=np.float64)
    material = Material(
        lattice=lattice,
        reciprocal_lattice=np.linalg.inv(lattice).T,
        symbols=symbols,
        masses=masses,
        fractional_coords=frac,
    )

    q_points = np.asarray(data["qpoints"], dtype=np.float64)
    nq = len(q_points)
    distances = (np.asarray(data["distances"], dtype=np.float64)
                 if "distances" in data else np.arange(nq, dtype=np.float64))
    # eigenvalues cm⁻¹ -> THz
    frequencies = np.asarray(data["eigenvalues"], dtype=np.float64) / THZ_TO_CM1
    # vectors [Nq][Nb][Na][3][2] -> complex (Nq,Nb,Na,3); stored as RAW e_s
    v = np.asarray(data["vectors"], dtype=np.float64)
    eig = v[..., 0] + 1j * v[..., 1]
    seg_n, seg_labels = _segments_from_highsym(data.get("highsym_qpts"), nq)

    band = BandPath(
        material=material,
        q_points=q_points,
        distances=distances,
        frequencies=frequencies,
        eigenvectors=eig,
        segment_nqpoint=seg_n,
        labels=seg_labels,
    )
    meta = {}
    if "name" in data:
        meta["name"] = data["name"]
    return PhononDocument(material=material, band=band, metadata=meta)
