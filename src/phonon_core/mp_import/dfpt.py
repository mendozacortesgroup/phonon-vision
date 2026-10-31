"""MP dfpt phononwebsite document -> json_v1 band document (Track B, 0.6.90).

dfpt materials on MP ship BANDS + EIGENVECTORS but NO force constants
(Track A's manifest confirms: the dfpt FC accessor errors "No phonon force
constants data found"). So a dfpt import is a PRECOMPUTED PATH — the document
can be displayed and animated at exactly the q-points it carries, and nothing
else. Track B therefore converts, HONESTLY LABELED:

- the written document's ``extensions`` stay ``{}`` (``engine=None``): per the
  schema's own contract that field is absent "whenever the document was
  parsed, not engine-evaluated" — which is exactly what this is;
- ``metadata.arbitrary_q`` is ``False`` with a one-line reason, and the MP
  identifiers ride along (``metadata.mp``);
- ``conventions.source_code`` is the MP document's own calculator (abinit for
  legacy MP dfpt), not the default "phonopy".

FORMAT: MP's ``dfpt_phononwebsite.json`` is the phononwebsite INTERNAL format
— ``lattice / atom_types / atom_pos_red / qpoints / eigenvalues (cm^-1) /
vectors [Nq][Nb][Na][3][Re,Im] / highsym_qpts`` — the exact target of the
FROZEN reader ``phonon_core.io.json_v1.from_phononwebsite_json`` (e18613db,
byte-identical this version). Parsing is delegated to it (eigenvalues
cm^-1 -> THz via THZ_TO_CM1 = 33.35641), THEN two Track-B corrections apply:

CONVENTION FINDING (0.6.90, refining the 0.6.89 pin): the 0.6.89 "unit-norm
mass-weighted, no transform" pin holds for PHEASY documents — it is FALSE for
the dfpt PHONONWEBSITE export. MP website ``vectors`` are DISPLACEMENT-SCALED
by sqrt(M_avg/m_s) (the phononwebsite renderer convention — the same
pre-scaling our own ``to_phononwebsite_json`` applies, which the frozen
reader's docstring warns about). Measured: per-pure-species-mode norms equal
M_avg/m_s exactly — GaN 0.600/2.989 (Ga 41.86/69.72, N 41.86/14.007), GaAs
0.9653/1.0373; silicon is INVISIBLE to this (equal masses -> uniform scale,
norms stay 1), which is exactly why a monoatomic check could not have caught
it. This module UNDOES it — ``e_s = v_s * sqrt(m_s)``, renormalized per mode
— restoring native mass-weighted eigenvectors (post-correction norms
1 +/- 7e-16, per-q orthonormality at file precision <= 1.1e-06).

UNIT-CONSTANT NOTE (recorded, not corrected): website-vs-bandstructure
frequency agreement through the frozen constant is <= 3.1e-07 THz, purely
MULTIPLICATIVE (relative ~2e-8/THz) — a cm^-1 conversion-constant mismatch at
the 1e-8 level. The exact c-based constant (33.356409519815204) fits WORSE
(2x), so MP's own constant is attributable to neither; the residual is unit
noise, five decades inside any display tolerance.

NOTE the fused tick label: MP's ``highsym_qpts`` carries compound labels at
path discontinuities (silicon: ``'K|U'``) — plot furniture in the source data
itself, the same class the 0.8.4 ledger entry warns about. The frozen reader
keeps them verbatim in the segment label pairs; q/frequencies/eigenvectors
are exact regardless.

PURITY: json + os + numpy + phonon_core.io (frozen) — NO phonopy, NO
pymatgen, NO mp-api. Track B is a format transform; nothing is evaluated.
"""
import dataclasses
import json
import os

import numpy as np

from phonon_core.io.json_v1 import (
    JsonV1Error,
    from_phononwebsite_json,
    read_json_v1,
    write_json_v1,
)
from phonon_core.mp_import.pheasy import MPImportError

_WEBSITE = "dfpt_phononwebsite.json"
_METADATA = "dfpt_metadata.json"
_BAND = "dfpt_bandstructure.json"


def convert_dfpt(fixture_dir, out_path):
    """Convert a harvested MP dfpt fixture into a json_v1 band document.

    Parameters
    ----------
    fixture_dir : str
        Directory holding ``dfpt_phononwebsite.json`` (+ optionally
        ``dfpt_metadata.json`` and ``dfpt_bandstructure.json``, both consumed
        for provenance/preconditions when present).
    out_path : str
        File path for the written json_v1 document.

    Returns
    -------
    dict with ``"json_v1"`` (the path), ``"document"`` (the in-memory
    :class:`PhononDocument`), and ``"info"``.

    Preconditions (raise :class:`MPImportError` loudly, with the numbers):

    (i)    the phononwebsite file is the INTERNAL format the frozen reader
           accepts (delegated; failures re-raised with fixture context);
    (ii)   band-count identity ``n_bands == 3 * natoms`` on the website doc;
    (ii-b) when ``dfpt_bandstructure.json`` is present: its own band-count
           identity holds, and its q-point set matches the website doc's;
    (iii)  the written document round-trips through the frozen
           ``read_json_v1`` with band shapes intact.
    """
    fixture_dir = os.fspath(fixture_dir)
    out_path = os.fspath(out_path)
    site_path = os.path.join(fixture_dir, _WEBSITE)
    data = json.load(open(site_path))

    try:                                                    # (i)
        doc = from_phononwebsite_json(data)
    except (JsonV1Error, NotImplementedError) as exc:
        raise MPImportError(
            f"precondition (i): {site_path} did not parse as the "
            f"phononwebsite INTERNAL format: {exc}")

    nb = doc.band.frequencies.shape[1]
    na = len(doc.material.symbols)
    if nb != 3 * na:                                        # (ii)
        raise MPImportError(
            f"precondition (ii): band-count identity FAILED — website doc "
            f"has n_bands = {nb} but {na} atoms require 3 * {na} = {3 * na}")

    # ---- Track-B correction: undo the phononwebsite displacement scaling ----
    # (see module docstring: website vectors are sqrt(M_avg/m_s)-scaled;
    # multiply back by sqrt(m_s) and renormalize per mode -> native
    # mass-weighted eigenvectors, the convention every consumer expects)
    masses = np.asarray(doc.material.masses, dtype=np.float64)
    e = doc.band.eigenvectors * np.sqrt(masses)[None, None, :, None]
    nrm = np.sqrt((np.abs(e) ** 2).sum(axis=(2, 3), keepdims=True))
    if not np.all(nrm > 0):
        raise MPImportError("eigenvector normalization: a zero-norm mode "
                            "after mass correction — corrupt vectors block")
    doc.band = dataclasses.replace(doc.band, eigenvectors=e / nrm)
    norms = (np.abs(doc.band.eigenvectors) ** 2).sum(axis=(2, 3))
    if float(np.abs(norms - 1).max()) > 1e-9:
        raise MPImportError("eigenvector normalization failed post-correction")

    meta_path = os.path.join(fixture_dir, _METADATA)
    mp_meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}

    band_path = os.path.join(fixture_dir, _BAND)
    bs = json.load(open(band_path)) if os.path.isfile(band_path) else None
    if bs is not None:                                      # (ii-b)
        n_sites_bs = len(bs["structure"]["sites"])
        nb_bs = len(bs["frequencies"])
        if nb_bs != 3 * n_sites_bs:
            raise MPImportError(
                f"precondition (ii-b): band-count identity FAILED — the band "
                f"document stores n_bands = {nb_bs} but its own structure has "
                f"{n_sites_bs} sites, requiring 3 * {n_sites_bs} = "
                f"{3 * n_sites_bs}")
        q_bs = np.asarray(bs["qpoints"], dtype=np.float64)
        if q_bs.shape != doc.band.q_points.shape or not np.allclose(
                q_bs, doc.band.q_points, atol=1e-10):
            raise MPImportError(
                f"precondition (ii-b): the band document's q-points "
                f"({q_bs.shape}) do not match the website document's "
                f"({doc.band.q_points.shape}) — the two exports disagree "
                f"about the path")

    # ---- HONEST LABELING ----------------------------------------------------
    # engine stays None: extensions == {} — parsed, not engine-evaluated.
    doc.source_code = str(mp_meta.get("code") or "unknown")
    doc.metadata = dict(doc.metadata)
    doc.metadata["arbitrary_q"] = False
    doc.metadata["arbitrary_q_note"] = (
        "MP dfpt import: precomputed band path only — no force constants "
        "ship with dfpt documents, so D(q) cannot be evaluated at any other q")
    doc.metadata["eigenvector_source_convention"] = (
        "phononwebsite displacement-scaled sqrt(M_avg/m); converted to native "
        "mass-weighted (x sqrt(m), renormalized per mode) at import")
    doc.metadata["mp"] = {
        "phonon_method": mp_meta.get("phonon_method"),
        "identifier": mp_meta.get("identifier"),
        "code": mp_meta.get("code"),
        "has_nac": None if bs is None else bs.get("has_nac"),
    }

    write_json_v1(doc, out_path)

    reloaded = read_json_v1(out_path)                       # (iii)
    if (reloaded.band is None
            or reloaded.band.frequencies.shape != doc.band.frequencies.shape
            or reloaded.band.eigenvectors is None):
        raise MPImportError(
            "precondition (iii): the written json_v1 document did not "
            "round-trip with its band block intact")

    return {
        "json_v1": out_path,
        "document": doc,
        "info": {
            "n_qpoints": int(doc.band.frequencies.shape[0]),
            "n_bands": nb,
            "n_atoms": na,
            "identifier": mp_meta.get("identifier"),
            "code": mp_meta.get("code"),
            "has_nac": None if bs is None else bs.get("has_nac"),
        },
    }
