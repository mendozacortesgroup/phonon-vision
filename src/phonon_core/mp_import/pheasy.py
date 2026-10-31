"""MP pheasy document set -> phonopy file triple (Track A converter).

INPUT (a harvested fixture directory):

- ``pheasy_metadata.json``       — pymatgen ``Structure.as_dict()`` under
  ``structure``; ``supercell_matrix``; ``primitive_matrix``; ``born``;
  ``epsilon_static``; ``sum_rules_breaking``; ``phonon_method``; ``code``.
- ``pheasy_forceconstants.json`` — ``Phi[N_sc, N_sc, 3, 3]`` nested lists,
  on pheasy's supercell atom ordering.
- ``pheasy_bandstructure.json``  — (optional here, used for preconditions and
  ``band.conf``) emmet PhononBS: ``qpoints`` in DIRECT (fractional)
  coordinates of the PRIMITIVE basis the metadata's ``primitive_matrix``
  declares, ``frequencies`` in THz shaped ``(n_bands, n_qpoints)``,
  ``labels_dict``, ``kpath``.

OUTPUT (the phonopy triple + one sidecar, written into ``out_dir``):

- ``phonopy.yaml``      — via phonopy's own ``Phonopy.save`` (round-trips
  through ``phonopy.load``; asserted).
- ``FORCE_CONSTANTS``   — full ``(N_sc, N_sc)`` phonopy text format via
  ``phonopy.file_IO.write_FORCE_CONSTANTS``.
- ``band.conf``         — the SeeK/Setyawan-Curtarolo path from the band
  document's ``kpath`` + ``labels_dict`` (segment-per-label-pair spelling).
- ``mp_nac.json``       — Born charges + dielectric CARRIED, NOT APPLIED
  (``nac_applied: false``). The file is deliberately NOT named ``BORN``:
  phonopy/A1 auto-discovers a sibling literally named ``BORN`` and would
  silently switch the non-analytic correction ON (see CLAUDE.md, the BORN
  FILENAME TRAP).

CONVENTIONS (empirically pinned against the MP fixtures — see CLAUDE.md):

- ATOM ORDER: MP/pheasy's supercell atom ordering IS phonopy's own (the
  identity permutation). Proven by reproducing MP's own stored band
  frequencies from MP's own force constants at machine precision (silicon
  5.1e-11 THz over 1010 q); a single swapped pair moves that by 0.54 THz.
- ``primitive_matrix`` in the metadata is a det-1 UNIMODULAR basis change on
  every fixture; the band document's qpoints live in THAT primitive basis.
  Pass it through verbatim — never substitute ``'auto'`` (the phonopy-4
  basis trap, CLAUDE.md).
- ``epsilon_static`` actually holds the ION-CLAMPED high-frequency dielectric
  (eps_infinity), not the static constant its name suggests
  (``epsilon_electronic`` is null in every fixture).

PRECONDITIONS (raise :class:`MPImportError`, loudly, with the numbers):

(i)   ``primitive_matrix`` and ``supercell_matrix`` present and 3x3;
(ii)  ``N_sc == n_unitcell_atoms * round(|det(supercell_matrix)|)`` — the
      atom-count identity that gates the whole reconstruction;
(ii-b)``n_bands == 3 * n_sites`` on the band document when present — MP band
      documents are NOT guaranteed self-consistent (nacl_mp-1120767 stores 12
      bands for a 12-site structure; 36 are required);
(iii) the written ``phonopy.yaml`` + ``FORCE_CONSTANTS`` round-trip through
      ``phonopy.load`` without error.
"""
import json
import os

import numpy as np

_METADATA = "pheasy_metadata.json"
_FC = "pheasy_forceconstants.json"
_BAND = "pheasy_bandstructure.json"

#: Output basenames (fixed: the toolchain's discovery rules key on these).
OUT_PHONOPY_YAML = "phonopy.yaml"
OUT_FORCE_CONSTANTS = "FORCE_CONSTANTS"
OUT_BAND_CONF = "band.conf"
#: Deliberately NOT "BORN" — phonopy/A1 auto-discovers that filename and
#: would silently apply the non-analytic correction (Track A wires NAC off).
OUT_NAC_SIDECAR = "mp_nac.json"


class MPImportError(ValueError):
    """A harvested MP document set violates a converter precondition."""


def parse_mp_structure(sdict):
    """Hand-parse a pymatgen ``Structure.as_dict()`` payload.

    Returns ``(cell (3,3) float, frac (N,3) float, symbols [str])``. Only
    ``lattice.matrix``, ``sites[].abc`` and ``sites[].species[0].element``
    are consumed — by hand, per the purity contract (no pymatgen in src/).
    Occupancy != 1 is refused: force constants for a disordered structure
    are not a thing this importer can represent.
    """
    try:
        cell = np.asarray(sdict["lattice"]["matrix"], dtype=np.float64)
        sites = sdict["sites"]
    except (KeyError, TypeError) as exc:
        raise MPImportError(f"structure dict missing lattice/sites: {exc!r}")
    if cell.shape != (3, 3):
        raise MPImportError(f"lattice.matrix shape {cell.shape}, expected (3, 3)")
    frac = np.asarray([s["abc"] for s in sites], dtype=np.float64)
    symbols = []
    for i, s in enumerate(sites):
        spec = s["species"]
        if len(spec) != 1 or abs(float(spec[0].get("occu", 1.0)) - 1.0) > 1e-12:
            raise MPImportError(
                f"site {i}: partial/mixed occupancy {spec!r} — force constants "
                f"for disordered structures cannot be represented")
        symbols.append(str(spec[0]["element"]))
    return cell, frac, symbols


def _require_3x3(meta, key):
    if meta.get(key) is None:
        raise MPImportError(f"precondition (i): metadata field {key!r} is absent")
    m = np.asarray(meta[key], dtype=np.float64)
    if m.shape != (3, 3):
        raise MPImportError(
            f"precondition (i): metadata field {key!r} has shape {m.shape}, "
            f"expected (3, 3)")
    return m


def convert_pheasy(fixture_dir, out_dir, nac=False):
    """Convert one harvested MP pheasy fixture into the phonopy file triple.

    Parameters
    ----------
    fixture_dir : str
        Directory holding ``pheasy_metadata.json`` + ``pheasy_forceconstants
        .json`` (+ optionally ``pheasy_bandstructure.json``).
    out_dir : str
        Existing directory to write ``phonopy.yaml`` / ``FORCE_CONSTANTS`` /
        ``band.conf`` / ``mp_nac.json`` into.
    nac : bool, default False
        ``False`` (Track A, 0.6.89 behaviour — UNCHANGED): Born charges +
        dielectric are CARRIED in the sidecar only, ``nac_applied: false``.
        ``True`` (0.6.91): ADDITIONALLY write a phonopy ``BORN`` file with the
        PINNED pheasy convention (:data:`phonon_core.mp_import.nac
        .PHEASY_NAC_FACTOR` = 15.633302300230191 + the metadata's
        eps_infinity) — the DELIBERATE activation switch: A1's ``load_engine``
        auto-discovers the sibling and applies the non-analytic correction;
        A2 refuses the directory loudly (no silent NAC-less numbers). Raises
        :class:`MPImportError` when the metadata carries no Born charges or
        no dielectric.

    Returns
    -------
    dict with the four output paths under ``"phonopy_yaml"``,
    ``"force_constants"``, ``"band_conf"``, ``"nac_sidecar"``, plus an
    ``"info"`` sub-dict (atom counts, identifiers, det).
    """
    import phonopy                                # lazy: optional heavy dep
    from phonopy import Phonopy
    from phonopy.file_IO import write_FORCE_CONSTANTS
    from phonopy.structure.atoms import PhonopyAtoms

    fixture_dir = os.fspath(fixture_dir)
    out_dir = os.fspath(out_dir)
    meta = json.load(open(os.path.join(fixture_dir, _METADATA)))
    fc = np.asarray(json.load(open(os.path.join(fixture_dir, _FC))),
                    dtype=np.float64)

    # ---- preconditions -----------------------------------------------------
    smat = _require_3x3(meta, "supercell_matrix")           # (i)
    pmat = _require_3x3(meta, "primitive_matrix")           # (i)
    cell, frac, symbols = parse_mp_structure(meta["structure"])
    n_unit = len(symbols)
    n_expected = n_unit * int(round(abs(np.linalg.det(smat))))
    if fc.ndim != 4 or fc.shape[2:] != (3, 3) or fc.shape[0] != fc.shape[1]:
        raise MPImportError(
            f"precondition (ii): force-constant array has shape {fc.shape}, "
            f"expected (N_sc, N_sc, 3, 3)")
    if fc.shape[0] != n_expected:                            # (ii)
        raise MPImportError(
            f"precondition (ii): atom-count identity FAILED — FC is on "
            f"{fc.shape[0]} supercell atoms but n_unitcell_atoms * "
            f"round(|det(supercell_matrix)|) = {n_unit} * "
            f"{int(round(abs(np.linalg.det(smat))))} = {n_expected}")

    band_path = os.path.join(fixture_dir, _BAND)
    band = json.load(open(band_path)) if os.path.isfile(band_path) else None
    if band is not None:                                     # (ii-b)
        n_sites_bs = len(band["structure"]["sites"])
        n_bands = len(band["frequencies"])
        if n_bands != 3 * n_sites_bs:
            raise MPImportError(
                f"precondition (ii-b): band-count identity FAILED — the band "
                f"document stores n_bands = {n_bands} but its own structure "
                f"has {n_sites_bs} sites, requiring 3 * {n_sites_bs} = "
                f"{3 * n_sites_bs}. The MP document set is internally "
                f"inconsistent and cannot be converted.")

    # ---- build + write the triple -------------------------------------------
    ph = Phonopy(PhonopyAtoms(symbols=symbols, cell=cell, scaled_positions=frac),
                 supercell_matrix=smat, primitive_matrix=pmat, log_level=0)
    # ATOM ORDER: pheasy's supercell ordering == phonopy's own (pinned; the
    # ordering negative control in tests/mp_import proves a permutation bites).
    ph.force_constants = fc
    yaml_path = os.path.join(out_dir, OUT_PHONOPY_YAML)
    fc_path = os.path.join(out_dir, OUT_FORCE_CONSTANTS)
    ph.save(yaml_path, settings={"force_constants": False})
    write_FORCE_CONSTANTS(fc, filename=fc_path)

    born_path = None
    if nac:                                          # 0.6.91: opt-in NAC
        from phonon_core.mp_import.nac import write_phonopy_born
        if meta.get("born") is None or meta.get("epsilon_static") is None:
            raise MPImportError(
                "nac=True requires Born charges AND a dielectric tensor in "
                f"the metadata; got born={'present' if meta.get('born') else None}, "
                f"epsilon_static={'present' if meta.get('epsilon_static') else None}")
        born_path = write_phonopy_born(
            ph, meta["born"], meta["epsilon_static"],
            os.path.join(out_dir, "BORN"))

    conf_path = os.path.join(out_dir, OUT_BAND_CONF)
    if band is not None:
        _write_band_conf(conf_path, symbols, smat, pmat, band)
    else:
        conf_path = None

    sidecar_path = os.path.join(out_dir, OUT_NAC_SIDECAR)
    with open(sidecar_path, "w") as fh:
        json.dump({
            "nac_applied": bool(nac),
            "born": meta.get("born"),
            # NOTE: the MP field named epsilon_static actually holds the
            # ION-CLAMPED high-frequency dielectric (eps_infinity) — see
            # CLAUDE.md ("empirically pinned; derivation owed at 0.6.91").
            "epsilon_static_is_eps_infinity": meta.get("epsilon_static"),
            "epsilon_electronic": meta.get("epsilon_electronic"),
            "sum_rules_breaking": meta.get("sum_rules_breaking"),
            "has_nac_flag_from_mp_band_doc":
                None if band is None else band.get("has_nac"),
            "pinned_nac_factor_provisional_0691": 15.633302300230191,
            "identifier": meta.get("identifier"),
            "phonon_method": meta.get("phonon_method"),
            "code": meta.get("code"),
        }, fh, indent=1)

    # ---- precondition (iii): round-trip through phonopy's own loader --------
    reloaded = phonopy.load(phonopy_yaml=yaml_path,
                            force_constants_filename=fc_path,
                            symmetrize_fc=False, log_level=0)
    if len(reloaded.supercell) != n_expected:
        raise MPImportError(
            f"precondition (iii): round-trip supercell has "
            f"{len(reloaded.supercell)} atoms, expected {n_expected}")

    return {
        "phonopy_yaml": yaml_path,
        "force_constants": fc_path,
        "band_conf": conf_path,
        "nac_sidecar": sidecar_path,
        "born": born_path,
        "info": {
            "n_unitcell_atoms": n_unit,
            "n_supercell_atoms": int(fc.shape[0]),
            "n_primitive_atoms": len(reloaded.primitive),
            "det_supercell_matrix": int(round(abs(np.linalg.det(smat)))),
            "identifier": meta.get("identifier"),
            "mp_has_nac_flag": None if band is None else band.get("has_nac"),
        },
    }


def _write_band_conf(path, symbols, smat, pmat, band):
    """phonopy band.conf from the band document's kpath + labels_dict.

    Spelled one SEGMENT PER LABEL PAIR (comma-joined), which phonopy accepts
    verbatim and which needs no discontinuity inference: MP's flat ``kpath``
    label sequence does not say which junctions are breaks.
    """
    labels_dict = band["labels_dict"]
    kpath = band["kpath"]
    if not kpath or any(lbl not in labels_dict for lbl in kpath):
        missing = [lbl for lbl in kpath if lbl not in labels_dict]
        raise MPImportError(f"band.conf: kpath labels missing from "
                            f"labels_dict: {missing!r}")

    def q_str(lbl):
        return " ".join(f"{x:.10g}" for x in labels_dict[lbl])

    def pretty(lbl):
        # '\\Gamma' -> '$\\Gamma$'; plain names pass through
        return f"${lbl}$" if lbl.startswith("\\") else lbl

    segs = [f"{q_str(a)}  {q_str(b)}" for a, b in zip(kpath, kpath[1:])]
    lbls = [f"{pretty(a)} {pretty(b)}" for a, b in zip(kpath, kpath[1:])]
    dim = " ".join(str(int(round(x))) for x in np.asarray(smat).diagonal()) \
        if np.allclose(smat, np.diag(np.diagonal(smat))) else \
        " ".join(str(int(round(x))) for x in np.asarray(smat).reshape(-1))
    pa = " ".join(f"{x:.10g}" for x in np.asarray(pmat).reshape(-1))
    with open(path, "w") as fh:
        fh.write(f"ATOM_NAME = {' '.join(dict.fromkeys(symbols))}\n")
        fh.write(f"DIM = {dim}\n")
        fh.write(f"PRIMITIVE_AXES = {pa}\n")
        fh.write(f"BAND = {', '.join(segs)}\n")
        fh.write(f"BAND_LABELS = {', '.join(lbls)}\n")
        fh.write("EIGENVECTORS = .TRUE.\n")
        fh.write("FORCE_CONSTANTS = READ\n")
