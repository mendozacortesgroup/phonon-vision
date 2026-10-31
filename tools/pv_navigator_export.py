#!/usr/bin/env python3
"""pv_navigator_export — bake a self-contained navigator JSON from band + mesh.

STEP 1a: a 2D band panel + a 3D bands-over-BZ surface, linked by a click. NO
kinematics, NO eigenvectors in the payload — only derived per-mode scalars
(frequency + Sn/Se character) and a robustly-built surface. The hub does the
physics: phonon_core's parsers apply the conventions, phonon_core.projection
computes the character, phonon_core.bz builds the first Brillouin zone.

    pv_navigator_export.py BAND.hdf5 [MESH.hdf5] -o navigator_data.json
        MESH.hdf5             optional — in LEGACY mode, without it the
                              bands-over-BZ surface block (and bz/ibz polygons
                              + z_range) is omitted and the viewer shows
                              dispersion + atoms only. In --engine mode the
                              surface is built from D(q) regardless (0.6.85).
        [--group-a Sn[,..]]   species whose |e|^2 fraction is the "character"
        [--grid N]            surface grid resolution per axis (default 96;
                              inert only when no surface is built — legacy
                              mode without a mesh; in --engine mode it always
                              drives the exact surface)
        [--dim {2,3}]         material dimensionality (default 2); 3 omits the
                              in-plane surface/BZ blocks (a q3=0 slice would be
                              mislabeled as "the BZ" for a 3D crystal)
        [--total-dos F]       phonopy total_dos.dat -> dashed Total curve
        [--proj-dos F]        phonopy projected_dos.dat -> per-species fills
                              (columns merged by symbol, or by --pdos-conf)
        [--pdos-conf F]       optional pdos.conf PDOS-line grouping override
        [--also-html]         also write a self-contained index.html + data .js
        [--engine PHONOPY_YAML]  OPT-IN engine-backed mode (0.6.80, Level-2):
                              band frequencies+eigenvectors AND the 3D surface
                              are computed EXACTLY from D(q) via the Stage-1 A1
                              engine (FORCE_CONSTANTS auto-discovered next to
                              PHONOPY_YAML) instead of parsed/interpolated; the
                              payload gains a `provenance` block. The band file
                              is CROSS-CHECKED against the engine and a mismatch
                              REFUSES to emit (never a silent fallback). Needs
                              the optional `engine` extra (phonopy).
        [--mesh-modes-grid N] engine mode (dim 2): in-BZ grid per axis
                              for the `mesh_modes` block (default 24; exact
                              modes incl. eigenvectors for Stage-3
                              click-anywhere — independent of --grid because
                              eigenvectors, not scalars, drive the size).

The 3D surface is built by SCATTERED-DATA interpolation on Cartesian q (the
phonopy mesh is Wigner-Seitz-folded and does NOT lie on a [0,1) raster), then
masked to the first-BZ polygon. Imaginary (negative) frequencies are kept.
In --engine mode the surface is instead EVALUATED at every in-BZ grid node
(no interpolation) and NO mesh file is required (0.6.85): the engine's
phonopy.yaml identifies the material and the band cross-check is a far
stronger consistency gate than a symbol/lattice comparison. A mesh may still
be passed (parsed and same-material-checked, for back-compat) but contributes
no number to the payload.
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "src"))

import numpy as np

from phonon_core import load_band_hdf5, load_mesh_hdf5, projection, bz
from phonon_core.kinematics import canonicalize_eigenvector_phase
from phonon_core.io.json_v1 import _split_eigvecs   # reuse the wire-format splitter
from phonon_core.radii import jmol_color, covalent_radius   # species color + radius


def _die(msg, code=1):
    print(f"pv_navigator_export: error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _character(eigvecs, symbols, group_a):
    """Per-mode fraction of |e|^2 on the group-A species, via the hub's
    projection. eigvecs (..., N_atoms, 3) -> (...,) in [0,1]."""
    weights, species = projection.species_weights_array(eigvecs, symbols)
    idx = [species.index(s) for s in group_a if s in species]
    if not idx:
        _die(f"none of group-A species {group_a} present in {species}", 2)
    return weights[..., idx].sum(axis=-1)


def _to_jsonable(arr):
    """Nested lists with None for non-finite (JSON has no NaN; plotly reads
    null as a gap)."""
    a = np.asarray(arr, dtype=float)
    out = a.tolist()

    def fix(x):
        if isinstance(x, list):
            return [fix(v) for v in x]
        return x if np.isfinite(x) else None
    return fix(out)


def _round_6g(arr):
    """Round every real/imag component to 6 SIGNIFICANT figures via the %.6g
    text round-trip: float(f"{x:.6g}") lands exactly on a ~6-digit decimal, so
    json's shortest-repr float serialization then writes ~6 digits instead of
    up to 17 (~2.4x smaller eigenvector text). Used for the mesh_modes
    eigenvectors (0.6.81 precision policy) and, since 0.6.92, the surface
    character/weight fractions; frequencies/q ALWAYS stay full precision."""
    a = np.asarray(arr)
    if np.iscomplexobj(a):
        return _round_6g(a.real) + 1j * _round_6g(a.imag)
    flat = np.fromiter((float(f"{x:.6g}") for x in a.ravel()),
                       dtype=np.float64, count=a.size)
    return flat.reshape(a.shape)


def _tex_norm(label):
    """Wrap an ATOMIC tick label in $...$ if it contains TeX markup (a
    backslash, '^' or '_') and is not already $-wrapped. '$\\Gamma$' and plain
    'M' pass through unchanged."""
    s = str(label)
    if s and ("\\" in s or "^" in s or "_" in s) \
            and not (s.startswith("$") and s.endswith("$")):
        return "$" + s + "$"
    return s


def _path_ticks(labels, q_frac, starts, ends):
    """Merged tick labels + trace-break q-index pairs for (possibly
    DISCONTINUOUS) band paths. Pure — unit-testable without HDF5.

    labels : phonopy-style per-segment (start_label, end_label) pairs
    q_frac : (Nq, 3) FRACTIONAL band-path q-points
    starts, ends : per-segment first/last q-indices

    The junction between segments i and i+1 shares one plotted distance but
    has TWO q-indices (e = ends[i], s = starts[i+1]). It is DISCONTINUOUS iff
    the q's differ: not allclose(q_frac[e], q_frac[s], atol=1e-10) on
    fractional q. At a discontinuous junction with differing labels the tick
    merges to phonopy's "A|B" convention; matching labels (or a continuous
    junction) keep the plain end label. TeX normalization (_tex_norm) is
    applied to each ATOMIC label BEFORE any "|" join, so a merged tick can
    read "$X^{\\prime}$|M".

    CONVENTION: the tick q-index (and therefore high_symmetry_points.q_cart)
    stays the END index of the incoming segment — at a discontinuous junction
    the two sides sit at different q, and we keep the incoming side's q,
    matching the tick_qindex = [0] + ends layout used since 0.6.1.

    Returns (tick_labels, breaks): tick_labels has n_seg+1 entries; breaks is
    [[e, s], ...] for the DISCONTINUOUS junctions only (continuous junctions
    keep the polyline connected).
    """
    q = np.asarray(q_frac, dtype=float)
    ticks = [_tex_norm(labels[0][0])]
    breaks = []
    for i in range(len(labels) - 1):
        e, s = int(ends[i]), int(starts[i + 1])
        end_lab, start_lab = labels[i][1], labels[i + 1][0]
        continuous = np.allclose(q[e], q[s], atol=1e-10)
        if not continuous:
            breaks.append([e, s])
        if continuous or end_lab == start_lab:
            ticks.append(_tex_norm(end_lab))
        else:
            ticks.append(_tex_norm(end_lab) + "|" + _tex_norm(start_lab))
    ticks.append(_tex_norm(labels[-1][1]))
    return ticks, breaks


# ---------------- DOS (1d-2): total + projected, presence-keyed ----------------

def _load_dos_dat(path, what, ncols=None):
    """np.loadtxt with '#' comments skipped; optional EXACT column-count check
    (no guessing — a mismatch dies with an explicit message)."""
    try:
        arr = np.loadtxt(path, comments="#", ndmin=2)
    except Exception as exc:               # noqa: BLE001 — surface the cause
        _die(f"could not read {what} file {path}: {exc}", 2)
    if ncols is not None and arr.shape[1] != ncols:
        _die(f"{what} file {path} has {arr.shape[1]} columns, expected {ncols}", 2)
    return arr


def _parse_pdos_conf(path):
    """Tolerant pdos.conf reader: 'PDOS' / 'ATOM_NAME' are matched as
    SUBSTRINGS of a line (real user files carry corrupt prefixes such as
    'uATOM_NAME'). Returns (groups, names): groups = list of 1-based atom-index
    lists from the comma-separated PDOS line, names = ATOM_NAME tokens; either
    may be None if absent."""
    groups = None
    names = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "=" not in line:
                continue
            key, rhs = line.split("=", 1)
            if "PDOS" in key:
                groups = [[int(tok) for tok in grp.split()]
                          for grp in rhs.split(",") if grp.strip()]
            elif "ATOM_NAME" in key:
                names = rhs.split()
    return groups, names


def _pdos_groups(symbols, conf_path=None):
    """[(label, [1-based atom indices]), ...] for merging per-ATOM projected-DOS
    columns. DEFAULT: group by SYMBOL from the band material, ordered by first
    appearance (e.g. Sn=[1], Se=[2,3]). With a pdos.conf, the PDOS line's
    groups override; each group is labeled by its member atoms' symbols
    (homogeneous -> the symbol, mixed -> symbols joined with '+'); ATOM_NAME is
    used only as a cross-check warning if it disagrees with the derived labels."""
    n = len(symbols)
    if conf_path is None:
        order = list(dict.fromkeys(symbols))
        return [(sym, [i + 1 for i, s in enumerate(symbols) if s == sym])
                for sym in order]
    groups, names = _parse_pdos_conf(conf_path)
    if not groups:
        _die(f"pdos conf {conf_path} has no parsable PDOS line", 2)
    out = []
    for grp in groups:
        for i in grp:
            if not 1 <= i <= n:
                _die(f"pdos conf {conf_path}: atom index {i} outside [1, {n}]", 2)
        uniq = list(dict.fromkeys(symbols[i - 1] for i in grp))
        out.append(("+".join(uniq), list(grp)))
    if names and names != [lab for lab, _ in out]:
        print(f"[pv_navigator_export] WARNING: ATOM_NAME {names} in {conf_path} "
              f"disagrees with symbol-derived group labels "
              f"{[lab for lab, _ in out]}; using the derived labels",
              file=sys.stderr)
    return out


def _trim_zero_bins(freq, cols):
    """Slice off all-zero head/tail rows (a row counts as zero when EVERY
    column is 0), keeping ONE margin bin each side so padding does not stretch
    the shared frequency axis. Degenerate all-zero input is returned as-is."""
    nz = np.nonzero(np.any(np.asarray(cols) != 0, axis=1))[0]
    if nz.size == 0:
        return freq, cols
    lo = max(int(nz[0]) - 1, 0)
    hi = min(int(nz[-1]) + 1, len(freq) - 1)
    return freq[lo:hi + 1], cols[lo:hi + 1]


#: Surface symmetry-reduction tolerances (0.6.92). Z is STRICT and gates the
#: whole op — frequencies are gauge-free, so a z mismatch means the crystal
#: truly lacks that mirror (or the field is interpolated, where symmetrizing
#: would launder interpolation asymmetry as physics). Character/weights are
#: gauge-SENSITIVE near degeneracies (the eigensolver may mix a degenerate
#: subspace differently at symmetry-image nodes), so they gate PER NODE: a
#: gauge-unstable node simply stays explicitly stored, and the op survives.
_SURF_SYM_ZTOL = 1e-6     # THz, max over merged pairs and bands
_SURF_SYM_CTOL = 1e-4     # character / per-species weight, per merged pair


def _surface_symmetry_reduce(inside, z, char, weights):
    """Deduplicate a masked surface grid by VERIFIED mirror symmetry.

    Candidate ops are the three grid-exact index maps on a symmetric axis
    (kx[j] == -kx[G-1-j], ky likewise): mirror_x (kx -> -kx), mirror_y
    (ky -> -ky) and their composition c2 (time reversal — physically exact for
    any material, but still verified rather than assumed). Ops act on grid
    INDICES, so reconstruction is exact indexing — no resampling anywhere.

    Every op is verified against the computed field BEFORE use (z residual
    <= _SURF_SYM_ZTOL over all in-BZ pairs, all bands). A kept op merges a
    node into a lower-index orbit representative only where character AND
    weights also agree (<= _SURF_SYM_CTOL); nodes that disagree (gauge noise
    at near-degeneracies) remain their own representative with their own
    exact values — compression never alters what a node displays beyond the
    published tolerances.

    Parameters: inside (G,G) bool; z, char (Nb,G,G) float (NaN outside);
    weights (Nb,nS,G,G) float or None.

    Returns (node_map, node_idx, sym_info): node_map (G,G) int, -1 outside,
    else index into the stored-node list; node_idx (N,2) int the (i,j) of
    each stored node; sym_info dict for provenance.
    """
    G_y, G_x = inside.shape
    # A node participates in dedup only when EVERY field is finite there (an
    # interpolation hole must stay a hole — merging would launder it as
    # physics; a NaN also breaks the >tol comparisons silently, since any
    # comparison with NaN is False).
    clean = inside & np.isfinite(z).all(axis=0) & np.isfinite(char).all(axis=0)
    if weights is not None:
        clean &= np.isfinite(weights).all(axis=(0, 1))
    # index maps as array FLIPS — exact on a symmetric axis, fully vectorized
    flips = {
        "mirror_x": (slice(None), slice(None, None, -1)),      # kx -> -kx
        "mirror_y": (slice(None, None, -1), slice(None)),      # ky -> -ky
        "c2": (slice(None, None, -1), slice(None, None, -1)),  # both
    }
    kept, max_dz = [], 0.0
    for name, fl in flips.items():
        both = clean & clean[fl]
        if not both.any():
            continue
        dz = np.abs(z - z[(slice(None),) + fl])[:, both]
        worst = float(dz.max())
        if worst <= _SURF_SYM_ZTOL:
            kept.append(name)
            max_dz = max(max_dz, worst)

    def _op(name, i, j):
        oi = G_y - 1 - i if name in ("mirror_y", "c2") else i
        oj = G_x - 1 - j if name in ("mirror_x", "c2") else j
        return oi, oj

    node_map = np.full((G_y, G_x), -1, dtype=int)
    node_idx = []
    max_dc = 0.0
    for i in range(G_y):
        for j in range(G_x):
            if not inside[i, j]:
                continue
            rep, rep_dc = None, 0.0
            for name in (kept if clean[i, j] else ()):
                oi, oj = _op(name, i, j)
                if (oi, oj) >= (i, j) or not clean[oi, oj]:
                    continue                       # only merge toward lower index
                cand = int(node_map[oi, oj])
                ri, rj = node_idx[cand]            # compare against the STORED
                dc = float(np.max(np.abs(char[:, i, j] - char[:, ri, rj])))
                if dc > _SURF_SYM_CTOL:            # representative, so the
                    continue                       # published tol is per node
                if weights is not None and float(np.max(np.abs(
                        weights[:, :, i, j] - weights[:, :, ri, rj]))) > _SURF_SYM_CTOL:
                    continue
                if rep is None or cand < rep:
                    rep, rep_dc = cand, dc
            if rep is not None:
                node_map[i, j] = rep
                max_dc = max(max_dc, rep_dc)
            else:
                node_map[i, j] = len(node_idx)
                node_idx.append((i, j))
    sym_info = {
        "ops": kept,
        "max_dz_THz": max_dz,
        "max_dchar": max_dc,
        "nodes_total": int(inside.sum()),
        "nodes_stored": len(node_idx),
    }
    return node_map, np.asarray(node_idx, dtype=int), sym_info


def _ray_to_hull(points_xy, poly):
    """Radial clamp onto a convex polygon containing Γ (0.7.1, Tier 0 of the
    extended-zone work; math shared with magnon-vision's rayToHull): for each
    point p, the Wigner-Seitz cell is convex about Γ, so the Γ→p ray has a
    UNIQUE exit point t·p with

        t = min over edges (a,b) with outward normal n=(b_y−a_y, a_x−b_x),
            n·p > 0  of  (n·a)/(n·p)

    Points already inside get t ≥ 1 clamped to 1 (returned unchanged).
    points_xy: (N,2); poly: (M,2) ordered ring. Returns (N,2)."""
    p = np.asarray(points_xy, dtype=float)
    V = np.asarray(poly, dtype=float)
    a = V
    b = np.roll(V, -1, axis=0)
    n_out = np.column_stack([b[:, 1] - a[:, 1], a[:, 0] - b[:, 0]])   # (M,2)
    num = np.einsum("md,md->m", n_out, a)                              # n·a
    den = p @ n_out.T                                                  # (N,M) n·p
    with np.errstate(divide="ignore", invalid="ignore"):
        t_edge = np.where(den > 1e-12, num[None, :] / den, np.inf)
    t = np.minimum(t_edge.min(axis=1), 1.0)
    return p * t[:, None]




def _emit_surface_bands(inside, z_stack, c_stack, w_stack):
    """Shared 0.6.92 emitter for BOTH surface branches: symmetry-reduce the
    dense masked fields, then emit flat per-node arrays. Precision follows
    the 0.6.81 rule exactly: FREQUENCIES (z) stay full precision — the 0.6.80
    single-source invariant (band vs surface < 1e-6 THz) keeps its teeth —
    while the bounded display fractions (character, weights) get 6
    significant figures. Returns (surf_bands, node_map_list, sym_info,
    node_idx) — node_idx is the stored-node (i, j) list, so additive
    channels can emit aligned flat arrays."""
    node_map, node_idx, sym = _surface_symmetry_reduce(
        inside, z_stack, c_stack, w_stack)
    ii, jj = node_idx[:, 0], node_idx[:, 1]
    surf_bands = []
    for b in range(z_stack.shape[0]):
        entry = {"z": _to_jsonable(z_stack[b, ii, jj]),
                 "character": _to_jsonable(_round_6g(c_stack[b, ii, jj]))}
        if w_stack is not None:
            entry["weights"] = [_to_jsonable(_round_6g(w_stack[b, s, ii, jj]))
                                for s in range(w_stack.shape[1])]
        surf_bands.append(entry)
    return surf_bands, node_map.tolist(), sym, node_idx


def expand_surface(surf):
    """Expand a compact (0.6.92, node_map) surface block to the dense legacy
    in-memory shape: bands[b].z/character/weights as [ky][kx] grids with None
    outside the BZ. A pre-0.6.92 dense block is returned unchanged. This is
    the SINGLE authoritative reader — the viewer's load shim mirrors it."""
    if "node_map" not in surf:
        return surf
    nm = surf["node_map"]
    G_y, G_x = len(nm), len(nm[0])

    def grid_of(flat):
        return [[(flat[nm[i][j]] if nm[i][j] >= 0 else None)
                 for j in range(G_x)] for i in range(G_y)]

    out = {k: v for k, v in surf.items()
           if k not in ("node_map", "n_nodes", "bands")}
    out["bands"] = []
    for bd in surf["bands"]:
        entry = {"z": grid_of(bd["z"]), "character": grid_of(bd["character"])}
        if "weights" in bd:
            entry["weights"] = [grid_of(w) for w in bd["weights"]]
        out["bands"].append(entry)
    return out


def build_payload(band_path, mesh_path, *, group_a=("Sn",), grid=96, dim=2,
                  total_dos=None, proj_dos=None, pdos_conf=None,
                  surface_weights=True, engine=None, mesh_modes_grid=24,
                  band_connect=False):
    if dim not in (2, 3):
        _die(f"dim must be 2 or 3, got {dim!r}", 2)
    band = load_band_hdf5(band_path)
    mat = band.material
    # mesh OPTIONAL in BOTH dims (0.6.15): without it the bands-over-BZ surface
    # block is omitted (the viewer keys the right panel off block PRESENCE).
    # When given, the same-material consistency check runs exactly as before.
    mesh = None
    if mesh_path is not None:
        mesh = load_mesh_hdf5(mesh_path, material=mat)
        # consistency: same crystal (the navigator overlays them)
        if mesh.material.symbols != mat.symbols or \
                not np.allclose(mesh.material.reciprocal_lattice, mat.reciprocal_lattice):
            _die("band and mesh are not the same material", 2)
    if dim == 2 and mesh is None:
        if engine is not None:
            # engine mode builds the surface + mesh_modes from D(q) directly;
            # a mesh file is not a prerequisite (0.6.85).
            print("[pv_navigator_export] engine mode without mesh: surface + "
                  "mesh_modes from D(q) only", file=sys.stderr)
        else:
            print("[pv_navigator_export] no mesh given: bands-over-BZ surface "
                  "omitted (--grid inert)", file=sys.stderr)

    recip = np.asarray(mat.reciprocal_lattice)          # rows b_i
    symbols = list(mat.symbols)
    group_a = list(group_a)

    # ---- OPT-IN engine mode (0.6.80, Level-2 Stage 2 Bump 2) ----
    # band + surface computed EXACTLY from D(q) via the Stage-1 A1 engine; the
    # payload gains `provenance`. engine=None -> LEGACY, untouched below.
    eng_ph = None
    eng_info = None
    band_max_dw = None
    if engine is not None:
        try:
            from phonon_core.engine import load_engine, evaluate_dq
            eng_ph = load_engine(engine)
        except ImportError:
            # STRICT: no silent downgrade to the parsed/interpolated path — a
            # present provenance block must ALWAYS be trustworthy.
            _die("engine mode requires phonopy; pip install phonon-core[engine]", 2)
        import hashlib
        import phonopy as _phonopy

        def _md5_file(p):
            h = hashlib.md5()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()

        _fc_path = os.path.join(os.path.dirname(os.path.abspath(engine)),
                                "FORCE_CONSTANTS")   # load_engine's discovery rule
        eng_info = {
            "kind": "phonopy-A1",
            "phonopy_version": str(_phonopy.__version__),
            "fc_path": _fc_path if os.path.isfile(_fc_path) else None,
            "fc_md5": _md5_file(_fc_path) if os.path.isfile(_fc_path) else None,
            "phonopy_yaml_md5": _md5_file(engine),
        }

        # ENGINE-MATERIAL GATE (0.6.85): in engine mode the phonopy.yaml — not
        # a mesh file — is the material authority. Compare the BAND material
        # against the engine's Phonopy primitive (symbols + reciprocal lattice,
        # rows b_i via inv(cell).T, the SAME tolerance as the mesh gate) and die
        # with a NAMED message. This runs BEFORE the band cross-check below, so
        # a wrong-material pairing reads honestly ("not the same material")
        # rather than as a confusing max|dω| cross-check failure.
        _eng_prim = eng_ph.primitive
        if list(_eng_prim.symbols) != list(mat.symbols) or not np.allclose(
                np.linalg.inv(np.asarray(_eng_prim.cell)).T,
                mat.reciprocal_lattice):
            _die("band and engine phonopy.yaml are not the same material", 2)

    # 0.6.85: engine mode builds the in-plane surface from D(q) WITHOUT a mesh;
    # legacy still requires one. Single predicate set for "this payload has a
    # surface panel" — surface/bz_polygon/ibz_polygon/z_range live and die
    # together (the viewer keys the right panel off block PRESENCE).
    engine_surface = (dim == 2 and eng_ph is not None)
    legacy_surface = (dim == 2 and mesh is not None and eng_ph is None)
    has_surface = engine_surface or legacy_surface

    # ---- band panel ----
    b_freq = np.asarray(band.frequencies)               # (Nq, Nb)
    if eng_ph is not None:
        # evaluate D(q) at EACH band q; CROSS-CHECK against the band file's own
        # frequencies+eigenvectors (Stage-1 PATH reproduction, reused via
        # phonon_core.engine.compare with its degenerate-subspace projector
        # handling). A mismatch is the FC<->band provenance error the arc
        # exists to catch: REFUSE to emit, never fall back.
        from phonon_core.engine.compare import mode_agreement
        _file_eig = np.asarray(band.eigenvectors)
        e_freq = np.empty_like(b_freq)
        e_eig = np.empty_like(_file_eig, dtype=np.complex128)
        for _qi in range(b_freq.shape[0]):
            _f, _v = evaluate_dq(eng_ph, band.q_points[_qi])
            e_freq[_qi] = _f
            e_eig[_qi] = _v
        _dw, _ov, _pd = mode_agreement(b_freq, _file_eig, e_freq, e_eig)
        if _dw > 1e-6 or _ov < 1 - 1e-6 or _pd > 1e-6:
            _die(f"engine/band cross-check FAILED — the FORCE_CONSTANTS next to "
                 f"{os.path.basename(engine)} do not reproduce "
                 f"{os.path.basename(band_path)}: max|dω| = {_dw:.3e} THz "
                 f"(tol 1e-06), min eigenvector overlap = {_ov:.9f} "
                 f"(tol 1 - 1e-06), max degenerate-projector deviation = "
                 f"{_pd:.3e} (tol 1e-06). REFUSING to emit an "
                 f"engine-provenance payload from mismatched inputs.", 3)
        band_max_dw = float(_dw)
        b_freq = e_freq                                  # engine is the source
    _band_eig_src = (e_eig if eng_ph is not None
                     else np.asarray(band.eigenvectors))  # single eigvec source
    b_char = _character(_band_eig_src, symbols, group_a)  # (Nq,Nb)
    # q_cart: dim==2 keeps the xy projection (in-plane navigator); dim==3 keeps
    # ALL THREE Cartesian components (no surface to project onto).
    b_qcart = np.asarray(band.q_points) @ recip          # (Nq,3)
    if dim == 2:
        b_qcart = b_qcart[:, :2]                         # (Nq,2)

    # eigenvector egress for 1b (the JS atom-motion animator). Canonicalize the
    # global phase PER MODE — each (Na,3) block independently — so φ=0 is
    # reproducible; canonicalizing the whole (Nq,Nb,Na,3) array at once would
    # impose a single global phase across all modes, which is wrong. Then split
    # real/imag with json_v1's OWN splitter so the JS decoder and the lossless
    # round-trip match the frozen wire format exactly (not hand-rolled here).
    b_eig = _band_eig_src                               # (Nq, Nb, Na, 3) complex
    ev_canon = np.empty_like(b_eig, dtype=np.complex128)
    for _qi in range(b_eig.shape[0]):
        for _bi in range(b_eig.shape[1]):
            ev_canon[_qi, _bi] = canonicalize_eigenvector_phase(b_eig[_qi, _bi])
    b_eig_real, b_eig_imag = _split_eigvecs(ev_canon)
    dist = np.asarray(band.distances)
    seg = np.asarray(band.segment_nqpoint)
    ends = np.cumsum(seg) - 1
    starts = np.concatenate([[0], np.cumsum(seg)[:-1]])
    # high-symmetry tick labels from the path, DISCONTINUITY-AWARE (0.6.16):
    # merged "A|B" at discontinuous junctions, TeX-normalized atomic labels,
    # plus the [e, s] q-index pairs where the plotted line must break.
    labels, path_breaks = _path_ticks(band.labels, np.asarray(band.q_points),
                                      starts, ends)
    tick_positions = [float(dist[0])] + [float(dist[e]) for e in ends]
    segments = [[float(dist[s]), float(dist[e])] for s, e in zip(starts, ends)]

    # the band q-index of each labeled tick (Γ at path start, then each seg end)
    tick_qindex = [0] + [int(e) for e in ends]
    high_symmetry_points = [
        {"label": labels[i],
         "q_cart": [float(x) for x in b_qcart[qi]],
         "distance": float(tick_positions[i])}
        for i, qi in enumerate(tick_qindex)]

    # IBZ wedge = triangle [Γ, M, K] from Γ and the first two DISTINCT non-Γ
    # high-symmetry points on the path. For the standard Γ-M-K-Γ path this is the
    # holohedral (lattice / point-group-of-the-lattice) hexagonal irreducible
    # wedge, whose perimeter the band path traces. NOTE: this is the *lattice*
    # irreducible wedge derived from the path, NOT a full space-group IBZ
    # analysis (which for a 3-fold crystal would be larger); general-symmetry IBZ
    # is out of scope for the navigator. dim==3 OMITS it entirely: it is a q3=0
    # in-plane construction — a slice, not the zone, for a 3D crystal. It is also
    # omitted without a mesh: the surf3d panel is its ONLY consumer, so the
    # surface/bz_polygon/ibz_polygon/z_range blocks live and die together.
    if has_surface:
        gamma_xy = np.array([b_qcart[0, 0], b_qcart[0, 1]])
        ibz_verts = [[float(gamma_xy[0]), float(gamma_xy[1])]]
        for hsp in high_symmetry_points[1:]:
            if np.linalg.norm(np.array(hsp["q_cart"]) - gamma_xy) > 1e-8:
                if all(np.linalg.norm(np.array(hsp["q_cart"]) - np.array(v)) > 1e-8
                       for v in ibz_verts):
                    ibz_verts.append(hsp["q_cart"])
            if len(ibz_verts) == 3:
                break

    # ---- 3D surface (scattered interpolation on Cartesian q, masked to BZ) ----
    # dim==3 OR no mesh: skipped ENTIRELY (the interpolation work, not just the
    # keys). The surface, bz_polygon and ibz_polygon are q3=0 in-plane
    # constructions — a slice, not the zone, for a 3D crystal — so a 3D payload
    # omits them rather than mislabel a slice as "the BZ"; without a mesh there
    # is nothing to interpolate. No isosurface replaces them (by decision).
    if engine_surface:
        # ---- ENGINE-EXACT surface (0.6.80): D(q) evaluated at every in-BZ
        # grid node — no interpolation, no gaps. NO mesh file needed (0.6.85):
        # the engine's phonopy.yaml is the material authority (gate above) and
        # nothing in this branch reads a mesh — the surface is pure D(q).
        from matplotlib.path import Path

        poly = bz.first_brillouin_zone_2d(recip)            # (Nv,2) ordered ring

        # regular grid over the BZ bounding box — IDENTICAL construction to the
        # legacy branch (same kx/ky axes, same inside mask), so the two modes
        # differ only in how node values are obtained.
        xmin, ymin = poly.min(axis=0)
        xmax, ymax = poly.max(axis=0)
        padx = 0.04 * (xmax - xmin)
        pady = 0.04 * (ymax - ymin)
        # EXACTLY symmetric axes (0.6.92): kx[j] == -kx[grid-1-j] bitwise, so
        # the mirror index maps in _surface_symmetry_reduce are exact geometry,
        # not approximations. The Γ-centred BZ makes the bbox symmetric anyway
        # (|xmin - -xmax| ~ 1e-16); this construction removes even that noise.
        half_x = max(abs(float(xmin)), abs(float(xmax))) + padx
        half_y = max(abs(float(ymin)), abs(float(ymax))) + pady
        off = (grid - 1) / 2.0
        kx = (np.arange(grid) - off) * (2.0 * half_x / (grid - 1))
        ky = (np.arange(grid) - off) * (2.0 * half_y / (grid - 1))
        KX, KY = np.meshgrid(kx, ky)                         # (grid, grid), 'xy'
        gpts = np.column_stack([KX.ravel(), KY.ravel()])
        inside = Path(poly).contains_points(gpts).reshape(grid, grid)

        # Cartesian (kx, ky, 0) -> fractional q = k @ inv(recip) (rows b_i, no
        # 2π); for these layered materials c* ⊥ the plane so q_frac[2] ≈ 0.
        inv_recip = np.linalg.inv(recip)
        # SMOOTH RIM (0.7.1, Tier 0): nodes OUTSIDE the zone are not
        # discarded — each is clamped onto the exact zone boundary along its
        # Γ-ray (_ray_to_hull) and D(q) is EVALUATED there (engine mode has
        # no reason to resample: the exact solve at the clamp is the same
        # price as anywhere else). The stored grid then has a value at EVERY
        # cell; the viewer moves those vertices onto the hexagon and the rim
        # follows the zone exactly instead of staircasing along the grid.
        outside = ~inside
        out_pts = gpts[outside.ravel()]                      # (No, 2)
        out_clamped = _ray_to_hull(out_pts, poly)            # (No, 2) on the rim
        nodes = gpts[inside.ravel()]                         # (Nn, 2) in-BZ
        q_frac_nodes = np.column_stack(
            [nodes, np.zeros(len(nodes))]) @ inv_recip       # (Nn, 3)
        q_frac_rim = np.column_stack(
            [out_clamped, np.zeros(len(out_clamped))]) @ inv_recip
        node_freq_l, node_eig_l = [], []
        for _k in range(len(q_frac_nodes)):
            _f, _v = evaluate_dq(eng_ph, q_frac_nodes[_k])
            node_freq_l.append(_f)
            node_eig_l.append(_v)
        node_freq = np.asarray(node_freq_l)                  # (Nn, Nb)
        node_eig = np.asarray(node_eig_l)                    # (Nn, Nb, Na, 3)
        node_char = _character(node_eig, symbols, group_a)   # (Nn, Nb)
        rim_freq_l, rim_eig_l = [], []
        for _k in range(len(q_frac_rim)):
            _f, _v = evaluate_dq(eng_ph, q_frac_rim[_k])
            rim_freq_l.append(_f)
            rim_eig_l.append(_v)
        rim_freq = (np.asarray(rim_freq_l) if rim_freq_l
                    else np.zeros((0, node_freq.shape[1])))
        rim_eig = (np.asarray(rim_eig_l) if rim_eig_l
                   else np.zeros((0,) + node_eig.shape[1:], dtype=complex))
        rim_char = (_character(rim_eig, symbols, group_a) if len(rim_eig)
                    else np.zeros((0, node_char.shape[1])))

        n_bands = node_freq.shape[1]
        node_w = species_order = None
        if surface_weights:
            # same bare |e|^2 species projection as legacy, but EXACT at each
            # node (no interpolation to commute with — the sum_s = 1 and
            # character == group-A sum identities hold pointwise).
            node_w, species_order = projection.species_weights_array(
                node_eig, symbols)                           # (Nn,Nb,nS), [sym]
        # dense masked field stacks — the FULL fields are still computed
        # exactly as before (0.6.92 changes STORAGE, never what is evaluated):
        # they are the substrate the symmetry reduction is VERIFIED against.
        z_stack = np.full((n_bands, grid, grid), np.nan)
        c_stack = np.full((n_bands, grid, grid), np.nan)
        z_stack[:, inside] = node_freq.T                     # (C-order matches
        c_stack[:, inside] = node_char.T                     #  gpts scatter)
        z_stack[:, outside] = rim_freq.T                     # rim clamps (0.7.1)
        c_stack[:, outside] = rim_char.T
        w_stack = None
        if surface_weights:
            rim_w, _spo2 = (projection.species_weights_array(rim_eig, symbols)
                            if len(rim_eig) else
                            (np.zeros((0, n_bands, len(species_order))), species_order))
            w_stack = np.full((n_bands, len(species_order), grid, grid), np.nan)
            w_stack[:, :, inside] = np.transpose(node_w, (1, 2, 0))
            w_stack[:, :, outside] = np.transpose(rim_w, (1, 2, 0))
        # every inside node was evaluated — engine-exact has no interpolation
        # gaps; the honest coverage number is 1.0.
        coverage = [float(np.isfinite(z_stack[b][inside]).mean())
                    for b in range(n_bands)]
        surf_bands, surf_node_map, surf_sym, surf_node_idx = _emit_surface_bands(
            np.ones_like(inside), z_stack, c_stack, w_stack)


        # z_range from the band + the ENGINE surface z (not the mesh file)
        zmin = float(min(np.nanmin(b_freq), float(node_freq.min())))
        zmax = float(max(np.nanmax(b_freq), float(node_freq.max())))

        # ---- mesh_modes (0.6.81): exact engine modes on an INDEPENDENT
        # in-BZ grid — Stage 3's click-anywhere store. The viewer will snap
        # to the NEAREST node and animate its EXACT eigenvector (never
        # interpolate an eigenvector across a band crossing). Deliberately
        # NOT the surface --grid: at 96 that would be ~4600 nodes of
        # eigenvectors (far over budget); the default 24x24 is ~430 in-BZ
        # nodes (~0.7 MB at 6 significant figures). Same padded bounding box
        # + BZ mask as the surface, so node coverage matches what is drawn.
        mm = int(mesh_modes_grid)
        mm_kx = np.linspace(xmin - padx, xmax + padx, mm)
        mm_ky = np.linspace(ymin - pady, ymax + pady, mm)
        MMX, MMY = np.meshgrid(mm_kx, mm_ky)                 # (mm, mm), 'xy'
        mm_gpts = np.column_stack([MMX.ravel(), MMY.ravel()])
        mm_nodes = mm_gpts[Path(poly).contains_points(mm_gpts)]   # (Nn, 2)
        mm_qfrac = np.column_stack(
            [mm_nodes, np.zeros(len(mm_nodes))]) @ inv_recip      # (Nn, 3)
        mm_freq_l, mm_eig_l = [], []
        for _k in range(len(mm_qfrac)):
            _f, _v = evaluate_dq(eng_ph, mm_qfrac[_k])
            mm_freq_l.append(_f)
            # canonicalize the global phase PER MODE (the band-egress rule:
            # each (Na,3) block independently) so phi=0 is reproducible for
            # the Stage-3 animator.
            mm_eig_l.append(np.stack([canonicalize_eigenvector_phase(_v[_b])
                                      for _b in range(_v.shape[0])]))
        mm_freq = np.asarray(mm_freq_l)                      # (Nn, Nb)
        mm_eig = np.asarray(mm_eig_l)                        # (Nn, Nb, Na, 3)
        # PRECISION POLICY (the size driver): eigenvector components to
        # 6 SIGNIFICANT figures — rounded AFTER canonicalize, BEFORE the
        # json_v1 wire split. Frequencies / q_frac / q_cart stay FULL
        # precision (small; keeps the frequency invariant tight at 1e-6).
        mm_real, mm_imag = _split_eigvecs(_round_6g(mm_eig))
        mesh_modes_block = {
            "grid": [mm, mm],
            "q_frac": mm_qfrac.tolist(),
            "q_cart": mm_nodes.tolist(),
            "frequency": mm_freq.tolist(),
            "eigenvectors_real": mm_real,
            "eigenvectors_imag": mm_imag,
        }
    elif legacy_surface:
        from scipy.interpolate import LinearNDInterpolator
        from matplotlib.path import Path

        poly = bz.first_brillouin_zone_2d(recip)            # (Nv,2) ordered ring
        qc = (np.asarray(mesh.q_points) @ recip)[:, :2]     # (Nq,2) scattered cloud
        m_freq = np.asarray(mesh.frequencies)               # (Nq, Nb)
        m_char = _character(np.asarray(mesh.eigenvectors), symbols, group_a)  # (Nq,Nb)

        # regular grid over the BZ bounding box (slightly padded), masked to
        # polygon — EXACTLY symmetric axes, same construction as the engine
        # branch (0.6.92; see the comment there).
        xmin, ymin = poly.min(axis=0)
        xmax, ymax = poly.max(axis=0)
        padx = 0.04 * (xmax - xmin)
        pady = 0.04 * (ymax - ymin)
        half_x = max(abs(float(xmin)), abs(float(xmax))) + padx
        half_y = max(abs(float(ymin)), abs(float(ymax))) + pady
        off = (grid - 1) / 2.0
        kx = (np.arange(grid) - off) * (2.0 * half_x / (grid - 1))
        ky = (np.arange(grid) - off) * (2.0 * half_y / (grid - 1))
        KX, KY = np.meshgrid(kx, ky)                         # (grid, grid), 'xy'
        gpts = np.column_stack([KX.ravel(), KY.ravel()])
        inside = Path(poly).contains_points(gpts).reshape(grid, grid)

        n_bands = m_freq.shape[1]
        # E2 (0.6.70) per-species surface weight grids. The mesh per-atom |e|^2 is
        # grouped by SPECIES and normalized so sum_s = 1 per mesh q BEFORE
        # interpolation (bare projection, Togo & Tanaka 2015; NOT mass-weighted).
        # The existing `character` grid is exactly the group-A sum of these same
        # columns, and LinearNDInterpolator is linear over the fixed `qc`
        # triangulation, so interpolation commutes with the species sum — the
        # consistency golden (sum over group-A of the weight grids == character).
        # Every weight grid uses the SAME interpolator + NaN mask as z/character.
        m_weights = species_order = None
        if surface_weights:
            m_weights, species_order = projection.species_weights_array(
                np.asarray(mesh.eigenvectors), symbols)          # (Nq,Nb,nS), [sym]
        # dense masked field stacks, interpolated exactly as before; the
        # symmetry reduction verifies against them (interpolated fields are
        # NOT grid-symmetric in general — the strict z gate then rejects the
        # mirror ops and legacy payloads keep every node explicitly; they
        # still gain the flat/null-free storage + the 6g precision policy).
        nS = len(species_order) if surface_weights else 0
        # SMOOTH RIM (0.7.1): outside nodes are clamped onto the zone
        # boundary and the SAME interpolators are evaluated AT the clamp —
        # no separate resampling scheme. Where the boundary point falls
        # outside the scattered mesh hull the interpolator returns NaN and
        # the hole honestly stays a hole.
        outside = ~inside
        out_clamped = _ray_to_hull(gpts[outside.ravel()], poly)
        z_stack = np.full((n_bands, grid, grid), np.nan)
        c_stack = np.full((n_bands, grid, grid), np.nan)
        w_stack = np.full((n_bands, nS, grid, grid), np.nan) if surface_weights else None
        for b in range(n_bands):
            fz = LinearNDInterpolator(qc, m_freq[:, b])
            fc = LinearNDInterpolator(qc, m_char[:, b])
            z = fz(KX, KY)                                   # (grid,grid)
            c = fc(KX, KY)
            z_stack[b] = np.where(inside, z, np.nan)
            c_stack[b] = np.where(inside, c, np.nan)
            z_stack[b][outside] = fz(out_clamped[:, 0], out_clamped[:, 1])
            c_stack[b][outside] = fc(out_clamped[:, 0], out_clamped[:, 1])
            if surface_weights:
                # one grid per species, ordered by species_order (below); the
                # viewer blends these with the SAME palette as the 2D fatband.
                for s in range(nS):
                    fw = LinearNDInterpolator(qc, m_weights[:, b, s])
                    w = fw(KX, KY)
                    w_stack[b, s] = np.where(inside, w, np.nan)
                    w_stack[b, s][outside] = fw(out_clamped[:, 0], out_clamped[:, 1])
        coverage = [float(np.isfinite(z_stack[b][inside]).mean())
                    for b in range(n_bands)]
        surf_bands, surf_node_map, surf_sym, surf_node_idx = _emit_surface_bands(
            np.ones_like(inside), z_stack, c_stack, w_stack)

        zmin = float(min(np.nanmin(b_freq), np.nanmin(m_freq)))
        zmax = float(max(np.nanmax(b_freq), np.nanmax(m_freq)))
    else:
        # mesh is optional (0.6.15); when given (either dim) the same-material
        # consistency check above still runs (it caught real mistakes before).
        m_freq = np.asarray(mesh.frequencies) if mesh is not None else None

    # ---- DOS (1d-2): OPTIONAL, presence-keyed like the surface. Each curve
    # set keeps its NATIVE frequency grid — the total and projected .dat files
    # are independently sampled, and resampling through van Hove peaks distorts
    # them (median residual ~1%, p95 up to 30% at sharp features) — so the
    # fills and the total line are separate exact curves; no stacking.
    dos_block = {}
    if total_dos is not None:
        arr = _load_dos_dat(total_dos, "total DOS", ncols=2)
        tfreq, tcols = _trim_zero_bins(arr[:, 0], arr[:, 1:2])
        dos_block["total"] = {"frequency": [float(x) for x in tfreq],
                              "dos": [float(x) for x in tcols[:, 0]]}
    if proj_dos is not None:
        arr = _load_dos_dat(proj_dos, "projected DOS", ncols=1 + len(symbols))
        groups = _pdos_groups(symbols, pdos_conf)
        merged = np.column_stack([arr[:, idx].sum(axis=1)
                                  for _, idx in groups])
        pfreq, pcols = _trim_zero_bins(arr[:, 0], merged)
        dos_block["projected"] = {
            "frequency": [float(x) for x in pfreq],
            "curves": [{"label": lab, "dos": [float(x) for x in pcols[:, gi]]}
                       for gi, (lab, _) in enumerate(groups)],
        }

    # conventions + source: built with ORDER-PRESERVING conditional inserts so a
    # dim==2 payload is byte-identical to 0.6.13 except the added dimensionality.
    conventions = {
        "frequency_units": "THz (signed; negative = imaginary mode)",
        "q_cartesian": ("q_frac @ reciprocal_lattice (rows = b_i), xy used"
                        if dim == 2 else
                        "q_frac @ reciprocal_lattice (rows = b_i), full 3-component"),
        "character": (f"fraction of |e|^2 on species {group_a} "
                      "(group-A / total); 0..1, shared band/surface scale"),
        "character_group_a": group_a,
    }
    if engine_surface:
        conventions["surface_construction"] = (
            "exact D(q) evaluation at every grid node: in-BZ nodes at their "
            "own q; outside nodes at their Γ-ray exit point on the zone "
            "boundary (the 0.7.1 smooth rim — the viewer draws those "
            "vertices ON the hexagon, so the rim follows the zone, not the "
            "grid). Engine mode; no interpolation anywhere.")
        conventions["surface_evaluation"] = (
            "D(q) evaluated exactly from FORCE_CONSTANTS via phonopy; "
            "Togo & Tanaka 2015")
        conventions["mesh_modes"] = (
            "exact engine modes on an independent in-BZ grid; eigenvectors "
            "6 significant figures; consumer snaps to nearest node — no "
            "eigenvector interpolation")
        if surface_weights:
            conventions["surface_weights"] = (
                "surface.bands[b].weights[s] = per-species bare |e|^2 projection "
                "computed EXACTLY at each grid node from the engine eigenvectors "
                "(sum_s=1 per node; NOT mass-weighted), same NaN mask as "
                "character; column order = surface.species_order. character == "
                "sum over group-A species (pointwise; nothing to commute).")
    elif legacy_surface:
        conventions["surface_construction"] = (
            "scattered LinearNDInterpolation of the Wigner-Seitz-folded "
            "mesh on Cartesian q; outside nodes interpolated AT their "
            "Γ-ray boundary clamp (0.7.1 smooth rim; NaN where the clamp "
            "leaves the mesh hull — holes stay holes)")
        if surface_weights:
            conventions["surface_weights"] = (
                "surface.bands[b].weights[s] = per-species bare |e|^2 projection "
                "(sum_s=1 per mesh q before interpolation; NOT mass-weighted), "
                "same LinearND grid + NaN mask as character; column order = "
                "surface.species_order. character == sum over group-A species "
                "(interpolation is linear, so it commutes with the species sum).")
    if has_surface:
        # 0.6.92 compact storage — shared by both surface modes.
        conventions["surface_storage"] = (
            "flat per-node arrays + node_map (grid index -> node index, -1 "
            "outside BZ), deduplicated by VERIFIED mirror symmetry: an op is "
            "used only if the computed field itself satisfies it "
            f"(max|dz| <= {_SURF_SYM_ZTOL:g} THz over all in-BZ pairs); "
            "character/weights gate per node "
            f"(<= {_SURF_SYM_CTOL:g}; gauge-sensitive nodes near degeneracies "
            "stay explicitly stored). z (frequencies) FULL precision — the "
            "band-vs-surface single-source invariant keeps its 1e-6 THz "
            "teeth; character/weights 6 significant figures (the 0.6.81 "
            "precision policy for bounded display fractions; verification "
            "ran at full precision before rounding). expand_surface() "
            "reconstructs the dense grids by pure indexing.")
        conventions["bz_method"] = "Wigner-Seitz / Voronoi cell of the reciprocal lattice"
        conventions["ibz_method"] = ("lattice (holohedral) irreducible wedge as the "
                                     "triangle [Γ, M, K] from the band path; NOT a full "
                                     "space-group IBZ (larger for a 3-fold crystal)")
    conventions["displacement"] = ("u = A*Re[e*exp(i(2pi*n.q_frac + phi))]/sqrt(m); "
                                   "Dove 4.16, mass-weighted")
    conventions["bloch_phase"] = ("2*pi * (n_int . q_frac); q fractional, n integer, "
                                  "no metric")
    conventions["eigenvector_phase"] = ("canonicalized (largest component "
                                        "real-positive at phi=0)")
    conventions["eigenvector_layout"] = "[Nq][Nbranch][Natom][3], real/imag split"
    if dos_block:
        conventions["dos"] = ("phonopy total/projected DOS as written (frequencies "
                              "THz); per-atom projected columns merged by symbol or "
                              "pdos.conf groups; all-zero head/tail bins trimmed to "
                              "one margin bin; NO resampling — total and projected "
                              "keep their independently sampled native grids")

    source = {"band": os.path.basename(band_path)}
    if mesh is not None:
        source["mesh"] = os.path.basename(mesh_path)
    if total_dos is not None:
        source["total_dos"] = os.path.basename(total_dos)
    if proj_dos is not None:
        source["projected_dos"] = os.path.basename(proj_dos)
    source["n_qpoints_band"] = int(b_freq.shape[0])
    if mesh is not None:
        # read the mesh's own q-count directly (engine-exact mode no longer
        # binds m_freq — the surface is D(q), not the mesh; 0.6.85). Legacy
        # value is bit-identical (m_freq == mesh.frequencies there).
        source["n_qpoints_mesh"] = int(np.asarray(mesh.frequencies).shape[0])
        source["mesh_dims"] = [int(x) for x in mesh.mesh]
    if has_surface:
        source["surface_grid"] = int(grid)
        source["surface_bz_coverage"] = coverage
        # 0.6.92 storage record: how many grid nodes the payload physically
        # carries after verified-symmetry dedup (== n over nodes_total).
        source["surface_nodes_stored"] = surf_sym["nodes_stored"]
        source["surface_nodes_total"] = surf_sym["nodes_total"]
        source["surface_sym_ops"] = surf_sym["ops"]
    if eng_ph is not None:
        # engine-mode provenance echoes in source (basenames; md5s live in the
        # top-level provenance block below)
        source["phonopy_yaml"] = os.path.basename(engine)
        if eng_info["fc_path"] is not None:
            source["force_constants"] = os.path.basename(eng_info["fc_path"])
        source["engine_kind"] = eng_info["kind"]
        if engine_surface:
            source["mesh_modes_grid"] = int(mesh_modes_grid)

    payload = {
        "meta": {
            "material": {
                "symbols": symbols,
                "masses": [float(x) for x in mat.masses],
                "lattice": np.asarray(mat.lattice).tolist(),
                "reciprocal_lattice": recip.tolist(),
                "fractional_coords": np.asarray(mat.fractional_coords).tolist(),
                # self-describing payloads: absent -> 2 in the viewer, so every
                # pre-0.6.14 payload keeps working unchanged.
                "dimensionality": int(dim),
                # per-species color + physical covalent radius (Å) from the SAME
                # source the Blender add-on uses (radii.jmol_color / covalent_radius,
                # Cordero) — not a hand-written JS palette. The viewer scales the
                # display radius for clarity; the baked value is the physical one.
                "species_style": {
                    sym: {"color": list(jmol_color(sym)),
                          "radius": float(covalent_radius(sym))}
                    for sym in dict.fromkeys(symbols)
                },
            },
            "conventions": conventions,
            "source": source,
        },
        "band": {
            "distance": [float(x) for x in dist],
            "segments": segments,
            "labels": labels,
            "tick_positions": tick_positions,
            "breaks": path_breaks,                # [[e,s], ...] discontinuous junctions (0.6.16)
            "high_symmetry_points": high_symmetry_points,
            "frequency": _to_jsonable(b_freq),
            "character": _to_jsonable(b_char),
            "q_cart": _to_jsonable(b_qcart),
            "q_frac": np.asarray(band.q_points).tolist(),     # (Nq,3) fractional
            "eigenvectors_real": b_eig_real,                  # (Nq,Nb,Na,3)
            "eigenvectors_imag": b_eig_imag,                  # canonicalized per mode
        },
    }
    # ---- 3.1.0: opt-in band connectivity (max-overlap band-following).
    # Computed from the band block's OWN eigenvector arrays and its own
    # segment structure, so the baked table can never disagree with the data
    # beside it. Presence-keyed: absent unless --band-connect; a permutation
    # RELABELLING only — frequencies/eigenvectors are untouched, consumers
    # opt in. Never connects across a path discontinuity (segments restart).
    if band_connect:
        from phonon_core.band_connect import connect_bands as _connect_bands
        _seg_nq = [int(e2 - s2 + 1) for s2, e2 in zip(starts, ends)]
        _perms, _cinfo = _connect_bands(
            np.asarray(b_eig_real), np.asarray(b_eig_imag), segments=_seg_nq)
        payload["band"]["connect"] = {
            "perm": _perms.tolist(),                  # (Nq,Nb) raw index of connected band b
            "min_overlap": float(_cinfo["min_overlap"]),
            "n_reorder": int(_cinfo["n_reorder"]),
            "segment_starts": [int(x) for x in _cinfo["segment_starts"]],
            "convention": "connected band b at point i is raw (frequency-"
                          "sorted) index perm[i][b]; row = identity at each "
                          "segment start; overlaps are gauge-invariant |<e|e>|^2",
        }
        print(f"[pv_navigator_export] band connectivity: "
              f"{_cinfo['n_reorder']} reorder events, "
              f"min matched overlap {_cinfo['min_overlap']:.3f}", file=sys.stderr)

    if dos_block:
        payload["dos"] = dos_block
    if has_surface:
        surf_block = {
            "kx": [float(x) for x in kx],
            "ky": [float(x) for x in ky],
            # 0.6.92 compact storage: bands[b].z/character/weights[s] are FLAT
            # per-stored-node arrays; node_map[i][j] indexes into them (-1 =
            # outside the BZ). expand_surface() (and the viewer's load shim)
            # reconstructs the dense [ky][kx] grids by pure indexing.
            "node_map": surf_node_map,
            "n_nodes": surf_sym["nodes_stored"],
            "sym": {
                "ops": surf_sym["ops"],
                "max_dz_THz": surf_sym["max_dz_THz"],
                "max_dchar": surf_sym["max_dchar"],
                "nodes_total": surf_sym["nodes_total"],
                "nodes_stored": surf_sym["nodes_stored"],
            },
            "bands": surf_bands,
        }
        # E2 (0.6.70): explicit species column order for bands[b].weights[] —
        # additive + presence-keyed (absent -> pre-0.6.70 payload; the current
        # viewer ignores it and colours by `character` as before). Never rely on
        # dict ordering: this list IS the weights[] axis order.
        if surface_weights:
            surf_block["species_order"] = list(species_order)
        payload["surface"] = surf_block
        payload["bz_polygon"] = _to_jsonable(poly)
        payload["ibz_polygon"] = ibz_verts
        payload["z_range"] = [zmin, zmax]
        if eng_ph is not None:
            # presence-keyed, ENGINE-MODE-ONLY (legacy payloads never carry
            # it); inert to the current viewer (named-keys reader) until the
            # Stage-3 wiring.
            payload["mesh_modes"] = mesh_modes_block
    if eng_ph is not None:
        # `provenance` (engine mode ONLY — a legacy payload never carries it, so
        # a PRESENT block is always trustworthy; the strict fail modes above
        # guarantee it). The "engine" sub-dict serializes an EngineProvenance
        # field-for-field in json_v1's 1.1.0 wire shape (built INLINE — json_v1
        # is frozen at e18613db this version; the cross-format consistency test
        # asserts this dict == _engine_block(EngineProvenance(...)) so the two
        # formats cannot drift). The viewer reads named keys only and ignores
        # this block entirely (verified consumer list, 0.6.80).
        provenance = {}
        if engine_surface:
            provenance["surface"] = "engine-exact"
        provenance["engine"] = {
            "kind": eng_info["kind"],
            "evaluated": {"band": True,
                          "mesh": bool(engine_surface)},
            "phonopy_version": eng_info["phonopy_version"],
            "fc_md5": eng_info["fc_md5"],
            "phonopy_yaml_md5": eng_info["phonopy_yaml_md5"],
        }
        provenance["cross_check"] = {"band_max_dw_THz": band_max_dw}
        payload["provenance"] = provenance
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pv_navigator_export", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("band", help="band hdf5 (phonopy)")
    ap.add_argument("mesh", nargs="?", default=None,
                    help="mesh hdf5 (phonopy); optional — without it the "
                         "bands-over-BZ surface block is omitted")
    ap.add_argument("-o", "--output", required=True, help="output JSON path")
    ap.add_argument("--group-a", default="Sn",
                    help="comma-separated species for the character numerator")
    ap.add_argument("--grid", type=int, default=96, help="surface grid per axis")
    ap.add_argument("--dim", type=int, choices=(2, 3), default=2,
                    help="material dimensionality; 3 omits the in-plane "
                         "surface/BZ blocks")
    ap.add_argument("--total-dos", default=None,
                    help="phonopy total_dos.dat (2 cols: freq THz, dos)")
    ap.add_argument("--proj-dos", default=None,
                    help="phonopy projected_dos.dat (1 + Natoms cols)")
    ap.add_argument("--pdos-conf", default=None,
                    help="optional pdos.conf whose PDOS line overrides the "
                         "by-symbol grouping of --proj-dos columns")
    ap.add_argument("--also-html", action="store_true",
                    help="also write index.html + navigator_data.js next to -o")
    ap.add_argument("--engine", default=None, metavar="PHONOPY_YAML",
                    help="OPT-IN engine-backed mode: evaluate band + surface "
                         "EXACTLY from D(q) (Stage-1 A1 engine; FORCE_CONSTANTS "
                         "auto-discovered next to PHONOPY_YAML) and add a "
                         "`provenance` block. Cross-checks the band file and "
                         "REFUSES on mismatch. Requires phonon-core[engine].")
    ap.add_argument("--mesh-modes-grid", type=int, default=24,
                    help="in-BZ grid per axis for the mesh_modes block "
                         "(engine mode with a surface only; independent of "
                         "--grid — eigenvectors are the size driver)")
    ap.add_argument("--band-connect", action="store_true",
                    help="bake a band.connect block: max-overlap band-"
                         "following permutations along the path (relabelling "
                         "only; never crosses a path discontinuity)")
    args = ap.parse_args(argv)
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    for p in (args.band, args.mesh, args.total_dos, args.proj_dos,
              args.pdos_conf, args.engine):
        if p is not None and not os.path.isfile(p):
            _die(f"input not found: {p}", 2)

    group_a = tuple(s.strip() for s in args.group_a.split(",") if s.strip())
    payload = build_payload(args.band, args.mesh, group_a=group_a, grid=args.grid,
                            dim=args.dim, total_dos=args.total_dos,
                            proj_dos=args.proj_dos, pdos_conf=args.pdos_conf,
                            engine=args.engine,
                            mesh_modes_grid=args.mesh_modes_grid,
                            band_connect=args.band_connect)
    if "provenance" in payload:
        _cc = payload["provenance"]["cross_check"]["band_max_dw_THz"]
        print(f"[pv_navigator_export] engine mode ({payload['provenance']['engine']['kind']}): "
              f"band cross-check max|dω| = {_cc:.3e} THz; "
              f"surface = {payload['provenance'].get('surface', 'n/a (no surface block)')}")

    out_parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, allow_nan=False)
    if "surface" in payload:
        nb = len(payload["surface"]["bands"])
        cov = np.mean(payload["meta"]["source"]["surface_bz_coverage"])
        print(f"[pv_navigator_export] wrote {args.output}  "
              f"({payload['meta']['material']['symbols']}, {nb} bands, "
              f"{len(payload['bz_polygon'])}-gon BZ, surface grid "
              f"{args.grid}x{args.grid}, in-BZ coverage {cov:.1%})")
    else:
        nb = len(payload["band"]["frequency"][0])
        why = "dim=3" if args.dim == 3 else "no mesh"
        print(f"[pv_navigator_export] wrote {args.output}  "
              f"({payload['meta']['material']['symbols']}, {nb} bands, {why}: "
              f"no in-plane surface/BZ blocks)")

    if args.also_html:
        data_js = os.path.join(out_parent, "navigator_data.js")
        with open(data_js, "w", encoding="utf-8") as f:
            f.write("window.NAV_DATA = ")
            json.dump(payload, f, allow_nan=False)
            f.write(";\n")
        # 0.7.0: a single-entry materials manifest, so the baked page loads
        # with no missing-file request and the format is demonstrated in
        # every bake. NEVER clobber an existing one — a multi-material
        # bundle's hand-assembled manifest owns the file.
        manifest_js = os.path.join(out_parent, "materials_manifest.js")
        if not os.path.exists(manifest_js):
            from collections import Counter
            counts = Counter(payload["meta"]["material"]["symbols"])
            formula = "".join(f"{el}{(n if n > 1 else '')}"
                              for el, n in counts.items())
            with open(manifest_js, "w", encoding="utf-8") as f:
                f.write("window.NAV_MANIFEST = ")
                json.dump({"materials": [{"slug": "default",
                                          "label": formula,
                                          "file": "navigator_data.js"}]}, f)
                f.write(";\n")
        import shutil
        web_dir = os.path.join(_HERE, os.pardir, "web")
        html_src = os.path.join(web_dir, "index.html")
        plotly_src = os.path.join(web_dir, "plotly.min.js")
        # vendor every local asset the page loads into the output folder so it is
        # fully self-contained (renders offline / from file:// with no network):
        # Plotly (2D/3D panels) + three.js & TrackballControls & kinematics.js (the
        # atom-motion panel). Each is a classic script; none requires a server.
        assets = ["plotly.min.js", "three.min.js", "TrackballControls.js", "kinematics.js"]
        missing = []
        for name in assets:
            src = os.path.join(web_dir, name)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(out_parent, name))
            else:
                missing.append(name)
        if missing:
            print(f"[pv_navigator_export] WARNING: missing web assets {missing}; "
                  f"the page may not render fully offline")
        if os.path.isfile(html_src):
            shutil.copy(html_src, os.path.join(out_parent, "index.html"))
            extra = "" if not missing else f" (MISSING {missing})"
            print(f"[pv_navigator_export] wrote navigator_data.js + "
                  f"{', '.join(a for a in assets if a not in missing)} + index.html"
                  f"{extra} -> open {os.path.join(out_parent, 'index.html')}")
        else:
            print(f"[pv_navigator_export] wrote navigator_data.js (index.html "
                  f"template not found at {html_src})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
