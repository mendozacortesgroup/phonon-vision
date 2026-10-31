"""phonon_vision.hsp — high-symmetry jump targets derived from the MATERIAL.

The engine Mode Navigator's jump row must be a function of the loaded
material, NEVER of session load order (field defect 0.8.1: a bare phonopy.yaml
showed only Γ because the high-symmetry points were remembered only when a
band FILE happened to be loaded earlier in the same Blender session). Two
layered, composable sources — both pure / bpy-free so they are directly
tool-testable, and both feed state.assemble_jump_targets (0.8.1, unchanged)
which dedups by wrapped q:

  SOURCE 1 (authoritative) — discover_sibling_band(): a band file sitting next
    to phonopy.yaml (band.hdf5 then band.yaml, first hit wins), parsed with
    the vendored parsers, accepted only if it is the SAME material (symbols +
    reciprocal lattice). Non-fatal at every step.
  SOURCE 2 (fallback) — lattice_jump_targets(): in-plane high-symmetry points
    derived from the reciprocal lattice alone, pure numpy (Blender ships
    neither scipy nor spglib). CANONICAL for a slab (supercell N×M×1); for an
    orthogonal 3D bulk they are the correct-but-partial k_z=0 subset, labelled
    honestly via lattice_provenance / is_slab_supercell (0.8.3).

Dependencies: numpy (always); the vendored parsers are imported LAZILY inside
discover_sibling_band (bpy-free discipline, mirrors motion.py). No bpy.
"""
from __future__ import annotations

import numpy as np

# 2D Bravais classification tolerances (task-pinned).
_LEN_RTOL = 1e-4       # relative, on |b1| vs |b2|
_ANG_TOL = 0.1         # degrees, on the b1^b2 angle
_ORTHO_RTOL = 1e-4     # relative, for the c*-decoupling (2D) test
_GAMMA = "\u0393"


def same_material(mat_a, mat_b, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Symbols identical AND reciprocal lattices ``np.allclose`` — the
    exporter's same-material idiom (phonon-core 0.6.85). Rejects a sibling
    band file that describes a DIFFERENT crystal."""
    if list(mat_a.symbols) != list(mat_b.symbols):
        return False
    return bool(np.allclose(np.asarray(mat_a.reciprocal_lattice, dtype=float),
                            np.asarray(mat_b.reciprocal_lattice, dtype=float),
                            rtol=rtol, atol=atol))


def _analyze(reciprocal_lattice, dim_hint=None):
    """(is_2d, bravais_name_or_None, points_without_Gamma).

    is_2d is False (→ caller returns []) when the lattice is NOT a decoupled-c*
    2D lattice: an explicit ``dim_hint >= 3``, a bad shape, a degenerate
    vector, or a c* axis COUPLED to the in-plane vectors (a genuine 3D
    lattice, which we refuse to classify — see lattice_jump_targets). Within
    2D, name is None for OBLIQUE (→ caller returns [Γ] only)."""
    if dim_hint is not None and int(dim_hint) >= 3:
        return False, None, []
    R = np.asarray(reciprocal_lattice, dtype=float)
    if R.shape != (3, 3):
        return False, None, []
    b1, b2, b3 = R[0], R[1], R[2]
    L1, L2, L3 = (float(np.linalg.norm(v)) for v in (b1, b2, b3))
    if min(L1, L2, L3) < 1e-12:
        return False, None, []
    # 2D iff c* is DECOUPLED (b3 orthogonal to both in-plane vectors); a
    # coupled c* is a real 3D lattice we will not label.
    if (abs(b3 @ b1) / (L3 * L1) > _ORTHO_RTOL
            or abs(b3 @ b2) / (L3 * L2) > _ORTHO_RTOL):
        return False, None, []
    theta = float(np.degrees(np.arccos(np.clip(b1 @ b2 / (L1 * L2), -1.0, 1.0))))
    len_eq = abs(L1 - L2) / max(L1, L2) < _LEN_RTOL

    def near(a, b):
        return abs(a - b) < _ANG_TOL

    if len_eq and (near(theta, 60.0) or near(theta, 120.0)):
        return True, "hexagonal", [("M", (0.5, 0.0, 0.0)),
                                   ("K", (1.0 / 3.0, 1.0 / 3.0, 0.0))]
    if len_eq and near(theta, 90.0):
        return True, "square", [("X", (0.5, 0.0, 0.0)),
                                ("M", (0.5, 0.5, 0.0))]
    if (not len_eq) and near(theta, 90.0):
        return True, "rectangular", [("X", (0.5, 0.0, 0.0)),
                                     ("Y", (0.0, 0.5, 0.0)),
                                     ("S", (0.5, 0.5, 0.0))]
    if len_eq:
        return True, "centred-rectangular", [("X", (0.5, 0.0, 0.0)),
                                             ("S", (0.5, 0.5, 0.0))]
    return True, None, []          # oblique — Γ only


def lattice_jump_targets(reciprocal_lattice, dim_hint=None, is_slab=None):
    """SOURCE 2: [(label, q_frac), ...] derived from the reciprocal lattice.

    Pure numpy — Blender ships neither scipy nor spglib. The classifier keys on
    L1=|b1|, L2=|b2|, θ=∠(b1,b2) of the IN-PLANE reciprocal vectors when the
    third reciprocal vector is orthogonal to them, per Setyawan & Curtarolo,
    Comput. Mater. Sci. 49, 299 (2010):

        hexagonal    L1≈L2, θ≈60° or 120°  →  Γ, M(½,0,0), K(⅓,⅓,0)
        square       L1≈L2, θ≈90°          →  Γ, X(½,0,0), M(½,½,0)
        rectangular  L1≠L2, θ≈90°          →  Γ, X(½,0,0), Y(0,½,0), S(½,½,0)
        centred-rect L1≈L2, θ otherwise     →  Γ, X(½,0,0), S(½,½,0)  [S approx]
        oblique      none of the above     →  Γ only ([Γ])

    THE CONTRACT (corrected 0.8.3). The 0.8.2 claim "2D-only, 3D → []" was
    FALSE: the c*⟂a*,b* test is satisfied by every ORTHOGONAL 3D family too
    (cubic/tetragonal → "square", orthorhombic → "rectangular", 3D-hexagonal →
    "hexagonal"). The q's are NOT wrong there — they are genuine k_z=0
    high-symmetry points of the 3D zone, correctly labelled — but they are a
    PARTIAL subset. So:
      • SLAB (supercell N×M×1) — these ARE the canonical 2D BZ points and the
        Bravais name IS the crystal's Bravais type;
      • ORTHOGONAL 3D BULK — the SAME points, correct but INCOMPLETE (the
        k_z≠0 points R/Z/A/L/H need spglib, absent in Blender); offering the
        correct partial set beats offering nothing, but it must NOT be sold as
        the whole story — see lattice_provenance / is_slab_supercell;
      • NON-ORTHOGONAL c* (triclinic / monoclinic-ish) — genuinely
        undetectable here → ``[]`` (Γ only).

    ``is_slab`` is accepted for call-site symmetry but does NOT change the
    returned points (they are correct in BOTH regimes); it shapes only the
    provenance (lattice_provenance). The K=(⅓,⅓,0) hexagonal convention is
    VERIFIED against the real SnSe2 phonopy.yaml (K matches its
    band.hdf5-labelled K to 3.3e-7 < 1e-6). Γ is returned FIRST so
    state.assemble_jump_targets's wrapped-q dedup composes with Source 1.

    Returns ``[]`` only for the non-orthogonal-c* case; ``[(Γ, ...)]`` else.
    """
    is_2d, _name, pts = _analyze(reciprocal_lattice, dim_hint)
    if not is_2d:
        return []
    return [(_GAMMA, (0.0, 0.0, 0.0))] + pts


def lattice_bravais_name(reciprocal_lattice, dim_hint=None):
    """The IN-PLANE 2D Bravais name (``"hexagonal"``/``"square"``/
    ``"rectangular"``/``"centred-rectangular"``) or ``None`` (oblique or
    non-orthogonal c*). For a slab this IS the crystal's Bravais type; for an
    orthogonal 3D bulk it is the k_z=0 in-plane type only (do NOT present it as
    the crystal's Bravais type — see lattice_provenance)."""
    return _analyze(reciprocal_lattice, dim_hint)[1]


def read_supercell_matrix(yaml_path):
    """The 3×3 integer ``supercell_matrix`` from a phonopy.yaml, read via the
    vendored ``_yaml_loader`` (never re-parsed by hand). A bare 3-vector
    (diagonal spec) is expanded to a diagonal matrix. Returns None if the key
    is absent / unshaped / the file cannot be read."""
    import os
    try:
        from phonon_core.parsers._yaml_loader import load_yaml
        data = load_yaml(os.fspath(yaml_path))
    except Exception as e:
        print(f"[phonon-vision] supercell_matrix unreadable "
              f"({type(e).__name__}: {e})")
        return None
    sm = data.get("supercell_matrix") if isinstance(data, dict) else None
    if sm is None:
        return None
    arr = np.asarray(sm)
    if arr.shape == (3,):
        arr = np.diag(arr)
    if arr.shape != (3, 3):
        return None
    try:
        return np.rint(arr.astype(float)).astype(int)
    except (ValueError, TypeError):
        return None


def is_slab_supercell(yaml_path):
    """True iff the phonopy supercell does NOT repeat along the third
    (out-of-plane, axis-2) direction — a 2D SLAB calculation (N×M×1); False if
    it repeats along c (3D bulk); None if ``supercell_matrix`` is unreadable.

    EXACT RULE: the THIRD ROW of ``supercell_matrix`` equals ``[0, 0, 1]``.
    phonopy builds the supercell as ``A_super = S · A_prim`` (lattice vectors
    as rows), so ``A_super[2] == A_prim[2]`` — no replication and no shear
    along c — exactly when ``S[2] == [0, 0, 1]``. This generalises the common
    diagonal ``[N, M, 1]`` case and also accepts in-plane shear (off-diagonals
    in the top-left 2×2). It is consistent with the c*=b3 decoupling the
    lattice classifier already assumes (axis 2 is the out-of-plane axis).
    Callers treat None as the conservative 'not a slab' (honest) branch."""
    sm = read_supercell_matrix(yaml_path)
    if sm is None:
        return None
    return bool(np.array_equal(sm[2], np.array([0, 0, 1])))


def lattice_provenance(reciprocal_lattice, is_slab=None):
    """The Data-panel provenance string for the lattice route — HONEST about
    slab vs orthogonal-3D bulk (0.8.3):

      • a classified in-plane lattice + ``is_slab is True`` →
        ``"lattice symmetry (<name>)"`` (canonical: these ARE the 2D BZ's
        high-symmetry points and <name> IS the crystal's Bravais type);
      • a classified lattice + ``is_slab`` False OR None → the honest partial
        label ``"in-plane (k_z=0) symmetry — 3D zone; k_z≠0 points need
        spglib"`` — NO 2D Bravais type is claimed for a 3D crystal;
      • oblique / non-orthogonal c* (no in-plane points) → ``"Γ only"``.

    ``is_slab`` None is treated as False (the conservative / honest branch)."""
    _is_2d, name, pts = _analyze(reciprocal_lattice)
    if not pts:
        return "\u0393 only"
    if is_slab:
        return f"lattice symmetry ({name})"
    return ("in-plane (k_z=0) symmetry \u2014 3D zone; "
            "k_z\u22600 points need spglib")


def discover_sibling_band(yaml_path, engine_material):
    """SOURCE 1: an authoritative sibling band file in phonopy.yaml's dir.

    Returns ``(BandPath, basename)`` on the FIRST sibling that PARSES and
    passes the same-material guard; ``(None, None)`` otherwise. Discovery
    order ``band.hdf5`` then ``band.yaml`` (first hit wins) mirrors the
    engine's own sibling-FORCE_CONSTANTS idiom (same directory). NON-FATAL at
    every step — absent, unparseable (h5py may be missing for the hdf5), or a
    DIFFERENT material: log once and fall through, so a bad sibling can never
    block an engine load that already succeeded.
    """
    import os
    here = os.path.dirname(os.path.abspath(os.fspath(yaml_path)))
    for fname in ("band.hdf5", "band.yaml"):
        cand = os.path.join(here, fname)
        if not os.path.isfile(cand):
            continue
        try:
            from phonon_core import load_band_hdf5, load_band_yaml
            loader = load_band_hdf5 if fname.endswith(".hdf5") else load_band_yaml
            band = loader(cand)
        except Exception as e:      # h5py absent, parse failure, ...
            print(f"[phonon-vision] sibling {fname} unusable "
                  f"({type(e).__name__}: {e}); ignoring")
            continue
        try:
            if not same_material(band.material, engine_material):
                print(f"[phonon-vision] sibling {fname} is a DIFFERENT "
                      "material; ignoring (jump targets from lattice instead)")
                continue
        except Exception as e:
            print(f"[phonon-vision] sibling {fname} material check failed "
                  f"({type(e).__name__}: {e}); ignoring")
            continue
        return band, fname
    return None, None
