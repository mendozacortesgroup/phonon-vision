"""phonon_vision.bonds — bpy-free bond criterion + navigator side-car parsing.

Single source of truth for WHICH atom pairs bond, shared by the strict and
halo(PBC-image) paths, and testable in plain Python (no bpy, no phonon_core —
the covalent-radius function is injected). scene_setup imports this; the
0.7.8 in-scene `_build_bond_list` moved here verbatim and grew per-pair
absolute cutoffs.

Side-car contract (BINDING, docs/bonds_sidecar.md, "phonon-vision-bonds" v1,
shipped by the web navigator since core 0.6.69):

- Pair keys: species pairs, CANONICAL = lexicographic sort, ASCII hyphen
  ("Se-Sn" never "Sn-Se"); en-dash tolerated on read. Index keys forbidden.
- ``max``: ABSOLUTE cutoff in Angstrom — never multiplied by covalent radii
  or any tolerance. Clamped to [0.1, 10] on read (the navigator's clamp).
- REPLACE semantics: a loaded spec is the COMPLETE bond definition. Pairs
  absent from bonds{} form NO bonds; the global tolerance criterion is
  fully bypassed while a spec is loaded.
- A pair entry WITHOUT ``max`` falls back to the covalent default for that
  pair, at the contract constant DEFAULT_TOL = 1.2 (the navigator's fixed
  BOND_TOL) — NOT the add-on's live bond_tolerance, so the file's meaning
  never depends on tool internals. (The navigator always writes ``max`` in
  practice; this is contract completeness.)
- ``radius_A`` (0.6.76+): AUTHORITATIVE absolute bond-cylinder radius in Å,
  clamped [0.02, 1.0] on read (the navigator's import clamp). When present
  it fully determines the drawn radius — the add-on's ``bond_thickness``
  is bypassed for that pair (1 Blender unit ≡ 1 Å, so the Å value is used
  verbatim), the same way a loaded spec bypasses ``bond_tolerance`` for
  detection. Precedence mirrors the navigator: radius_A (finite) wins ->
  else radius_scale -> else default (see ``resolve_radius``).
- ``radius_scale``: LEGACY relative display multiplier, clamped [0.3, 3.0],
  default 1.0. Used only when ``radius_A`` is absent.
- ``color`` (0.7.12): per-pair curated bond color — a validated ``#rrggbb``
  DISPLAY-sRGB hex (the navigator's custom-color mode; regex
  ``^#[0-9a-fA-F]{6}$``, malformed dropped == absent). Applied as a SOLID
  cylinder color that overrides the per-species two-color split for that pair
  only (the navigator's ``bondColorMode[key] == "custom"`` branch). Blender's
  material Base Color is LINEAR, so the sRGB hex is linearized via the standard
  sRGB EOTF (``hex_to_linear``). See ``pair_colors`` / ``bond_material_plan``.
- Loose structure binding: formula/natoms mismatch -> warning, apply anyway.
  Rejection (state untouched): unparseable JSON, non-object, wrong
  ``format``, missing/non-object ``bonds``. Unknown keys ignored.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np

FORMAT_NAME = "phonon-vision-bonds"
DEFAULT_TOL = 1.2          # the navigator's fixed BOND_TOL (contract constant)
MAX_CLAMP = (0.1, 10.0)    # Å, the navigator's import clamp
SCALE_CLAMP = (0.3, 3.0)
RADIUS_A_CLAMP = (0.02, 1.0)   # Å, the navigator's radius_A import clamp

_PAIR_SPLIT = re.compile("[-–]")   # ASCII hyphen or en-dash
_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")   # the navigator's custom-color regex


class SidecarError(ValueError):
    """A side-car file that must be REJECTED (state untouched)."""


def canonicalize_pair(a: str, b: str) -> str:
    """Two species symbols -> the canonical pair key: lexicographic sort,
    ASCII hyphen. canonicalize_pair("Sn","Se") == "Se-Sn"."""
    return "-".join(sorted((a, b)))


def parse_sidecar(text: str):
    """Parse a "phonon-vision-bonds" v1 side-car.

    Returns ``(spec, warnings)`` where ``spec`` is the normalized dict (same
    side-car shape, pair keys canonicalized, values clamped) and ``warnings``
    is a list of human-readable strings (malformed entries skipped, etc.).
    Raises SidecarError on the contract's rejection cases.
    """
    try:
        obj = json.loads(text)
    except Exception as e:
        raise SidecarError(f"unparseable JSON: {e}")
    if not isinstance(obj, dict):
        raise SidecarError("not a JSON object")
    if obj.get("format") != FORMAT_NAME:
        raise SidecarError(f'format is {obj.get("format")!r}, '
                           f'expected "{FORMAT_NAME}"')
    raw_bonds = obj.get("bonds")
    if not isinstance(raw_bonds, dict):
        raise SidecarError("missing bonds{} table")

    warnings: list[str] = []
    pairs: dict = {}
    for key, entry in raw_bonds.items():   # insertion order: last one wins
        parts = [p for p in _PAIR_SPLIT.split(str(key)) if p]
        if len(parts) != 2:
            warnings.append(f"skipped malformed pair key {key!r}")
            continue
        canon = canonicalize_pair(parts[0], parts[1])
        if not isinstance(entry, dict):
            warnings.append(f"skipped pair {canon!r}: entry is not an object")
            continue
        norm: dict = {}
        if "max" in entry:
            try:
                lo, hi = MAX_CLAMP
                norm["max"] = min(hi, max(lo, float(entry["max"])))
            except (TypeError, ValueError):
                warnings.append(f"pair {canon!r}: non-numeric max ignored "
                                f"(covalent default applies)")
        if "radius_A" in entry:
            try:
                lo, hi = RADIUS_A_CLAMP
                norm["radius_A"] = min(hi, max(lo, float(entry["radius_A"])))
            except (TypeError, ValueError):
                warnings.append(f"pair {canon!r}: non-numeric radius_A ignored")
        if "radius_scale" in entry:
            try:
                lo, hi = SCALE_CLAMP
                norm["radius_scale"] = min(hi, max(lo, float(entry["radius_scale"])))
            except (TypeError, ValueError):
                warnings.append(f"pair {canon!r}: non-numeric radius_scale ignored")
        if "color" in entry:
            c = entry["color"]
            if isinstance(c, str) and _HEX_COLOR.match(c):
                norm["color"] = c          # validated #rrggbb display-sRGB hex
            else:                          # mirror the navigator: malformed -> absent
                warnings.append(f"pair {canon!r}: malformed color {c!r} ignored")
        pairs[canon] = norm                    # unknown per-pair keys dropped

    material = obj.get("material") if isinstance(obj.get("material"), dict) else {}
    spec = {
        "format": FORMAT_NAME,
        "version": obj.get("version", 1),
        "generator": obj.get("generator", ""),
        "material": {"formula": material.get("formula"),
                     "natoms": material.get("natoms")},
        "units": "angstrom",
        "bonds": pairs,
    }
    return spec, warnings


def structure_warnings(spec: dict, formula, natoms) -> list[str]:
    """Loose structure binding: mismatches WARN, never reject."""
    out: list[str] = []
    m = spec.get("material") or {}
    if m.get("formula") and formula and m["formula"] != formula:
        out.append(f'side-car material {m["formula"]!r} != loaded {formula!r} '
                   f"— applying anyway")
    if m.get("natoms") is not None and natoms is not None \
            and int(m["natoms"]) != int(natoms):
        out.append(f'side-car natoms {m["natoms"]} != loaded {natoms} '
                   f"— applying anyway")
    return out


def spec_from_json(text):
    """Stored-state helper: ``props.bonds_spec_json`` -> spec dict or None.
    Tolerant (returns None on anything invalid) — state corruption must
    degrade to default bonding, never crash a scene build."""
    if not text:
        return None
    try:
        spec, _warnings = parse_sidecar(text)
        return spec
    except SidecarError:
        return None


def pair_scales(spec) -> dict:
    """Per-pair display radius_scale map (missing -> 1.0 at the call site)."""
    if not spec:
        return {}
    return {k: float(v.get("radius_scale", 1.0))
            for k, v in spec["bonds"].items()}


def pair_radii_A(spec) -> dict:
    """Per-pair ABSOLUTE cylinder radius in Å (side-car ``radius_A``,
    navigator 0.6.76+). Only pairs that carry the key appear; values were
    clamped to RADIUS_A_CLAMP by parse_sidecar, so every entry is > 0."""
    if not spec:
        return {}
    return {k: float(v["radius_A"]) for k, v in spec["bonds"].items()
            if "radius_A" in v}


def pair_colors(spec) -> dict:
    """Per-pair curated bond color as a validated ``#rrggbb`` display-sRGB hex
    (navigator custom-color mode). Only pairs that carry a valid color appear;
    parse_sidecar has already dropped malformed values, so every entry matches
    ``_HEX_COLOR``. Absent == the pair keeps the per-species two-color split."""
    if not spec:
        return {}
    return {k: v["color"] for k, v in spec["bonds"].items() if "color" in v}


def resolve_radius(radius_A, radius_scale, base_radius) -> float:
    """Final bond-cylinder RADIUS in Blender units (1 BU ≡ 1 Å).

    Mirrors the navigator's 0.6.76 import precedence:

    - ``radius_A`` finite -> clamp(radius_A, 0.02, 1.0), ABSOLUTE Å used
      verbatim. ``base_radius`` (which folds in the global bond_thickness)
      is FULLY BYPASSED — the analogue of a loaded spec bypassing
      bond_tolerance for detection.
    - else ``radius_scale`` finite -> base_radius × radius_scale (the
      legacy 0.7.9 path; base_radius = bond_thickness × 0.06).
    - else -> base_radius.
    """
    if radius_A is not None and math.isfinite(radius_A):
        lo, hi = RADIUS_A_CLAMP
        return min(hi, max(lo, float(radius_A)))
    if radius_scale is not None and math.isfinite(radius_scale):
        return float(base_radius) * float(radius_scale)
    return float(base_radius)


def all_pairs_have_radius_A(spec) -> bool:
    """True iff the spec curates >= 1 pair and EVERY pair carries radius_A.
    Under replace semantics the global bond_thickness then drives nothing
    drawn — the UI greys the slider on this predicate (any navigator
    >= 0.6.76 export satisfies it: radius_A is emitted for every pair)."""
    if not spec or not spec["bonds"]:
        return False
    return all("radius_A" in e for e in spec["bonds"].values())


def _srgb_channel_to_linear(c: float) -> float:
    """Standard sRGB EOTF (IEC 61966-2-1): a display-sRGB channel in [0, 1] ->
    scene-linear. Blender material Base Color / diffuse_color are LINEAR, while
    the navigator emits a display-sRGB hex, so a curated color must be
    linearized through THIS transfer function to match perceptually across the
    two tools (the analogue, for color, of ``radius_A`` matching by Å value)."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_to_linear(hex_str: str) -> tuple[float, float, float]:
    """A ``#rrggbb`` display-sRGB hex -> a LINEAR RGB triplet for Blender.

    Assumes a validated 6-hex string (parse_sidecar guarantees it via
    ``_HEX_COLOR``). Each 8-bit channel is normalized to [0, 1] then passed
    through the standard sRGB EOTF (``_srgb_channel_to_linear``)."""
    h = hex_str.lstrip("#")
    srgb = (int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)
    return tuple(_srgb_channel_to_linear(c) for c in srgb)


def bond_material_plan(color, two_color: bool, same_species: bool):
    """Per-bond cylinder/coloring plan — the navigator's per-pair color branch
    (``recolorBonds`` in web/index.html) mirrored for Blender's slot geometry.

    A curated ``color`` paints the WHOLE cylinder that hex, overriding the
    per-species split FOR THAT PAIR ONLY (custom mode wins, both halves), the
    same mutual exclusion the navigator applies (``bondColorMode[key]`` is
    ``"custom"`` XOR absent). Otherwise the 0.4.0 behavior is unchanged.

    Returns ``(n_cyl, style)``:

    - ``color`` present    -> ``(1, "custom")``   one solid curated cylinder
    - ``two_color`` off    -> ``(1, "grey")``     one grey cylinder
    - same species         -> ``(1, "species")``  one species-colored cylinder
    - different species    -> ``(2, "split")``    two per-species half-cylinders

    ``color`` is None or a validated hex; ``two_color`` is the global toggle;
    ``same_species`` is ``sp_i == sp_j``. Pure decision logic (no bpy) so the
    override precedence is pinned in tools/test_bonds.py; scene_setup maps each
    ``style`` to an actual material slot."""
    if color is not None:
        return 1, "custom"
    if not two_color:
        return 1, "grey"
    if same_species:
        return 1, "species"
    return 2, "split"


def _cutoff_matrix(symbols_per_slot, tolerance, spec, radius_fn):
    """(species_index_per_slot, S×S cutoff matrix in Å).

    spec=None  -> cut(a,b) = (r_a + r_b) * tolerance  — the 0.7.8 criterion,
                  same float ops in the same order (regression-golden pinned).
    spec given -> REPLACE semantics: in-spec pairs use the absolute ``max``
                  (or the DEFAULT_TOL covalent default when max is absent);
                  absent pairs get -inf (never bond).
    """
    species = sorted(set(symbols_per_slot))
    s_index = {s: i for i, s in enumerate(species)}
    si = np.array([s_index[s] for s in symbols_per_slot], dtype=np.int64)

    S = len(species)
    cut = np.empty((S, S), dtype=np.float64)
    for a_i, a in enumerate(species):
        for b_i, b in enumerate(species):
            if spec is None:
                cut[a_i, b_i] = (radius_fn(a) + radius_fn(b)) * tolerance
            else:
                entry = spec["bonds"].get(canonicalize_pair(a, b))
                if entry is None:
                    cut[a_i, b_i] = -np.inf          # pair absent -> NO bonds
                elif entry.get("max") is not None:
                    cut[a_i, b_i] = float(entry["max"])   # ABSOLUTE Å, verbatim
                else:
                    cut[a_i, b_i] = (radius_fn(a) + radius_fn(b)) * DEFAULT_TOL
    return si, cut


def build_bond_list(positions: np.ndarray, symbols_per_slot: list[str],
                    tolerance: float,
                    lattice: np.ndarray | None = None,
                    supercell: tuple[int, int, int] = (1, 1, 1),
                    include_images: bool = True,
                    spec: dict | None = None,
                    radius_fn=None,
                    ) -> list[tuple[int, int, tuple[float, float, float]]]:
    """Find atom pairs that should be bonded, optionally including periodic images.

    Moved verbatim from scene_setup (0.7.8) and extended with ``spec``.
    A bond is recorded as ``(i, j, image_offset_xyz)`` where ``image_offset``
    is the lattice translation to add to atom j's position to get the bonded
    partner location ((0,0,0) for internal bonds). Each unordered pair is
    recorded at most once per image direction.

    Parameters
    ----------
    positions : (N, 3) ndarray of equilibrium positions in Å. The criterion
        compares in Å — callers must not pre-scale.
    symbols_per_slot : list of N chemical symbols.
    tolerance : multiplier on (R_i + R_j) — BYPASSED while ``spec`` is loaded.
    lattice, supercell, include_images : PBC-image search (halo path); image
        bonds inherit the parent pair's cutoff (same matrix, both branches).
    spec : parsed side-car (see parse_sidecar) or None for default bonding.
    radius_fn : symbol -> covalent radius in Å. Default: phonon_core's
        Cordero table (imported lazily so plain-python tests can inject).
    """
    if radius_fn is None:
        from phonon_core.radii import covalent_radius as radius_fn

    n = len(positions)
    si, cut = _cutoff_matrix(symbols_per_slot, tolerance, spec, radius_fn)

    pairs: list[tuple[int, int, tuple[float, float, float]]] = []

    if not include_images or lattice is None:
        # Fast path: same as before
        for i in range(n):
            d = positions[i + 1:] - positions[i]
            dist = np.linalg.norm(d, axis=1)
            cutoff = cut[si[i], si[i + 1:]]
            mask = (dist > 0.5) & (dist < cutoff)
            for jrel, ok in enumerate(mask):
                if ok:
                    pairs.append((i, i + 1 + jrel, (0.0, 0.0, 0.0)))
        return pairs

    # PBC-aware: for each pair, search over supercell-image translations
    nx, ny, nz = supercell
    sc_a = nx * lattice[0]
    sc_b = ny * lattice[1]
    sc_c = nz * lattice[2]

    # 27 supercell images: offsets that translate atom j by 0 or ±one supercell
    image_offsets = []
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for iz in (-1, 0, 1):
                image_offsets.append(ix * sc_a + iy * sc_b + iz * sc_c)
    image_offsets = np.array(image_offsets)   # (27, 3)

    # Track which (i, j, image) triples have been added (avoid double-counting)
    seen = set()

    for i in range(n):
        r_i = positions[i]
        for j in range(i, n):
            r_j = positions[j]
            r_diffs = (r_j + image_offsets) - r_i        # (27, 3)
            dists = np.linalg.norm(r_diffs, axis=1)
            cutoff = cut[si[i], si[j]]
            # For self-pair (i == j), exclude the zero-image (would be distance 0)
            for k, dist in enumerate(dists):
                if dist < 0.5 or dist >= cutoff:
                    continue
                ox, oy, oz = image_offsets[k]
                # Canonicalize the pair direction to avoid (i,j,+T) and (j,i,-T) duplicates
                if i < j:
                    key = (i, j, round(ox, 4), round(oy, 4), round(oz, 4))
                elif i > j:
                    # Re-cast as (j, i, -offset) for uniqueness
                    key = (j, i, round(-ox, 4), round(-oy, 4), round(-oz, 4))
                else:  # i == j, self-image (only across supercell boundary)
                    # Pick a canonical sign for the offset
                    sig = (ox, oy, oz)
                    neg = (-ox, -oy, -oz)
                    if sig < neg:
                        continue   # use the opposite-sign version
                    key = (i, j, round(ox, 4), round(oy, 4), round(oz, 4))
                if key in seen:
                    continue
                seen.add(key)
                ii, jj = (i, j) if (i, j) == (key[0], key[1]) else (j, i)
                # The offset stored is for atom jj (so partner = r_jj + offset)
                if (ii, jj) == (i, j):
                    pairs.append((i, j, (float(ox), float(oy), float(oz))))
                else:
                    pairs.append((j, i, (float(-ox), float(-oy), float(-oz))))
    return pairs
