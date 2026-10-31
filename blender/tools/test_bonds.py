#!/usr/bin/env python3
"""Regression + contract tests for bonds.py (0.7.9/0.7.10) — plain Python, no bpy.

bonds.py is bpy-free and takes the covalent-radius function as an argument,
so the whole criterion (strict + halo/PBC-image paths) and the side-car
contract run under plain python:

    python tools/test_bonds.py

The REGRESSION GOLDENS were captured from the PRE-CHANGE 0.7.8
scene_setup._build_bond_list (verbatim extraction, run with the vendored
phonon_core Cordero radii: r_Sn = 1.39 Å, r_Se = 1.20 Å) on the synthetic
structure below. spec=None must reproduce them exactly, forever.

0.7.10 adds the radius_A display-geometry suite: absolute-Å cylinder radius
(cross-tool congruence with navigator 0.6.76), precedence over the legacy
radius_scale, the bond_thickness bypass, the radius-not-diameter unit guard
(against the REAL mesh builder, loaded with stubbed bpy — the house pattern
from test_job_roundtrip), and 0.7.9 resolved-radius goldens.
"""
import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_bonds_module():
    spec = importlib.util.spec_from_file_location(
        "pv_bonds", os.path.join(_HERE, os.pardir, "bonds.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load_bonds_module()

# Cordero radii as vendored (phonon_core.radii) at golden-capture time.
_RADII = {"Sn": 1.39, "Se": 1.2}


def radius_fn(sym):
    return _RADII[sym]


# Synthetic structure: 6 atoms, Å. Distances of note:
#   0-1 Sn-Se 2.5 | 0-3 Sn-Sn 2.7 | 0-4 Sn-Se 3.0 (near-cutoff: 3.108)
#   1-2 Se-Se 2.7 | 2-3 Se-Sn 2.5 | 5 bonds nothing internally;
#   5-0' Sn-Sn 2.0 across the (8,8,8) boundary (image bond).
POS = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [2.5, 2.7, 0.0],
                [0.0, 2.7, 0.0], [0.0, 0.0, 3.0], [6.0, 0.0, 0.0]])
SYM = ["Sn", "Se", "Se", "Sn", "Se", "Sn"]
LAT = np.diag([8.0, 8.0, 8.0])
TOL = 1.2

# Captured from the pre-change code path (see module docstring).
STRICT_GOLDEN = [[0, 1, [0.0, 0.0, 0.0]], [0, 3, [0.0, 0.0, 0.0]],
                 [0, 4, [0.0, 0.0, 0.0]], [1, 2, [0.0, 0.0, 0.0]],
                 [2, 3, [0.0, 0.0, 0.0]]]
HALO_GOLDEN = [[0, 1, [0.0, 0.0, 0.0]], [0, 3, [0.0, 0.0, 0.0]],
               [0, 4, [0.0, 0.0, 0.0]], [0, 5, [-8.0, 0.0, 0.0]],
               [1, 2, [0.0, 0.0, 0.0]], [2, 3, [0.0, 0.0, 0.0]]]


def _as_lists(pairs):
    return [[p[0], p[1], list(p[2])] for p in pairs]


def _sidecar(bonds_obj, **top):
    doc = {"format": "phonon-vision-bonds", "version": 1,
           "generator": "test", "units": "angstrom", "bonds": bonds_obj}
    doc.update(top)
    return json.dumps(doc)


def _bond_list(spec, **kw):
    args = dict(tolerance=TOL, spec=spec, radius_fn=radius_fn,
                include_images=False)
    args.update(kw)
    return B.build_bond_list(POS, SYM, **args)


# --- REGRESSION GOLDEN ------------------------------------------------------
def test_golden_strict_spec_none():
    assert _as_lists(_bond_list(None)) == STRICT_GOLDEN


def test_golden_halo_spec_none():
    got = _bond_list(None, lattice=LAT, supercell=(1, 1, 1),
                     include_images=True)
    assert _as_lists(got) == HALO_GOLDEN


# --- REPLACE SEMANTICS ------------------------------------------------------
def test_replace_semantics_absent_pair_forms_no_bonds():
    # Cordero WOULD bond Se-Se (d12 = 2.7 < 2.88). The spec omits Se-Se
    # (and Sn-Sn) -> those pairs form NO bonds; only Se-Sn survives.
    spec, _ = B.parse_sidecar(_sidecar({"Se-Sn": {"max": 3.11}}))
    got = _as_lists(_bond_list(spec))
    assert got == [[0, 1, [0.0, 0.0, 0.0]], [0, 4, [0.0, 0.0, 0.0]],
                   [2, 3, [0.0, 0.0, 0.0]]], got
    # explicitly: zero Se-Se and zero Sn-Sn bonds
    for i, j, _o in got:
        assert {SYM[i], SYM[j]} == {"Se", "Sn"}


def test_spec_bypasses_global_tolerance():
    # With a spec loaded the passed tolerance must be COMPLETELY inert:
    # an absurd tolerance changes nothing.
    spec, _ = B.parse_sidecar(_sidecar({"Se-Sn": {"max": 3.11}}))
    a = _as_lists(B.build_bond_list(POS, SYM, tolerance=TOL, spec=spec,
                                    radius_fn=radius_fn, include_images=False))
    b = _as_lists(B.build_bond_list(POS, SYM, tolerance=5.0, spec=spec,
                                    radius_fn=radius_fn, include_images=False))
    assert a == b


# --- CANONICALIZATION -------------------------------------------------------
def test_canonicalization_key_order_and_endash():
    texts = [_sidecar({"Se-Sn": {"max": 3.11}}),
             _sidecar({"Sn-Se": {"max": 3.11}}),
             _sidecar({"Se–Sn": {"max": 3.11}})]   # en-dash
    specs = [B.parse_sidecar(t)[0] for t in texts]
    assert all(list(s["bonds"]) == ["Se-Sn"] for s in specs), \
        [list(s["bonds"]) for s in specs]
    lists = [_as_lists(_bond_list(s)) for s in specs]
    assert lists[0] == lists[1] == lists[2]


# --- ABSOLUTE-Å PIN ---------------------------------------------------------
def test_absolute_angstrom_cutoff_pin():
    # d(0,4) = 3.000 Å exactly. max just below vs just above must toggle
    # EXACTLY that bond — catches any radius/tolerance/unit scaling of max.
    below, _ = B.parse_sidecar(_sidecar({"Se-Sn": {"max": 2.99}}))
    above, _ = B.parse_sidecar(_sidecar({"Se-Sn": {"max": 3.01}}))
    got_b = _as_lists(_bond_list(below))
    got_a = _as_lists(_bond_list(above))
    assert [0, 4, [0.0, 0.0, 0.0]] not in got_b
    assert [0, 4, [0.0, 0.0, 0.0]] in got_a
    assert [x for x in got_a if x != [0, 4, [0.0, 0.0, 0.0]]] == got_b


def test_pair_without_max_gets_contract_default():
    # An in-spec pair with NO max falls back to (r_a+r_b) × DEFAULT_TOL(1.2)
    # = 3.108 for Se-Sn -> d(0,4)=3.0 bonds. (Contract constant, NOT the
    # add-on's live tolerance.)
    spec, _ = B.parse_sidecar(_sidecar({"Se-Sn": {}}))
    got = _as_lists(_bond_list(spec, tolerance=0.1))   # live tol inert
    assert [0, 4, [0.0, 0.0, 0.0]] in got


# --- HALO INHERITANCE -------------------------------------------------------
def test_halo_image_bond_respects_pair_cutoff():
    # The Sn-Sn image bond (0,5,(-8,0,0)) sits at d = 2.0 Å.
    keep, _ = B.parse_sidecar(_sidecar({"Sn-Sn": {"max": 2.5}}))
    drop, _ = B.parse_sidecar(_sidecar({"Sn-Sn": {"max": 1.9}}))
    got_keep = _as_lists(_bond_list(keep, lattice=LAT, supercell=(1, 1, 1),
                                    include_images=True))
    got_drop = _as_lists(_bond_list(drop, lattice=LAT, supercell=(1, 1, 1),
                                    include_images=True))
    assert [0, 5, [-8.0, 0.0, 0.0]] in got_keep
    assert [0, 5, [-8.0, 0.0, 0.0]] not in got_drop
    # replace semantics hold in the image path too: every non-Sn-Sn pair is
    # absent from the spec -> no other bonds anywhere; and the INTERNAL
    # Sn-Sn bond (0,3) at d = 2.7 > max 2.5 is dropped by the same cutoff
    # the image bond obeys (one matrix, both paths).
    assert got_keep == [[0, 5, [-8.0, 0.0, 0.0]]], got_keep


# --- PARSING: rejection / warnings / clamps ---------------------------------
def test_rejections_raise():
    for bad in ["not json {", json.dumps(["a", "list"]),
                json.dumps({"format": "something-else", "bonds": {}}),
                json.dumps({"format": "phonon-vision-bonds"})]:
        try:
            B.parse_sidecar(bad)
        except B.SidecarError:
            continue
        raise AssertionError(f"not rejected: {bad[:40]}")


def test_structure_mismatch_warns_but_applies():
    spec, w = B.parse_sidecar(_sidecar({"Se-Sn": {"max": 3.0}},
                                       material={"formula": "SnSeSe",
                                                 "natoms": 3}))
    assert w == []
    warns = B.structure_warnings(spec, "WSe2", 3)
    assert len(warns) == 1 and "applying anyway" in warns[0]
    warns2 = B.structure_warnings(spec, "SnSeSe", 4)
    assert len(warns2) == 1 and "natoms" in warns2[0]
    assert B.structure_warnings(spec, "SnSeSe", 3) == []


def test_clamps_and_stored_color():
    spec, _ = B.parse_sidecar(_sidecar(
        {"Se-Sn": {"max": 99.0, "radius_scale": 0.05, "color": "#aa33cc"},
         "Se-Se": {"max": 0.01, "radius_scale": 12.0}}))
    e = spec["bonds"]["Se-Sn"]
    assert e["max"] == 10.0 and e["radius_scale"] == 0.3
    assert e["color"] == "#aa33cc"          # stored (and applied at 0.7.12)
    e2 = spec["bonds"]["Se-Se"]
    assert e2["max"] == 0.1 and e2["radius_scale"] == 3.0
    assert B.pair_scales(spec) == {"Se-Sn": 0.3, "Se-Se": 3.0}


def test_spec_from_json_tolerant():
    assert B.spec_from_json("") is None
    assert B.spec_from_json("garbage {") is None
    spec, _ = B.parse_sidecar(_sidecar({"Se-Sn": {"max": 3.0}}))
    spec["source"] = {"filename": "x.json", "imported": "2026-07-03"}
    back = B.spec_from_json(json.dumps(spec))
    assert back is not None and back["bonds"] == spec["bonds"]


# --- 0.7.10 radius_A: absolute-Å cylinder radius (cross-tool congruence) ----
def _load_scene_setup_module():
    """Load ../scene_setup.py with stubbed bpy/mathutils/phonon_core so its
    numpy-pure mesh builder is testable in plain python (the house pattern
    from test_job_roundtrip). `from . import bonds` resolves to the module
    under test, so the guard exercises the REAL resolution + builder pair."""
    pkg = type(sys)("phonon_vision")
    pkg.__path__ = [os.path.join(_HERE, os.pardir)]
    bpy = type(sys)("bpy")
    bpy.types = type(sys)("bpy.types")
    mathutils = type(sys)("mathutils")
    mathutils.Vector = tuple
    pc = type(sys)("phonon_core")
    kin = type(sys)("phonon_core.kinematics")
    kin.compute_bloch_phases = None
    kin.compute_displacements_at_phase = None
    kin.canonicalize_eigenvector_phase = None
    pc.kinematics = kin
    state = type(sys)("phonon_vision.state")
    state.get_loaded_data = None
    state.PhononVisionProps = None
    sys.modules["phonon_vision"] = pkg
    sys.modules["phonon_vision.state"] = state
    sys.modules["phonon_vision.bonds"] = B
    sys.modules["bpy"] = bpy
    sys.modules["mathutils"] = mathutils
    sys.modules["phonon_core"] = pc
    sys.modules["phonon_core.kinematics"] = kin
    spec = importlib.util.spec_from_file_location(
        "phonon_vision.scene_setup",
        os.path.join(_HERE, os.pardir, "scene_setup.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phonon_vision.scene_setup"] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolved_radius(spec, key, thickness):
    """Reproduce the scene_setup resolution call site exactly: the stashed
    -1.0 sentinel for "no radius_A", the legacy scale default 1.0, and the
    0.7.9 base formula base_radius = bond_thickness × 0.06 (golden)."""
    base = float(thickness) * 0.06
    r_A = float(B.pair_radii_A(spec).get(key, -1.0))
    scale = float(B.pair_scales(spec).get(key, 1.0))
    return B.resolve_radius(r_A if r_A > 0.0 else None, scale, base)


def test_radius_A_absolute_congruence_pin():
    # THE cross-tool pin: radius_A = 0.30 -> resolved cylinder radius 0.30 BU
    # (1 BU ≡ 1 Å), INDEPENDENT of bond_thickness — the navigator renders the
    # same Å value, so one curated file yields geometrically congruent bonds
    # in both tools.
    spec, _ = B.parse_sidecar(_sidecar({"Se-Sn": {"radius_A": 0.30}}))
    assert _resolved_radius(spec, "Se-Sn", 1.0) == 0.30
    assert _resolved_radius(spec, "Se-Sn", 3.0) == 0.30   # thickness bypassed


def test_radius_precedence_matches_navigator():
    # navigator 0.6.76 import order: radius_A (finite) WINS -> else
    # radius_scale (legacy, × base) -> else the pair default (base).
    both, _ = B.parse_sidecar(_sidecar(
        {"Se-Sn": {"radius_A": 0.25, "radius_scale": 2.0}}))
    assert _resolved_radius(both, "Se-Sn", 1.0) == 0.25
    legacy, _ = B.parse_sidecar(_sidecar({"Se-Sn": {"radius_scale": 2.0}}))
    assert abs(_resolved_radius(legacy, "Se-Sn", 1.0) - 0.12) < 1e-12
    neither, _ = B.parse_sidecar(_sidecar({"Se-Sn": {}}))
    assert _resolved_radius(neither, "Se-Sn", 1.0) == 0.06


def test_radius_A_clamp():
    spec, _ = B.parse_sidecar(_sidecar({"Se-Sn": {"radius_A": 5.0},
                                        "Se-Se": {"radius_A": 0.001}}))
    assert spec["bonds"]["Se-Sn"]["radius_A"] == 1.0
    assert spec["bonds"]["Se-Se"]["radius_A"] == 0.02
    # resolve_radius clamps too (defense in depth for un-parsed values)
    assert B.resolve_radius(5.0, None, 0.06) == 1.0
    assert B.resolve_radius(0.001, None, 0.06) == 0.02


def test_radius_A_roundtrip_and_bad_value():
    # Non-numeric radius_A: warned, dropped, pair falls through to legacy.
    spec, w = B.parse_sidecar(_sidecar({"Se-Sn": {"radius_A": "wide"}}))
    assert any("radius_A" in x for x in w)
    assert "radius_A" not in spec["bonds"]["Se-Sn"]
    # The operators store the NORMALIZED spec (json.dumps(spec)) — radius_A
    # must survive the parse -> dumps -> spec_from_json round-trip.
    spec, _ = B.parse_sidecar(_sidecar({"Se-Sn": {"radius_A": 0.30}}))
    back = B.spec_from_json(json.dumps(spec))
    assert B.pair_radii_A(back) == {"Se-Sn": 0.30}


def test_unit_guard_radius_not_diameter():
    # The value handed to the REAL mesh builder is the RADIUS: ring verts sit
    # at distance r from the cylinder axis, so a 0.30 Å spec yields a 0.60
    # Å-DIAMETER cylinder. Catches any halving/doubling (diameter
    # interpretation) between side-car and mesh.
    S = _load_scene_setup_module()
    spec, _ = B.parse_sidecar(_sidecar({"Se-Sn": {"radius_A": 0.30}}))
    r = _resolved_radius(spec, "Se-Sn", 3.0)
    p0 = np.array([[0.0, 0.0, 0.0]])
    p1 = np.array([[0.0, 0.0, 3.0]])       # axis = z, length 3 Å
    verts = S._compute_bond_verts_vectorized(p0, p1, r, n_sides=16)
    axis_dist = np.linalg.norm(verts[:, :2], axis=1)
    assert np.allclose(axis_dist, 0.30, atol=1e-12), axis_dist
    assert abs(2.0 * float(axis_dist.max()) - 0.60) < 1e-9   # diameter 0.60


def test_regression_0_7_9_radius_goldens():
    # GOLDENS from the 0.7.9 resolution (radius = bond_thickness × 0.06 ×
    # radius_scale): spec=None and radius_scale-only files resolve exactly
    # as before — and DO keep scaling with bond_thickness.
    assert _resolved_radius(None, "Se-Sn", 1.0) == 0.06
    assert abs(_resolved_radius(None, "Se-Sn", 3.0) - 0.18) < 1e-12
    legacy, _ = B.parse_sidecar(_sidecar({"Se-Sn": {"radius_scale": 2.0},
                                          "Se-Se": {"radius_scale": 0.5}}))
    assert B.pair_radii_A(legacy) == {}          # no radius_A surfaced
    assert abs(_resolved_radius(legacy, "Se-Sn", 1.0) - 0.12) < 1e-12
    assert abs(_resolved_radius(legacy, "Se-Se", 1.0) - 0.03) < 1e-12
    assert abs(_resolved_radius(legacy, "Se-Sn", 2.0) - 0.24) < 1e-12


def test_all_pairs_have_radius_A_predicate():
    # Drives the UI grey-out of the bond_thickness slider.
    assert B.all_pairs_have_radius_A(None) is False
    empty, _ = B.parse_sidecar(_sidecar({}))
    assert B.all_pairs_have_radius_A(empty) is False
    full, _ = B.parse_sidecar(_sidecar({"Se-Sn": {"radius_A": 0.3},
                                        "Sn-Sn": {"radius_A": 0.15}}))
    assert B.all_pairs_have_radius_A(full) is True
    mixed, _ = B.parse_sidecar(_sidecar({"Se-Sn": {"radius_A": 0.3},
                                         "Sn-Sn": {"radius_scale": 1.5}}))
    assert B.all_pairs_have_radius_A(mixed) is False   # thickness still live


# --- 0.7.12 per-pair COLOR: validation, sRGB->linear, split override --------
def test_color_parse_valid_malformed_absent():
    # Valid #rrggbb stays; malformed -> dropped + warned (treated as absent);
    # a pair with no color key has none. Mirrors the navigator's import regex
    # /^#[0-9a-fA-F]{6}$/ (index.html applyBondSettings).
    spec, w = B.parse_sidecar(_sidecar({
        "Se-Sn": {"color": "#7f8fa6"},       # valid
        "Se-Se": {"color": "#ZZZ"},           # malformed: non-hex, too short
        "Sn-Sn": {"color": "7f8fa6"},         # malformed: no '#'
        "Se-W":  {"color": "#12345"},         # malformed: 5 hex digits
        "Sn-W":  {"radius_A": 0.3},           # no color key at all
    }))
    assert spec["bonds"]["Se-Sn"]["color"] == "#7f8fa6"
    for bad in ("Se-Se", "Sn-Sn", "Se-W"):
        assert "color" not in spec["bonds"][bad], bad
    assert "color" not in spec["bonds"]["Sn-W"]
    assert sum("malformed color" in x for x in w) == 3, w


def test_pair_colors_helper():
    spec, _ = B.parse_sidecar(_sidecar({
        "Se-Sn": {"color": "#aa33cc", "radius_A": 0.3},
        "Se-Se": {"radius_A": 0.2},           # radius only, no color
        "Sn-Sn": {"color": "#010203"}}))
    assert B.pair_colors(spec) == {"Se-Sn": "#aa33cc", "Sn-Sn": "#010203"}
    assert B.pair_colors(None) == {}
    empty, _ = B.parse_sidecar(_sidecar({}))
    assert B.pair_colors(empty) == {}


def test_srgb_to_linear_color_match_pin():
    # THE navigator-vs-Blender color-match guarantee: the display-sRGB hex is
    # linearized via the standard sRGB EOTF before it becomes a Blender LINEAR
    # base color. Independent reproduction of the transfer function asserts the
    # channel math; a build that skipped it (assigned the raw sRGB fractions)
    # differs by ~0.28 in red for #7f8fa6 and fails both asserts below.
    def srgb_lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    hexstr = "#7f8fa6"
    chans = [int(hexstr[1:][i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    exp = tuple(srgb_lin(c) for c in chans)
    got = B.hex_to_linear(hexstr)
    assert all(abs(a - b) < 1e-12 for a, b in zip(got, exp)), (got, exp)
    # ... and it must NOT be the un-linearized raw sRGB fractions.
    assert all(abs(a - b) > 1e-3 for a, b in zip(got, chans)), got
    # black and white are transfer-function fixed points.
    assert B.hex_to_linear("#000000") == (0.0, 0.0, 0.0)
    assert all(abs(c - 1.0) < 1e-12 for c in B.hex_to_linear("#ffffff"))


def test_bond_material_plan_override_precedence():
    P = B.bond_material_plan
    # A curated color WINS regardless of two_color / same-species (both halves
    # solid) — the navigator's custom-mode mutual exclusion.
    assert P("#7f8fa6", True, False) == (1, "custom")
    assert P("#7f8fa6", False, False) == (1, "custom")
    assert P("#7f8fa6", True, True) == (1, "custom")
    # No color -> the pre-0.7.12 behavior, unchanged.
    assert P(None, True, False) == (2, "split")      # diff species -> two halves
    assert P(None, True, True) == (1, "species")     # same species -> one cyl
    assert P(None, False, False) == (1, "grey")      # two_color off -> grey
    assert P(None, False, True) == (1, "grey")


def test_absent_color_keeps_split_default():
    # A radius-only curated file (the 0.7.10 case) carries NO color, so every
    # pair keeps the per-species split/species/grey path — "silence == default".
    spec, _ = B.parse_sidecar(_sidecar({"Se-Sn": {"radius_A": 0.3},
                                        "Se-Se": {"radius_A": 0.2}}))
    colors = B.pair_colors(spec)
    assert colors == {}
    for key in spec["bonds"]:
        _n, style = B.bond_material_plan(colors.get(key), True, False)
        assert style != "custom", key


def test_color_roundtrip_through_spec_json():
    # Operators store json.dumps(spec); color must survive parse -> dumps ->
    # spec_from_json (like radius_A). And a color-less spec is unchanged from
    # the 0.7.11 behavior (pair_colors empty).
    spec, _ = B.parse_sidecar(_sidecar({"Se-Sn": {"color": "#7f8fa6",
                                                  "radius_A": 0.3}}))
    back = B.spec_from_json(json.dumps(spec))
    assert B.pair_colors(back) == {"Se-Sn": "#7f8fa6"}
    plain, _ = B.parse_sidecar(_sidecar({"Se-Sn": {"max": 3.0}}))
    assert B.pair_colors(plain) == {}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"\nALL {len(tests)} BOND-CRITERION / SIDE-CAR TESTS PASS")
