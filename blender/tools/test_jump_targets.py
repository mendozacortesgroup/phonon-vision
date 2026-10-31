#!/usr/bin/env python3
"""Engine jump-target dedup: identity is WRAPPED q, not the label string (0.8.1).

FIELD DEFECT (0.8.0 user acceptance): the Jump row showed TWO Gamma buttons.
The standing Gamma is seeded as the literal "Γ" while band-file labels
arrive raw from BandPath.tick_locations() (e.g. "\\Gamma"), so the old
label-string dedup missed the path start-Gamma and rendered it beside the
standing one. The fix (state.assemble_jump_targets) deduplicates by q wrapped
to [-0.5, 0.5); label plays no part in identity and the first spelling wins.

This drives the REAL shipped function: ../state.py is loaded with a MINIMAL
bpy stub (it imports bpy + bpy.props and defines PropertyGroup subclasses at
module top; the function under test is itself pure/bpy-free).

    python tools/test_jump_targets.py
"""
import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_STATE = os.path.join(_HERE, os.pardir, "state.py")


def _load_state_module():
    """Load ../state.py under a minimal bpy stub (import-time needs only:
    bpy.types.PropertyGroup as a base class and callable bpy.props.* for the
    class-body prop annotations — evaluated eagerly since state.py has no
    ``from __future__ import annotations`` by design)."""
    bpy = types.ModuleType("bpy")

    class _PropertyGroup:
        pass

    bpy.types = types.SimpleNamespace(PropertyGroup=_PropertyGroup)

    def _prop(*args, **kwargs):
        return None

    props = types.ModuleType("bpy.props")
    for _name in ("StringProperty", "IntProperty", "FloatProperty",
                  "BoolProperty", "EnumProperty", "PointerProperty",
                  "FloatVectorProperty", "IntVectorProperty",
                  "CollectionProperty"):
        setattr(props, _name, _prop)
    bpy.props = props
    bpy.data = types.SimpleNamespace()
    bpy.app = types.SimpleNamespace()
    bpy.context = types.SimpleNamespace()
    sys.modules["bpy"] = bpy
    sys.modules["bpy.props"] = props

    pkg = types.ModuleType("phonon_vision")
    pkg.__path__ = [os.path.join(_HERE, os.pardir)]
    sys.modules.setdefault("phonon_vision", pkg)
    spec = importlib.util.spec_from_file_location("phonon_vision.state", _STATE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phonon_vision.state"] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load_state_module()
GAMMA = "Γ"


def _n_gamma(targets):
    """How many targets sit at q ≡ (0,0,0) (any spelling / integer image)."""
    return sum(1 for _lab, q in targets
               if max(abs(S._wrap_q_component(c)) for c in q) < 1e-9)


def test_gamma_absorbs_path_gamma_by_q():
    band_hsp = [("\\Gamma", (0, 0, 0)), ("M", (0.5, 0, 0)),
                ("K", (1 / 3, 1 / 3, 0)), ("\\Gamma", (0, 0, 0))]
    targets = S.assemble_jump_targets(band_hsp)
    assert [t[0] for t in targets] == [GAMMA, "M", "K"], targets
    assert targets[0] == (GAMMA, (0.0, 0.0, 0.0))     # the standing Gamma
    assert _n_gamma(targets) == 1                      # not two


def test_zone_boundary_alias_merges_first_label_wins():
    # M' at -0.5 is the wrapped-image of M at +0.5 -> merged, "M" retained.
    band_hsp = [("M", (0.5, 0, 0)), ("M'", (-0.5, 0, 0))]
    targets = S.assemble_jump_targets(band_hsp)
    assert [t[0] for t in targets] == [GAMMA, "M"], targets
    assert targets[1] == ("M", (0.5, 0.0, 0.0))       # first q spelling wins


def test_distinct_q_same_label_both_kept():
    # Same label, DIFFERENT q -> both kept (label is not identity).
    band_hsp = [("X", (0.5, 0, 0)), ("X", (0, 0.5, 0))]
    targets = S.assemble_jump_targets(band_hsp)
    assert [t[0] for t in targets] == [GAMMA, "X", "X"], targets
    assert targets[1][1] == (0.5, 0.0, 0.0)
    assert targets[2][1] == (0.0, 0.5, 0.0)


def test_negctl_pre_edit_label_dedup_shows_two_gamma():
    """NEGATIVE CONTROL (fail-then-pass, side by side): the PRE-0.8.1
    label-string assembly produced TWO Gamma on the first case and so FAILS
    that test's assertion; the shipped q-based function produces exactly one."""
    band_hsp = [("\\Gamma", (0, 0, 0)), ("M", (0.5, 0, 0)),
                ("K", (1 / 3, 1 / 3, 0)), ("\\Gamma", (0, 0, 0))]

    # verbatim pre-edit assembly (identity == the label string)
    old = [(GAMMA, (0.0, 0.0, 0.0))]
    for lab, q in band_hsp:
        if lab not in [t[0] for t in old]:
            old.append((lab, tuple(q)))
    old_labels = [t[0] for t in old]
    assert old_labels == [GAMMA, "\\Gamma", "M", "K"]     # the DEFECT
    assert _n_gamma(old) == 2                             # two Gamma buttons

    # the first test's expectation applied to the PRE-edit output FAILS:
    try:
        assert old_labels == [GAMMA, "M", "K"]
    except AssertionError:
        pass                                             # expected (defect)
    else:
        raise AssertionError("pre-edit assembly unexpectedly deduped by q")

    # ...and PASSES on the shipped function, identical input:
    new_labels = [t[0] for t in S.assemble_jump_targets(band_hsp)]
    assert new_labels == [GAMMA, "M", "K"]
    assert _n_gamma(S.assemble_jump_targets(band_hsp)) == 1


# ===========================================================================
# 0.8.2: jump targets from the MATERIAL (hsp.py) — sibling band file (Source
# 1) or reciprocal-lattice symmetry (Source 2), never session load order.
# ===========================================================================
_HSP = os.path.join(_HERE, os.pardir, "hsp.py")
from _subset import WHEEL_VERSION  # single source: tools/_subset.py
_WHEEL = os.path.join(_HERE, os.pardir, "wheels",
                      f"phonon_core-{WHEEL_VERSION}-py3-none-any.whl")
_DS = os.path.join(_HERE, os.pardir, os.pardir, "Phonopy-examples",
                   "fatbands_SnSe2")


def _load_hsp_module():
    """Load ../hsp.py standalone — it is bpy-free (numpy + LAZY vendored
    parsers), so no stub is needed; the wheel is only required for Source-1
    tests that actually parse a band file."""
    if _WHEEL not in sys.path:
        sys.path.insert(0, _WHEEL)          # zip-importable pure-python wheel
    spec = importlib.util.spec_from_file_location("phonon_vision.hsp", _HSP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_hsp_module()

# synthetic reciprocal lattices (rows b_i)
import numpy as _np                                                # noqa: E402
_SQUARE = _np.diag([1.0, 1.0, 1.0])
_RECT = _np.diag([1.0, 2.0, 1.0])
_HEX = _np.array([[1.0, 0.0, 0.0],
                  [0.5, _np.sqrt(3) / 2, 0.0], [0.0, 0.0, 0.3]])
_CENTRED = _np.array([[1.0, 0.0, 0.0],
                      [_np.cos(_np.radians(70)), _np.sin(_np.radians(70)), 0.0],
                      [0.0, 0.0, 0.5]])
_OBLIQUE = _np.array([[1.0, 0.0, 0.0], [0.3, 0.5, 0.0], [0.0, 0.0, 1.0]])
_3D = _np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.0, 1.0]])


def _labels(recip):
    return [t[0] for t in H.lattice_jump_targets(recip)]


def test_lattice_snse2_matches_band_file():
    """THE load-bearing assertion: the lattice route on the REAL SnSe2
    phonopy.yaml yields [Γ, M, K] whose M, K reproduce the q's in the
    material's OWN band.hdf5-labelled path to 1e-6."""
    if not os.path.isdir(_DS):
        print("  (dataset absent — SnSe2 lattice test skipped)")
        return
    sys.path.insert(0, _WHEEL)
    from phonon_core import load_band_hdf5
    from phonon_core.engine import load_engine_np
    eng = load_engine_np(os.path.join(_DS, "data_task", "phonopy.yaml"))
    tg = H.lattice_jump_targets(eng.material.reciprocal_lattice)
    assert [t[0] for t in tg] == [GAMMA, "M", "K"], tg
    assert H.lattice_bravais_name(eng.material.reciprocal_lattice) == "hexagonal"
    band = load_band_hdf5(os.path.join(_DS, "band.hdf5"))
    _pos, labels = band.tick_locations()
    qs = [band.q_points[0]]
    cum = 0
    for n in band.segment_nqpoint:
        cum += n
        qs.append(band.q_points[cum - 1])
    qd = {str(lb): _np.asarray(q) for lb, q in zip(labels, qs)}
    tgd = {t[0]: _np.asarray(t[1]) for t in tg}
    dM = float(_np.max(_np.abs(tgd["M"] - qd["M"])))
    dK = float(_np.max(_np.abs(tgd["K"] - qd["K"])))
    print(f"  SnSe2 lattice vs band.hdf5: M dev {dM:.2e}, K dev {dK:.2e} THz-rlu")
    assert dM < 1e-6 and dK < 1e-6


def test_lattice_square_and_rectangular():
    assert _labels(_SQUARE) == [GAMMA, "X", "M"], _labels(_SQUARE)
    assert H.lattice_bravais_name(_SQUARE) == "square"
    assert _labels(_RECT) == [GAMMA, "X", "Y", "S"], _labels(_RECT)
    assert H.lattice_bravais_name(_RECT) == "rectangular"
    # centred-rectangular (L1≈L2, θ not special) -> Γ, X, S
    assert _labels(_CENTRED) == [GAMMA, "X", "S"], _labels(_CENTRED)


def test_lattice_hexagonal_60_and_120():
    hex120 = _np.array([[1.0, 0.0, 0.0],
                        [-0.5, _np.sqrt(3) / 2, 0.0], [0.0, 0.0, 0.3]])
    assert _labels(_HEX) == [GAMMA, "M", "K"]
    assert _labels(hex120) == [GAMMA, "M", "K"]     # per the S&C table


def test_lattice_oblique_gamma_only():
    # 2D but no special direction -> Γ only ([Γ]); no invented labels.
    assert H.lattice_jump_targets(_OBLIQUE) == [(GAMMA, (0.0, 0.0, 0.0))]
    assert H.lattice_bravais_name(_OBLIQUE) is None


def test_lattice_3d_returns_empty():
    # NON-ORTHOGONAL c* (genuinely undetectable) -> [].  0.8.3: this is now the
    # ONLY geometric [] case — an ORTHOGONAL 3D crystal returns its correct
    # k_z=0 points (see test_3d_bulk_honest_provenance), not [].
    assert H.lattice_jump_targets(_3D) == []
    assert H.lattice_bravais_name(_3D) is None
    # dim_hint>=3 stays an explicit caller override ("I know it's 3D, offer
    # nothing") — distinct from the default path, which is unused by the wiring.
    assert H.lattice_jump_targets(_SQUARE, dim_hint=3) == []


def test_same_material_guard_unit():
    class M:
        def __init__(self, sym, rec):
            self.symbols = sym
            self.reciprocal_lattice = rec
    a = M(["Sn", "Se", "Se"], _HEX)
    assert H.same_material(a, M(["Sn", "Se", "Se"], _HEX.copy()))
    assert not H.same_material(a, M(["W", "Se", "Se"], _HEX))       # symbols
    assert not H.same_material(a, M(["Sn", "Se", "Se"], _SQUARE))   # recip


def test_source1_sibling_discovery_and_material_guard():
    """Source 1 finds the same-material sibling band.hdf5, and IGNORES a
    band file whose material differs (targets then fall back to lattice)."""
    if not os.path.isdir(_DS):
        print("  (dataset absent — Source-1 discovery test skipped)")
        return
    sys.path.insert(0, _WHEEL)
    from phonon_core.engine import load_engine_np
    yaml_path = os.path.join(_DS, "data_task", "phonopy.yaml")
    eng = load_engine_np(yaml_path)
    band, prov = H.discover_sibling_band(yaml_path, eng.material)
    assert band is not None and prov == "band.hdf5", (band, prov)

    class Wrong:
        symbols = ["C", "C"]
        reciprocal_lattice = _np.eye(3)
    band2, prov2 = H.discover_sibling_band(yaml_path, Wrong())
    assert band2 is None and prov2 is None          # different material ignored


def test_composition_source1_through_assemble_still_gamma_M_K():
    """0.8.1 behavior intact: Source 1's raw '$\\Gamma$'-spelled labels
    through assemble_jump_targets still collapse to exactly [Γ, M, K]."""
    band_hsp = [("$\\Gamma$", (0, 0, 0)), ("M", (0.5, 0, 0)),
                ("K", (1 / 3, 1 / 3, 0)), ("$\\Gamma$", (0, 0, 0))]
    assert [t[0] for t in S.assemble_jump_targets(band_hsp)] == [GAMMA, "M", "K"]


def test_negctl_bare_phonopy_was_gamma_only():
    """NEGATIVE CONTROL (fail-then-pass): the 0.8.1 defect reproduced — with
    NOTHING remembered (a bare phonopy.yaml, no sibling loaded, no lattice
    route) the jump row is a bare Γ. WITH the lattice route (Source 2) the
    material's own M, K appear even with no band file anywhere."""
    pre = S.assemble_jump_targets([])                       # nothing remembered
    assert [t[0] for t in pre] == [GAMMA]                   # the defect: bare Γ
    post = S.assemble_jump_targets(H.lattice_jump_targets(_HEX))
    assert [t[0] for t in post] == [GAMMA, "M", "K"]        # fixed by the route
    assert pre != post                                      # fail-then-pass


# ===========================================================================
# 0.8.3: slab vs orthogonal-3D-bulk honesty — the lattice route offers the
# correct k_z=0 points in BOTH regimes, but only NAMES a 2D Bravais type for a
# genuine slab (supercell N×M×1). is_slab_supercell reads supercell_matrix.
# ===========================================================================
def test_slab_detector_snse2_is_slab():
    """SnSe2 data_task supercell is [8,8,1] → is_slab True; the slab branch
    reproduces the 0.8.2 canonical behaviour Axel accepted (regression)."""
    if not os.path.isdir(_DS):
        print("  (dataset absent — slab detector test skipped)")
        return
    sys.path.insert(0, _WHEEL)
    from phonon_core.engine import load_engine_np
    yaml_path = os.path.join(_DS, "data_task", "phonopy.yaml")
    sm = H.read_supercell_matrix(yaml_path)
    assert sm.tolist() == [[8, 0, 0], [0, 8, 0], [0, 0, 1]], sm.tolist()
    assert H.is_slab_supercell(yaml_path) is True
    R = load_engine_np(yaml_path).material.reciprocal_lattice
    assert [t[0] for t in H.lattice_jump_targets(R, is_slab=True)] == \
        [GAMMA, "M", "K"]
    assert H.lattice_provenance(R, is_slab=True) == "lattice symmetry (hexagonal)"


def test_3d_bulk_honest_provenance():
    """A cubic BULK (supercell [2,2,2]) → is_slab False; the SAME in-plane
    targets [Γ,X,M] (the correct k_z=0 subset) but a provenance that names
    k_z=0 / 3D and does NOT claim the 2D Bravais type 'square'."""
    import tempfile
    d = tempfile.mkdtemp()
    yaml_path = os.path.join(d, "phonopy.yaml")
    with open(yaml_path, "w") as f:
        f.write("supercell_matrix:\n- [2, 0, 0]\n- [0, 2, 0]\n- [0, 0, 2]\n")
    assert H.is_slab_supercell(yaml_path) is False
    is_slab = H.is_slab_supercell(yaml_path)
    cub = _np.eye(3)                                  # cubic in-plane == square
    assert [t[0] for t in H.lattice_jump_targets(cub, is_slab=is_slab)] == \
        [GAMMA, "X", "M"]
    prov = H.lattice_provenance(cub, is_slab=is_slab)
    print(f"  3D cubic provenance: {prov!r}")
    assert "k_z=0" in prov
    assert "square" not in prov                       # no false 2D-Bravais claim


def test_nonorthogonal_c_star_still_empty():
    non_ortho = _np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.0, 1.0]])
    assert H.lattice_jump_targets(non_ortho, is_slab=False) == []
    assert H.lattice_provenance(non_ortho, is_slab=False) == GAMMA + " only"


def test_negctl_3d_bravais_claim_reproduced_then_honest():
    """NEGATIVE CONTROL (fail-then-pass): with the slab flag IGNORED (the 0.8.2
    behaviour) a 3D cubic crystal's provenance NAMES the 2D Bravais type
    'square' — the architect defect — while the 0.8.3 slab-aware provenance
    does not, and instead flags the k_z=0 partial truth."""
    cub = _np.eye(3)
    name = H.lattice_bravais_name(cub)
    old_prov = f"lattice symmetry ({name})"           # exactly what 0.8.2 shipped
    assert name == "square" and "square" in old_prov  # the false 2D claim
    new_prov = H.lattice_provenance(cub, is_slab=False)
    assert "square" not in new_prov and "k_z=0" in new_prov
    assert new_prov != old_prov                       # fail-then-pass


# ===========================================================================
# 0.8.4: jump-point NAMES come from BandPath.labels (per-segment start/end),
# NOT tick_locations() (matplotlib x-axis strings, compound at discontinuities).
# ===========================================================================
def test_prettify_point_label_unit():
    p = S._prettify_point_label
    assert p("X,") == "X"                       # trailing-comma artifact
    assert p("$\\Gamma$") == GAMMA
    assert p("$\\Gamma$,") == GAMMA             # comma + TeX
    assert p("$R_2$") == "R2"                   # subscript collapse (ASCII)
    assert p("$T_2$,") == "T2"
    assert p("$X^{\\prime}$") == "X\u2032"      # X prime
    assert p("$V_2^{\\prime}$") == "V2\u2032"
    assert p("  Z  ") == "Z"
    assert p("Weird-Name") == "Weird-Name"      # verbatim: never invent


def test_snse2_jump_row_regression():
    """REAL SnSe2 (connected path, no discontinuity): still [Γ,M,K] — the
    accepted row must not regress under the labels-based builder."""
    if not os.path.isdir(_DS):
        print("  (SnSe2 dataset absent — skipped)")
        return
    sys.path.insert(0, _WHEEL)
    from phonon_core import load_band_hdf5
    S.remember_band_hsp(load_band_hdf5(os.path.join(_DS, "band.hdf5")))
    tg = S.assemble_jump_targets(S.get_band_hsp())
    labels = [t[0] for t in tg]
    assert labels == [GAMMA, "M", "K"], labels
    qd = {t[0]: _np.asarray(t[1]) for t in tg}
    assert _np.max(_np.abs(qd["M"] - [0.5, 0, 0])) < 1e-9
    assert _np.max(_np.abs(qd["K"] - [1 / 3, 1 / 3, 0])) < 1e-6


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"\nALL {len(tests)} JUMP-TARGET TESTS PASS")
