#!/usr/bin/env python3
"""Path walking (0.8.5) — the parameterization layer over engine mode.

One slider (`path_s`, 0..1) carries q CONTINUOUSLY along the band file's
declared path, with exact D(q) at every position: s -> segment (by cumulative
Cartesian length) -> t within segment -> q = (1-t)q_start + t q_end (fractional
interpolation) -> the q1/q2/q3 write goes through the existing guarded update.

Drives the REAL shipped state.py via a minimal bpy stub (the path helpers and
the segment table are pure). Dataset: SnSe2 (3 connected segments).

    python tools/test_path_walk.py
"""
import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
from _subset import WHEEL_VERSION  # single source: tools/_subset.py
_WHEEL = os.path.join(_HERE, os.pardir, "wheels",
                      f"phonon_core-{WHEEL_VERSION}-py3-none-any.whl")
_SNSE2 = os.path.join(_HERE, os.pardir, os.pardir, "Phonopy-examples",
                      "fatbands_SnSe2")


def _load_state_from(path):
    """Load a state.py (given path) under a minimal bpy stub."""
    bpy = types.ModuleType("bpy")
    bpy.types = types.SimpleNamespace(PropertyGroup=type("_PG", (), {}))

    def _prop(*a, **k):
        return None
    props = types.ModuleType("bpy.props")
    for nm in ("StringProperty", "IntProperty", "FloatProperty", "BoolProperty",
               "EnumProperty", "PointerProperty", "FloatVectorProperty",
               "IntVectorProperty", "CollectionProperty"):
        setattr(props, nm, _prop)
    bpy.props = props
    bpy.data = bpy.app = bpy.context = types.SimpleNamespace()
    sys.modules["bpy"] = bpy
    sys.modules["bpy.props"] = props
    pkg = types.ModuleType("phonon_vision")
    pkg.__path__ = [os.path.join(_HERE, os.pardir)]
    sys.modules.setdefault("phonon_vision", pkg)
    spec = importlib.util.spec_from_file_location("phonon_vision.state", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phonon_vision.state"] = mod
    spec.loader.exec_module(mod)
    return mod


if _WHEEL not in sys.path:
    sys.path.insert(0, _WHEEL)
S = _load_state_from(os.path.join(_HERE, os.pardir, "state.py"))
import numpy as _np                                                # noqa: E402


def _band(path):
    from phonon_core import load_band_hdf5
    return load_band_hdf5(path)


# --- fakes for the callback (one-way-flow) test -----------------------------
class _Props:
    def __init__(self):
        self.q_frac_1 = self.q_frac_2 = self.q_frac_3 = 0.0
        self.path_s = 0.0
        self.data_kind = "ENGINE"


class _Scene:
    def __init__(self, name):
        self.name = name
        self.phonon_vision = _Props()


class _Ctx:
    def __init__(self, scene):
        self.scene = scene


def test_interpolation_closed_form_and_engine():
    if not os.path.isdir(_SNSE2):
        print("  (SnSe2 absent — skipped)")
        return
    S.remember_band_hsp(_band(os.path.join(_SNSE2, "band.hdf5")))
    segs = S.get_path_segments()
    cum = [0.0]
    for s in segs:
        cum.append(cum[-1] + s[4])
    total = cum[-1]
    # mid-segment-1 (M-K): s that lands halfway through segment index 1
    s_mid = (cum[1] + 0.5 * segs[1][4]) / total
    q_walk = _np.asarray(S.q_at_path_s(s_mid))
    q_closed = 0.5 * _np.asarray(segs[1][2]) + 0.5 * _np.asarray(segs[1][3])
    dev = float(_np.max(_np.abs(q_walk - q_closed)))
    print(f"  mid-seg1: |walk - closed-form| = {dev:.1e}")
    assert dev < 1e-12
    # engine: feeding the walk's q equals feeding the hand-computed q
    from phonon_core.engine import load_engine_np, evaluate_dq_np
    eng = load_engine_np(os.path.join(_SNSE2, "data_task", "phonopy.yaml"))
    fw, _ = evaluate_dq_np(eng, q_walk)
    # same array in, same out (the evaluator is a deterministic function of
    # q — feeding the walk's q reproduces bitwise):
    assert _np.array_equal(fw, evaluate_dq_np(eng, q_walk)[0])
    # and it matches evaluation at the hand-computed q to engine precision:
    assert _np.max(_np.abs(fw - evaluate_dq_np(eng, q_closed)[0])) < 1e-9
    print("  SnSe2 mid-seg1: walk q == closed form; evaluate_dq_np "
          "deterministic + matches hand q")


def test_snse2_regression_connected_path():
    if not os.path.isdir(_SNSE2):
        print("  (SnSe2 absent — skipped)")
        return
    S.remember_band_hsp(_band(os.path.join(_SNSE2, "band.hdf5")))
    segs = S.get_path_segments()
    assert len(segs) == 3                                    # Γ-M, M-K, K-Γ
    assert [(s[0], s[1]) for s in segs] == [("Γ", "M"), ("M", "K"),
                                            ("K", "Γ")]
    assert _np.max(_np.abs(S.q_at_path_s(0.0))) < 1e-12      # s=0 -> Γ
    assert _np.max(_np.abs(S.q_at_path_s(1.0))) < 1e-12      # s=1 -> Γ (K->Γ end)
    # connected: every interior boundary shares its endpoint (seamless)
    for i in range(len(segs) - 1):
        assert segs[i][3] == segs[i + 1][2]
    # walking to a mid point gives an interior q
    qm = S.q_at_path_s(0.2)
    assert any(abs(c) > 1e-6 for c in qm)


def test_one_way_flow():
    b = os.path.join(_SNSE2, "band.hdf5")
    if not os.path.isfile(b):
        print("  (SnSe2 absent — skipped)")
        return
    S.remember_band_hsp(_band(b))                           # path exists, on-path
    assert not S.is_off_path()
    ctx = _Ctx(_Scene("walk_test"))                          # engine data absent
    p = ctx.scene.phonon_vision                              # -> eval no-ops (bpy-free)
    # slider drives q -> ON-path, q overwritten from the path
    p.path_s = 0.3
    S._on_path_s_changed(ctx)
    assert not S.is_off_path()
    assert [p.q_frac_1, p.q_frac_2, p.q_frac_3] == S.q_at_path_s(0.3)
    # a DIRECT q change (what a q slider / jump button fires) -> OFF-path
    p.q_frac_1 = 0.9
    S._on_engine_q_change(ctx)
    assert S.is_off_path()
    # grabbing the slider again -> back ON-path, q overwritten
    p.path_s = 0.6
    S._on_path_s_changed(ctx)
    assert not S.is_off_path()
    assert [p.q_frac_1, p.q_frac_2, p.q_frac_3] == S.q_at_path_s(0.6)
    print("  one-way flow: slider->on-path(overwrites q); direct q->off-path")


def test_negctl_pre_edit_has_no_segment_table():
    """NEGATIVE CONTROL (fail-then-pass): the pre-0.8.5 state.py has no path
    segment table (the API does not exist); the current one builds it."""
    import zipfile
    import tempfile
    z = os.path.join(_HERE, os.pardir, os.pardir, "phonon_vision-0_8_4.zip")
    b = os.path.join(_SNSE2, "band.hdf5")
    if not os.path.isfile(z) or not os.path.isfile(b):
        print("  (0.8.4 snapshot or SnSe2 absent — negctl skipped)")
        return
    d = tempfile.mkdtemp()
    with zipfile.ZipFile(z) as zf:
        zf.extract("state.py", d)
    pre = _load_state_from(os.path.join(d, "state.py"))
    assert not hasattr(pre, "get_path_segments")            # feature absent pre-edit
    # ...and present + building on the current tree:
    cur = _load_state_from(os.path.join(_HERE, os.pardir, "state.py"))
    cur.remember_band_hsp(_band(b))
    assert len(cur.get_path_segments()) == 3


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"\nALL {len(tests)} PATH-WALK TESTS PASS")
