#!/usr/bin/env python3
"""Regression test for type-aware vector job round-trip (0.7.5).

Runs in plain Python (no bpy) against ../job.py using a fake PropertyGroup whose
`bl_rna` reports element types and whose `supercell_n` rejects float tuples —
exactly as Blender's IntVectorProperty does. This locks in the fix for the bug
where an integer supercell (e.g. 9x9x1) was serialized as floats and then
silently rejected on apply, leaving the (3,3,3) default (wrong render).

    python tools/test_job_roundtrip.py
"""
import contextlib
import importlib.util
import io
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_JOB = os.path.join(_HERE, os.pardir, "job.py")


def _load_job_module():
    # Load ../job.py standalone, stubbing its `from .state import …`.
    pkg = type(sys)("phonon_vision")
    pkg.__path__ = [os.path.join(_HERE, os.pardir)]
    state = type(sys)("phonon_vision.state")

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    state._SuppressUpdates = _Ctx
    state._SuppressSpeciesUpdates = _Ctx
    state.get_loaded_data = lambda scene: getattr(scene, "_pv_data", None)
    sys.modules["phonon_vision"] = pkg
    sys.modules["phonon_vision.state"] = state
    spec = importlib.util.spec_from_file_location("phonon_vision.job", _JOB)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phonon_vision.job"] = mod
    spec.loader.exec_module(mod)
    return mod


J = _load_job_module()

_VEC_TYPES = {"supercell_n": "INT", "arrow_color": "FLOAT",
              "camera_target_offset": "FLOAT", "overlay_text_color": "FLOAT",
              "overlay_background_color": "FLOAT"}


class _RnaProp:
    def __init__(self, t):
        self.type = t


class _Rna:
    def __init__(self, types):
        self.properties = {k: _RnaProp(v) for k, v in types.items()}


class _Species:
    def __init__(self, name, color, radius):
        self.name = name
        self.color = list(color)
        self.radius = radius


class _Props:
    """Fake PropertyGroup: supercell_n behaves like an IntVectorProperty."""
    def __init__(self):
        object.__setattr__(self, "_d", {})
        object.__setattr__(self, "bl_rna", _Rna(_VEC_TYPES))
        d = self._d
        d["supercell_n"] = (3, 3, 3)            # the masking default
        d["arrow_color"] = [0.9, 0.1, 0.1]
        d["camera_target_offset"] = [0.0, 0.0, 0.0]
        d["overlay_text_color"] = [0.0, 0.0, 0.0]
        d["overlay_background_color"] = [1.0, 1.0, 1.0, 0.7]
        d["q_index"] = 0
        d["band_index"] = 0
        d["output_directory"] = "/tmp"
        d["output_format"] = "PNG"
        for _g, _k, p, kind in J._MAP:
            if p not in d:
                d[p] = "" if kind == "scalar" else [0, 0, 0]
        d["q_frac_1"] = 0.0
        d["q_frac_2"] = 0.0
        d["q_frac_3"] = 0.0
        d["path_s"] = 0.0
        d["camera_miller_h"] = 0.0
        d["camera_miller_k"] = 0.0
        d["camera_miller_l"] = 0.0
        d["species_overrides"] = [_Species("Sn", [0.4, 0.5, 0.9], 1.2)]
        d["bonds_spec_json"] = ""      # curated-bonds spec (schema v2)

    def __setattr__(self, k, v):
        if k == "supercell_n":
            for x in v:
                if isinstance(x, bool) or not isinstance(x, int):
                    raise TypeError("expected an int, not float")
            self._d[k] = tuple(int(x) for x in v)
            return
        self._d[k] = v

    def __getattr__(self, k):
        d = object.__getattribute__(self, "_d")
        if k in d:
            return d[k]
        raise AttributeError(k)


class _Scene:
    def __init__(self):
        self.phonon_vision = _Props()


def test_int_vector_survives_as_ints():
    sc = _Scene()
    sc.phonon_vision.supercell_n = (9, 9, 1)
    job = J.scene_to_job(sc)
    assert job["display"]["supercell"] == [9, 9, 1]
    assert all(isinstance(x, int) and not isinstance(x, bool)
               for x in job["display"]["supercell"])


def test_apply_int_vector_not_default():
    sc = _Scene()
    sc.phonon_vision.supercell_n = (9, 9, 1)
    job = J.scene_to_job(sc)
    fresh = _Scene()  # starts at default (3,3,3)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        J.apply_job_to_scene(fresh, job)
    assert fresh.phonon_vision.supercell_n == (9, 9, 1)
    assert "could not set" not in err.getvalue()


def test_existing_float_job_applies():
    fresh = _Scene()
    J.apply_job_to_scene(fresh, {"display": {"supercell": [9.0, 9.0, 1.0]}})
    assert fresh.phonon_vision.supercell_n == (9, 9, 1)


def test_float_vec_unchanged():
    sc = _Scene()
    sc.phonon_vision.arrow_color = [0.12, 0.34, 0.56]
    job = J.scene_to_job(sc)
    assert job["appearance"]["arrow_color"] == [0.12, 0.34, 0.56]
    assert all(isinstance(x, float) for x in job["appearance"]["arrow_color"])
    fresh = _Scene()
    J.apply_job_to_scene(fresh, job)
    assert [round(x, 6) for x in fresh.phonon_vision.arrow_color] == \
        [0.12, 0.34, 0.56]


_SPEC = {"format": "phonon-vision-bonds", "version": 1, "generator": "test",
         "material": {"formula": "SnSeSe", "natoms": 3}, "units": "angstrom",
         "bonds": {"Se-Sn": {"max": 3.11, "radius_scale": 1.5},
                   "Se-Se": {"max": 2.88}},
         "source": {"filename": "snse2_bonds.json", "imported": "2026-07-03"}}


def test_v2_bonds_roundtrip():
    import json
    sc = _Scene()
    sc.phonon_vision.bonds_spec_json = json.dumps(_SPEC)
    job = J.scene_to_job(sc)
    # the CONSTANT, not a literal (the 0.6.79 lesson): the writer
    # emits the current schema; v2 bonds semantics are unchanged.
    assert job["schema_version"] == J.SCHEMA_VERSION
    assert job["bonds"] == _SPEC          # embedded BY VALUE, incl. source
    fresh = _Scene()
    J.apply_job_to_scene(fresh, job)
    assert json.loads(fresh.phonon_vision.bonds_spec_json) == _SPEC


def test_v1_job_loads_with_bonds_empty():
    import json
    # A v1 job has no "bonds" key. Applying it must CLEAR any loaded spec
    # (a job is a complete render description) so it renders exactly as at
    # 0.7.8 regardless of prior GUI state.
    sc = _Scene()
    sc.phonon_vision.bonds_spec_json = json.dumps(_SPEC)   # stale GUI state
    v1_job = {"schema_version": 1,
              "display": {"supercell": [4, 4, 1]}}
    J.apply_job_to_scene(sc, v1_job)
    assert sc.phonon_vision.bonds_spec_json == ""
    assert sc.phonon_vision.supercell_n == (4, 4, 1)


def test_v2_job_without_bonds_clears():
    import json
    sc = _Scene()
    sc.phonon_vision.bonds_spec_json = json.dumps(_SPEC)
    J.apply_job_to_scene(sc, {"schema_version": 2})
    assert sc.phonon_vision.bonds_spec_json == ""


# ---------------------------------------------------------------------------
# 0.8.0: ENGINE-source jobs (schema v3)
# ---------------------------------------------------------------------------
_DS = os.path.join(_HERE, os.pardir, os.pardir, "Phonopy-examples",
                   "fatbands_SnSe2", "data_task")


class _FakeEngineData:
    """Duck-typed EngineSource for prop-level roundtrip (no wheel needed)."""
    def __init__(self):
        self.yaml_path = "/data/phonopy.yaml"
        self.fc_path = "/data/FORCE_CONSTANTS"
        self.set_q_calls = []

    def set_q(self, q):
        self.set_q_calls.append(tuple(q))


def test_engine_block_roundtrip():
    """schema v3: ENGINE source serializes {source, phonopy_yaml,
    force_constants, q_frac, band_index} and applies back — including the
    engine-cache re-sync (set_q called with the job's q). FAILS on the
    pre-0.8.0 job.py (no engine block emitted)."""
    sc = _Scene()
    sc._pv_data = _FakeEngineData()
    p = sc.phonon_vision
    p.data_kind = "ENGINE"
    p.yaml_path = "/data/phonopy.yaml"
    p.q_frac_1, p.q_frac_2, p.q_frac_3 = 0.137, 0.291, 0.0
    p.band_index = 4
    p.path_s = 0.42
    job = J.scene_to_job(sc)
    assert job["schema_version"] == J.SCHEMA_VERSION   # constant, not literal
    assert job["engine"] == {
        "source": "engine",
        "phonopy_yaml": "/data/phonopy.yaml",
        "force_constants": "/data/FORCE_CONSTANTS",
        "q_frac": [0.137, 0.291, 0.0],
        "band_index": 4,
        "path_s": 0.42,
    }, job.get("engine")
    fresh = _Scene()
    fresh._pv_data = _FakeEngineData()
    J.apply_job_to_scene(fresh, job)
    fp = fresh.phonon_vision
    assert (fp.q_frac_1, fp.q_frac_2, fp.q_frac_3) == (0.137, 0.291, 0.0)
    assert fp.band_index == 4
    assert abs(fp.path_s - 0.42) < 1e-12                        # UI hint restored
    assert fresh._pv_data.set_q_calls == [(0.137, 0.291, 0.0)]  # cache re-synced


def test_engine_replay_matches_fresh_evaluation():
    """Job replay = reconstruct engine + re-evaluate; must match a FRESH
    evaluation at 1e-12 (eigenvectors are never serialized — the FC file is
    the source of truth). Skips if the dataset is absent (bare-zip)."""
    if not os.path.isdir(_DS):
        print("  (dataset absent — replay test skipped)")
        return
    import numpy as np
    from _subset import WHEEL_VERSION  # single source: tools/_subset.py
    sys.path.insert(0, os.path.join(_HERE, os.pardir, "wheels",
                                    f"phonon_core-{WHEEL_VERSION}-py3-none-any.whl"))
    import importlib.util as _ilu
    mspec = _ilu.spec_from_file_location(
        "phonon_vision.motion", os.path.join(_HERE, os.pardir, "motion.py"))
    motion = _ilu.module_from_spec(mspec)
    sys.modules["phonon_vision.motion"] = motion
    mspec.loader.exec_module(motion)
    yaml_path = os.path.join(_DS, "phonopy.yaml")
    q = (0.137, 0.291, 0.0)
    job = {"schema_version": 3,
           "engine": {"source": "engine", "phonopy_yaml": yaml_path,
                      "force_constants": os.path.join(_DS, "FORCE_CONSTANTS"),
                      "q_frac": list(q), "band_index": 4}}
    replay = J.load_engine_source_from_job(job)
    from phonon_core.engine import evaluate_dq_np
    f_fresh, v_fresh = evaluate_dq_np(replay.engine, q)
    dfreq = float(np.max(np.abs(replay.frequencies - f_fresh)))
    m = replay.get_mode(0, 4)
    dvec = float(np.max(np.abs(m.eigenvector - v_fresh[4])))
    print(f"  replay vs fresh: max|dnu| = {dfreq:.3e} THz, "
          f"max|d eig| = {dvec:.3e}")
    assert dfreq <= 1e-12 and dvec <= 1e-12


def test_engine_job_missing_fc_named_error():
    """A dead force_constants path fails LOUDLY with the named error."""
    job = {"engine": {"source": "engine",
                      "phonopy_yaml": __file__,          # exists
                      "force_constants": "/nonexistent/FORCE_CONSTANTS",
                      "q_frac": [0, 0, 0], "band_index": 0}}
    try:
        J.load_engine_source_from_job(job)
    except FileNotFoundError as e:
        assert "force_constants not found" in str(e)
        return
    raise AssertionError("missing FC did not raise the named error")


def test_engine_job_path_s_replays_without_band_sibling():
    """0.8.5: a job carrying `path_s` replays from q_frac and does NOT require
    the band sibling to exist (path_s is a UI hint, inert to replay). Copy
    phonopy.yaml + FORCE_CONSTANTS to a dir with NO band file and replay."""
    if not os.path.isdir(_DS):
        print("  (dataset absent — path_s replay test skipped)")
        return
    import shutil
    import tempfile
    import numpy as np
    from _subset import WHEEL_VERSION  # single source: tools/_subset.py
    sys.path.insert(0, os.path.join(_HERE, os.pardir, "wheels",
                                    f"phonon_core-{WHEEL_VERSION}-py3-none-any.whl"))
    import importlib.util as _ilu
    mspec = _ilu.spec_from_file_location(
        "phonon_vision.motion", os.path.join(_HERE, os.pardir, "motion.py"))
    motion = _ilu.module_from_spec(mspec)
    sys.modules["phonon_vision.motion"] = motion
    mspec.loader.exec_module(motion)
    d = tempfile.mkdtemp()
    shutil.copy(os.path.join(_DS, "phonopy.yaml"), d)
    shutil.copy(os.path.join(_DS, "FORCE_CONSTANTS"), d)
    assert not any("band" in f.lower() for f in os.listdir(d))   # NO band sibling
    q = (0.137, 0.291, 0.0)
    job = {"schema_version": 3,
           "engine": {"source": "engine",
                      "phonopy_yaml": os.path.join(d, "phonopy.yaml"),
                      "force_constants": os.path.join(d, "FORCE_CONSTANTS"),
                      "q_frac": list(q), "band_index": 4, "path_s": 0.75}}
    src = J.load_engine_source_from_job(job)                 # must not need a band
    assert np.max(np.abs(np.asarray(src.current_q) - np.asarray(q))) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"\nALL {len(tests)} JOB ROUND-TRIP TESTS PASS")
