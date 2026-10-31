#!/usr/bin/env python3
"""Headless end-to-end kinematics render-path conformance gate (0.7.11;
ARBITRARY-Q ENGINE PATH added 0.8.8).

Builds the REAL add-on scene inside Blender for a fixture, advances through
animation frames via ``scene.frame_set`` (so the registered frame handler —
the actual render path — moves the atoms), reads every atom object's vertex
WORLD positions (``obj.matrix_world @ v.co``), and asserts they match the
bpy-free reference ``motion.positions_at_frame`` for identical inputs. This
is RENDER conformance: what Blender's meshes actually display == the engine's
displacements (the reference is itself engine-conformance-tested without
Blender by tools/test_motion.py).

TWO input classes:

- PRECOMPUTED FILE (band.yaml / band.hdf5): a fixed q-grid; the fixture picks
  a ``q_index`` into it. RENDER conformance only.
- ENGINE (phonopy.yaml + FORCE_CONSTANTS, 0.8.8): the vendored A2 engine
  evaluates D(q) at an ARBITRARY ``q_frac`` (the same seam the GUI q-scrub
  drives: props.q_frac_* -> EngineSource.set_q -> evaluate_dq_np, i.e. the
  dq_numpy code path). RENDER conformance PLUS optional PHYSICS conformance:
  the engine's frequencies at q_frac are compared to a phonopy-A1 golden
  baked into the fixture. On a pseudo-symmetric material at a generic q
  this PHYSICS check is a genuine tie-tolerance REGRESSION TEST in the
  production environment — a pre-0.6.87 (relative-tie) wheel fails it by
  ~0.1 THz, the fixed wheel (0.6.87+, dq_numpy d1059152) passes at ~1e-14.
  (High-symmetry materials have exact ties, so they cannot catch the tie
  bug; the gate needs a pseudo-symmetric fixture.)

Usage (Blender 4.5 LTS; on a cluster, module-load/conda Blender first):

    blender --background --python tools/conformance_headless.py -- \
        --job <fixture.json>

fixture.json (relative "input" resolves against the fixture's directory):

    FILE:   {
      "input": "band.yaml",               # phonopy file WITH eigenvectors
      "supercell": [2, 2, 1], "q_index": 300, "band_index": 4,
      "amplitude": 8.0, "frames_per_cycle": 24,
      "frames": [1, 4, 9, 13, 20, 24],    # optional; default spans a cycle
      "use_canonical_phase": true, "clamp": false, "max_displacement": 0.5
    }
    ENGINE: {
      "input": "phonopy.yaml",            # + sibling FORCE_CONSTANTS
      "q_frac": [0.137, 0.291, 0.053],    # REQUIRED — the arbitrary q
      "supercell": [2, 2, 2], "band_index": 10, "amplitude": 4.0,
      "frames_per_cycle": 60,
      "engine_freq_golden": [ ... ],      # optional: phonopy-A1 THz at q_frac
      "engine_freq_tol": 1e-6,            # optional (THz, |nu|>1e-3)
      "engine_lambda_tol": 1e-9           # optional (THz^2, near-zero)
    }

Exit codes: 0 conformant; 1 mismatch; 2 usage/load error.

Tolerance: Blender stores mesh vertex coordinates as float32, so the gate
compares at atol = 1e-4 Å against the float64 reference (float32 eps is
~1.2e-7 relative; supercell coordinates are O(10 Å)).
"""
import importlib
import importlib.util
import json
import os
import sys
import traceback

import numpy as np

import bpy


ATOL = 1e-4   # Å; float32 mesh storage (see module docstring)


def _args_after_ddash():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


def _fail_usage(msg):
    print(f"[conformance] ERROR: {msg}")
    print(__doc__)
    sys.exit(2)


def main():
    args = _args_after_ddash()
    if len(args) != 2 or args[0] != "--job":
        _fail_usage("expected: -- --job <fixture.json>")
    job_path = os.path.abspath(args[1])
    try:
        with open(job_path, "r", encoding="utf-8") as fh:
            fx = json.load(fh)
    except Exception as e:
        _fail_usage(f"cannot read fixture {job_path}: {e}")

    in_path = fx.get("input")
    if not in_path:
        _fail_usage('fixture is missing "input"')
    if not os.path.isabs(in_path):
        in_path = os.path.join(os.path.dirname(job_path), in_path)

    # Self-registering, no install step (pv_headless_smoke pattern), but
    # robust to a VERSIONED checkout dir ("phonon_vision-0.7.11" is not an
    # importable name): load the package explicitly under a canonical name
    # with the add-on dir as its search path, so every relative import
    # inside the add-on resolves regardless of what the dir is called.
    addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent = os.path.dirname(addon_dir)
    base = os.path.basename(addon_dir)
    pkg = base if base.isidentifier() else "phonon_vision"
    if parent not in sys.path:
        sys.path.insert(0, parent)

    if pkg in sys.modules:
        addon = sys.modules[pkg]
    else:
        spec = importlib.util.spec_from_file_location(
            pkg, os.path.join(addon_dir, "__init__.py"),
            submodule_search_locations=[addon_dir])
        addon = importlib.util.module_from_spec(spec)
        sys.modules[pkg] = addon
        spec.loader.exec_module(addon)
    if hasattr(addon, "register"):
        try:
            addon.register()
        except Exception as e:
            print(f"[conformance] register() note: {e}")

    api = importlib.import_module(f"{pkg}.api")
    motion = importlib.import_module(f"{pkg}.motion")
    state = importlib.import_module(f"{pkg}.state")

    scene = bpy.context.scene
    props = scene.phonon_vision

    print(f"[conformance] loading {in_path}")
    api.load_file(scene, in_path)
    is_engine = (props.data_kind == "ENGINE")

    sc = fx.get("supercell", [1, 1, 1])
    props.supercell_n = (int(sc[0]), int(sc[1]), int(sc[2]))
    props.band_index = int(fx.get("band_index", 0))
    props.amplitude = float(fx.get("amplitude", 8.0))
    props.frames_per_cycle = int(fx.get("frames_per_cycle", 24))
    props.use_canonical_phase = bool(fx.get("use_canonical_phase", True))
    props.clamp_amplitude = bool(fx.get("clamp", False))
    if "max_displacement" in fx:
        props.max_displacement = float(fx["max_displacement"])

    # q source: a precomputed file indexes its own q-grid; an ENGINE source
    # (phonopy.yaml input) evaluates D(q) at an ARBITRARY q_frac. Setting the
    # q_frac_* props mirrors the GUI; the explicit set_q below guarantees the
    # engine cache is at q_frac regardless of whether the property-update
    # callback fires under --background.
    if is_engine:
        qf = fx.get("q_frac")
        if not qf or len(qf) != 3:
            _fail_usage('engine fixture (phonopy.yaml input) requires '
                        '"q_frac": [x, y, z]')
        q_frac = tuple(float(x) for x in qf)
        props.q_frac_1, props.q_frac_2, props.q_frac_3 = q_frac
        props.q_index = 0
    else:
        props.q_index = int(fx.get("q_index", 0))

    api.build_scene_from_data(scene)

    data = state.get_loaded_data(scene)
    if is_engine:
        data.set_q(q_frac)   # evaluate_dq_np(engine, q_frac) -> the dq_numpy path
    mode = data.get_mode(int(props.q_index), int(props.band_index))
    q = np.asarray(mode.q_point, dtype=np.float64)
    src = "ENGINE q_frac" if is_engine else f"q_index {int(props.q_index)}"
    print(f"[conformance] {src} -> q = {q.tolist()}, "
          f"band {int(props.band_index)}, "
          f"{'NON-Gamma' if np.any(np.abs(q) > 1e-9) else 'Gamma'} mode, "
          f"supercell {tuple(int(x) for x in props.supercell_n)}, "
          f"clamp {'ON' if props.clamp_amplitude else 'OFF'}"
          + (f", {int(data.n_bands)} bands (arbitrary-q engine)"
             if is_engine else ""))

    # Species objects + shared stashed geometry (what the render path reads)
    species_objs = {o["species"]: o for o in bpy.data.objects
                    if o.name.startswith("PV.Atoms.")}
    if not species_objs:
        _fail_usage("scene build produced no PV.Atoms.* objects")
    first = next(iter(species_objs.values()))
    n_slots = int(first["n_slots_total"])
    base_pos, uc_idx = motion.unpack_geometry(
        first["all_base_pos"], first["uc_indices"], n_slots)
    atom_idx = np.asarray(first["atom_indices"], dtype=np.int64)

    fpc = int(props.frames_per_cycle)
    frames = fx.get("frames")
    if not frames:
        frames = sorted({1, 1 + fpc // 6, 1 + fpc // 3, 1 + fpc // 2,
                         1 + (2 * fpc) // 3, 1 + fpc})   # spans a full cycle
    frames = [int(f) for f in frames]

    failures = 0
    world_by_frame = {}
    for f in frames:
        scene.frame_set(f)   # fires frame_change_pre -> the real render path

        expected = motion.positions_at_frame(
            mode, data.material, base_pos, uc_idx,
            frame=f,
            frames_per_cycle=fpc,
            frame_start=int(scene.frame_start),
            amplitude=float(props.amplitude),
            atom_indices=atom_idx,
            use_canonical_phase=bool(props.use_canonical_phase),
            clamp=bool(props.clamp_amplitude),
            cap=float(props.max_displacement),
        )

        frame_world = []
        worst = 0.0
        for sp, obj in sorted(species_objs.items()):
            slots = list(obj["slot_indices"])
            mw = obj.matrix_world
            world = np.array([list(mw @ v.co) for v in obj.data.vertices],
                             dtype=np.float64)
            frame_world.append(world)
            err = float(np.max(np.abs(world - expected[slots]))) \
                if len(world) else 0.0
            worst = max(worst, err)
            if not np.allclose(world, expected[slots], rtol=0.0, atol=ATOL):
                failures += 1
                print(f"[conformance] FAIL frame {f} species {sp}: "
                      f"max|world - reference| = {err:.3e} Å > atol {ATOL}")
        if worst <= ATOL:
            print(f"[conformance] frame {f:4d}: OK "
                  f"(max deviation {worst:.3e} Å over {n_slots} slots)")
        world_by_frame[f] = np.concatenate(frame_world, axis=0)

    # Belt: the handler must actually have MOVED atoms between two frames
    # that are not a whole cycle apart (catches a dead frame handler, which
    # would otherwise "conform" at the build frame only).
    distinct = [f for f in frames if (f - frames[0]) % fpc != 0]
    if distinct:
        delta = float(np.max(np.abs(world_by_frame[distinct[0]]
                                    - world_by_frame[frames[0]])))
        if delta < 1e-6:
            failures += 1
            print(f"[conformance] FAIL: atoms did not move between frames "
                  f"{frames[0]} and {distinct[0]} (handler dead?)")

    # PHYSICS conformance (engine mode, 0.8.8): the render check above proves
    # Blender DISPLAYS what the engine computes; this proves the engine
    # COMPUTES the right thing on the cluster, by comparing its frequencies at
    # q_frac to a phonopy-A1 golden. Metric doctrine: |dnu| above NU_FLOOR,
    # |dlambda| below (near-zero eigensolver noise is LAPACK-build dependent).
    if is_engine and "engine_freq_golden" in fx:
        golden = np.sort(np.asarray(fx["engine_freq_golden"], dtype=np.float64))
        got = np.sort(np.asarray(data.frequencies, dtype=np.float64))
        nu_floor = 1e-3
        ftol = float(fx.get("engine_freq_tol", 1e-6))
        ltol = float(fx.get("engine_lambda_tol", 1e-9))
        if got.shape != golden.shape:
            failures += 1
            print(f"[conformance] FAIL physics: engine returned {got.shape[0]} "
                  f"bands, golden has {golden.shape[0]}")
        else:
            big = np.abs(golden) > nu_floor
            d_nu = float(np.abs(got[big] - golden[big]).max()) if big.any() else 0.0
            d_lam = (float(np.abs(np.sign(got[~big]) * got[~big] ** 2
                                  - np.sign(golden[~big]) * golden[~big] ** 2).max())
                     if (~big).any() else 0.0)
            ok = d_nu < ftol and d_lam < ltol
            print(f"[conformance] physics: engine vs phonopy-A1 golden at "
                  f"q_frac -> max|dnu| = {d_nu:.3e} THz (tol {ftol:.0e}), "
                  f"max|dlambda| = {d_lam:.3e} THz^2 (tol {ltol:.0e}) -> "
                  f"{'OK' if ok else 'FAIL'}")
            if not ok:
                failures += 1

    if failures:
        print(f"[conformance] RESULT: FAIL ({failures} check(s) failed)")
        sys.exit(1)
    print(f"[conformance] RESULT: PASS — rendered atom positions match the "
          f"engine reference on {len(frames)} frames (atol {ATOL} Å)")
    sys.exit(0)


if __name__ == "__main__":
    # Blender does NOT reliably turn an unhandled --python exception into a
    # non-zero exit code; a gate must never "pass" by crashing. sys.exit(N)
    # (SystemExit) IS propagated, so route every failure through it.
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        print("[conformance] RESULT: ERROR (unhandled exception)")
        sys.exit(2)
