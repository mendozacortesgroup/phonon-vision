"""phonon_vision.job — bidirectional mapping between a job dict and the scene props.

One mapping table (`_MAP`) drives BOTH directions, so apply/read can't drift:

    apply_job_to_scene(scene, job)  -> set every phonon_vision prop from the job
    scene_to_job(scene)             -> read every prop back into a job dict

Pure prop I/O — no load, no build, no render. See docs/job_schema_v1.md for the
schema. Groups: input / display / appearance / camera / render / overlay /
loop / output, plus the OPTIONAL top-level "bonds" key (schema_version 2):
the navigator bond side-car EMBEDDED BY VALUE — never a file path, so cluster
jobs stay self-contained — with a "source" provenance block. Schema_version 3
(0.8.0) adds the OPTIONAL top-level "engine" key for ENGINE-source jobs:
{"source": "engine", "phonopy_yaml", "force_constants" (RESOLVED path),
"q_frac" [3], "band_index"} — eigenvectors are NEVER serialized (the FC file
is the source of truth; replay re-evaluates and must byte-match a fresh
evaluation — asserted in tools/test_job_roundtrip.py). The reader accepts
schema_version 1-3; the writer emits 3. A job WITHOUT "bonds" clears any
loaded spec (a job is a complete render description), so v1 jobs render
byte-identically to 0.7.8; a job WITHOUT "engine" is a plain file job.
"""
from __future__ import annotations

SCHEMA_VERSION = 3

import json
import sys


def _vec_elem_is_int(props, prop) -> bool:
    """True if ``prop`` is an INT vector (e.g. supercell_n), False for FLOAT
    vectors (colors, offsets). Drives type-aware (de)serialization so an integer
    supercell round-trips as ints and a float color stays float."""
    try:
        return props.bl_rna.properties[prop].type == 'INT'
    except Exception:
        return False


# (group, json_key, prop_name, kind)
#   kind "scalar" -> assigned directly (str/int/float/bool/enum)
#   kind "vec"    -> FloatVector/IntVectorProperty <-> list (any size; round-trips)
# Species (collection) and camera.miller (3 scalar props as one list) are
# special-cased below but still defined in exactly one place each.
_MAP = [
    # --- input ---
    ("input", "file", "yaml_path", "scalar"),
    ("input", "data_kind", "data_kind", "scalar"),
    # --- display: mode + physics + geometry + bonds + cell (all build/anim-time) ---
    ("display", "q_index", "q_index", "scalar"),
    ("display", "band_index", "band_index", "scalar"),
    ("display", "amplitude", "amplitude", "scalar"),
    ("display", "clamp_amplitude", "clamp_amplitude", "scalar"),
    ("display", "max_displacement", "max_displacement", "scalar"),
    ("display", "canonical_phase", "use_canonical_phase", "scalar"),
    ("display", "imaginary_threshold", "imag_freq_threshold", "scalar"),
    ("display", "supercell", "supercell_n", "vec"),
    ("display", "mesh_quality", "render_quality", "scalar"),
    ("display", "show_arrows", "show_arrows", "scalar"),
    ("display", "arrow_scale", "arrow_scale", "scalar"),
    ("display", "show_unit_cell", "show_unit_cell", "scalar"),
    ("display", "unit_cell_kind", "unit_cell_kind", "scalar"),
    ("display", "cell_thickness", "cell_thickness", "scalar"),
    ("display", "show_bonds", "show_bonds", "scalar"),
    ("display", "bonds_across_pbc", "bond_periodic", "scalar"),
    ("display", "bond_tolerance", "bond_tolerance", "scalar"),
    ("display", "bond_thickness", "bond_thickness", "scalar"),
    ("display", "bond_two_color", "bond_two_color", "scalar"),
    ("display", "bond_max_distance_factor", "bond_break_max_factor", "scalar"),
    ("display", "bond_min_distance_factor", "bond_break_min_factor", "scalar"),
    # --- appearance: colors/materials (species handled specially) ---
    ("appearance", "vibrancy", "species_vibrancy", "scalar"),
    ("appearance", "arrow_color", "arrow_color", "vec"),
    # --- camera (miller handled specially) ---
    ("camera", "mode", "camera_view", "scalar"),
    ("camera", "auto_frame", "camera_auto_frame", "scalar"),
    ("camera", "ortho_scale", "camera_ortho_scale", "scalar"),
    ("camera", "distance", "camera_distance", "scalar"),
    ("camera", "azimuth", "camera_azimuth", "scalar"),
    ("camera", "polar", "camera_polar", "scalar"),
    ("camera", "radius", "camera_radius", "scalar"),
    ("camera", "target_offset", "camera_target_offset", "vec"),
    # --- render ---
    ("render", "preset", "render_preset", "scalar"),
    ("render", "lighting", "lighting_style", "scalar"),
    ("render", "eevee_samples", "eevee_samples", "scalar"),
    ("render", "cycles_samples", "cycles_samples", "scalar"),
    ("render", "denoising", "use_denoising", "scalar"),
    ("render", "resolution_x", "render_resolution_x", "scalar"),
    ("render", "resolution_y", "render_resolution_y", "scalar"),
    ("render", "resolution_percentage", "render_percentage", "scalar"),
    ("render", "transparent_background", "transparent_background", "scalar"),
    # --- overlay ---
    ("overlay", "show_overlay", "show_overlay", "scalar"),
    ("overlay", "show_mode_overlay", "show_mode_overlay", "scalar"),
    ("overlay", "text_size", "overlay_text_size", "scalar"),
    ("overlay", "text_color", "overlay_text_color", "vec"),
    ("overlay", "background_color", "overlay_background_color", "vec"),
    # --- loop ---
    ("loop", "frames_per_cycle", "frames_per_cycle", "scalar"),
    ("loop", "fps", "fps_render", "scalar"),
    # --- output (prop fields; output.path/animation are job-only, see below) ---
    ("output", "directory", "output_directory", "scalar"),
    ("output", "format", "output_format", "scalar"),
]

# Camera Miller indices are three separate FloatProperties exposed as one list.
_MILLER_PROPS = ("camera_miller_h", "camera_miller_k", "camera_miller_l")


def apply_job_to_scene(scene, job: dict) -> None:
    """Write every present prop from ``job`` into ``scene.phonon_vision``.

    Pure prop-setting: no load/build/render. Missing keys are left at their
    current value (so partial jobs are fine). Call AFTER api.load_file so this
    OVERRIDES the species defaults / indices that load populated.
    """
    props = scene.phonon_vision

    # Suppress all prop-update callbacks during the bulk write: setting dozens of
    # props would otherwise fire dozens of refreshes/rebuilds (and, before 0.7.1,
    # NameError spam). The caller (pv_inner) builds/syncs/frames explicitly after.
    from .state import _SuppressUpdates
    with _SuppressUpdates():
        _apply_job_body(props, job)

    # engine : ENGINE-source jobs (schema v3). Prop writes stay suppressed
    # (no callback storm); the loaded EngineSource — created by the caller's
    # load_file BEFORE apply, per the pv_inner order — is re-synced HERE so
    # the engine cache matches the job's q even though the q_frac_* update
    # callbacks never fired. The block's band_index is authoritative for
    # engine jobs (it also lives in display.band_index; same value when
    # written by scene_to_job).
    eng_blk = job.get("engine")
    if isinstance(eng_blk, dict):
        q = [float(x) for x in eng_blk.get("q_frac", (0.0, 0.0, 0.0))]
        with _SuppressUpdates():
            props.q_frac_1, props.q_frac_2, props.q_frac_3 = q
            if "band_index" in eng_blk:
                props.band_index = int(eng_blk["band_index"])
            if "path_s" in eng_blk:      # 0.8.5 UI hint; q_frac stays authoritative
                props.path_s = float(eng_blk["path_s"])
        from .state import get_loaded_data
        data = get_loaded_data(scene)
        if data is not None and hasattr(data, "set_q"):
            data.set_q(tuple(q))


def _apply_job_body(props, job: dict) -> None:
    for group, key, prop, kind in _MAP:
        g = job.get(group)
        if not isinstance(g, dict) or key not in g:
            continue
        value = g[key]
        try:
            if kind == "vec":
                # Type-aware: IntVectorProperty (supercell) needs ints — a float
                # tuple is rejected and would silently keep the default. round()
                # also lets existing float-valued job files apply correctly.
                if _vec_elem_is_int(props, prop):
                    setattr(props, prop, tuple(int(round(v)) for v in value))
                else:
                    setattr(props, prop, tuple(float(v) for v in value))
            else:
                setattr(props, prop, value)
        except Exception as e:
            print(f"[pv.job] could not set {group}.{key} -> {prop}: {e}",
                  file=sys.stderr)

    # camera.miller : [h, k, l]
    cam = job.get("camera")
    if isinstance(cam, dict) and "miller" in cam:
        mh = cam["miller"]
        for pname, val in zip(_MILLER_PROPS, mh):
            try:
                setattr(props, pname, float(val))
            except Exception as e:
                print(f"[pv.job] could not set camera.miller {pname}: {e}", file=sys.stderr)

    # bonds : the embedded side-car spec (schema v2). NOT partial-job keyed:
    # absent -> CLEAR, so a v1 job (or a v2 job without curation) always
    # renders with default bonding regardless of prior GUI state.
    b = job.get("bonds")
    try:
        props.bonds_spec_json = json.dumps(b) if isinstance(b, dict) else ""
    except Exception as e:
        print(f"[pv.job] could not set bonds spec: {e}", file=sys.stderr)

    # appearance.species : { "Sn": {"color":[r,g,b], "radius":x}, ... }
    appearance = job.get("appearance")
    if isinstance(appearance, dict) and isinstance(appearance.get("species"), dict):
        species = appearance["species"]
        from .state import _SuppressSpeciesUpdates
        with _SuppressSpeciesUpdates():
            for entry in props.species_overrides:
                spec = species.get(entry.name)
                if not spec:
                    continue
                if "color" in spec:
                    try:
                        entry.color = tuple(spec["color"])
                    except Exception as e:
                        print(f"[pv.job] species {entry.name} color: {e}", file=sys.stderr)
                if "radius" in spec:
                    try:
                        entry.radius = float(spec["radius"])
                    except Exception as e:
                        print(f"[pv.job] species {entry.name} radius: {e}", file=sys.stderr)


def scene_to_job(scene) -> dict:
    """Read every prop into a job dict (the inverse of apply_job_to_scene).

    Powers the GUI Export button and guarantees the field list is complete and
    round-trippable. Also synthesizes the job-only output.path / output.animation
    so the exported job is directly runnable by tools/pv_inner.py.
    """
    import os
    props = scene.phonon_vision

    job: dict = {"schema_version": SCHEMA_VERSION}
    for group, key, prop, kind in _MAP:
        bucket = job.setdefault(group, {})
        val = getattr(props, prop)
        if kind == "vec":
            if _vec_elem_is_int(props, prop):
                bucket[key] = [int(x) for x in val]
            else:
                bucket[key] = [float(x) for x in val]
        else:
            bucket[key] = val

    # camera.miller as a list
    job.setdefault("camera", {})["miller"] = [
        float(getattr(props, p)) for p in _MILLER_PROPS
    ]

    # render.device / render.gpu_backend are headless-only (no GUI prop): emit
    # the safe defaults so an exported job is complete and explicit. CPU =
    # today's behavior; edit the job to "GPU"/"AUTO" for HPC / local-GPU runs.
    rend = job.setdefault("render", {})
    rend.setdefault("device", "CPU")
    rend.setdefault("gpu_backend", "AUTO")

    # species block
    species = {}
    for entry in props.species_overrides:
        species[entry.name] = {
            "color": [float(x) for x in entry.color],
            "radius": float(entry.radius),
        }
    job.setdefault("appearance", {})["species"] = species

    # engine block (schema v3): present iff the loaded source IS an engine.
    # Paths come RESOLVED from the EngineSource (never re-derived); q from
    # the live props; eigenvectors never serialized (FC = source of truth).
    if getattr(props, "data_kind", "") == "ENGINE":
        from .state import get_loaded_data
        data = get_loaded_data(scene)
        job["engine"] = {
            "source": "engine",
            "phonopy_yaml": getattr(data, "yaml_path", props.yaml_path),
            "force_constants": getattr(data, "fc_path", ""),
            "q_frac": [float(props.q_frac_1), float(props.q_frac_2),
                       float(props.q_frac_3)],
            "band_index": int(props.band_index),
            # path_s (0.8.5): OPTIONAL UI restoration hint for the walk slider.
            # q_frac stays authoritative on replay; path_s never drives the
            # evaluation and replay does not require the band sibling to exist.
            "path_s": float(props.path_s),
        }

    # bonds block (schema v2): embed the loaded side-car spec BY VALUE
    # (incl. its "source" provenance). No spec loaded -> no key.
    if getattr(props, "bonds_spec_json", ""):
        try:
            job["bonds"] = json.loads(props.bonds_spec_json)
        except Exception as e:
            print(f"[pv.job] could not embed bonds spec: {e}", file=sys.stderr)

    # output.path / output.animation — job-only fields consumed by pv_inner.
    out_dir = props.output_directory
    base = f"PV_q{props.q_index:04d}_b{props.band_index:02d}"
    fmt = props.output_format
    if fmt == "PNG":
        out_path = os.path.join(out_dir, f"{base}_")     # sequence prefix
    elif fmt == "GIF":
        out_path = os.path.join(out_dir, f"{base}.gif")
    else:
        out_path = os.path.join(out_dir, f"{base}.mp4")
    job["output"]["path"] = out_path
    job["output"]["animation"] = True   # GUI render is a one-cycle animation

    return job


def load_engine_source_from_job(job: dict):
    """Reconstruct the EngineSource an ENGINE job describes (bpy-free).

    The job carries PATHS + q, never eigenvectors: the FORCE_CONSTANTS file
    is the source of truth, so replay re-evaluates D(q) fresh — and must
    byte-match a direct evaluation (asserted by tools/test_job_roundtrip.py).
    Raises KeyError (no engine block), FileNotFoundError (paths gone — the
    named error), or the engine's own errors (BORN refusal etc.).
    """
    import os
    eng_blk = job["engine"]
    yaml_path = eng_blk["phonopy_yaml"]
    fc = eng_blk.get("force_constants") or None
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"engine job: phonopy_yaml not found: {yaml_path}")
    if fc and fc != "(embedded)" and not os.path.isfile(fc):
        raise FileNotFoundError(f"engine job: force_constants not found: {fc}")
    from . import motion
    src = motion.load_engine_source(yaml_path,
                                    None if fc == "(embedded)" else fc)
    src.set_q(tuple(float(x) for x in eng_blk.get("q_frac", (0.0, 0.0, 0.0))))
    return src
