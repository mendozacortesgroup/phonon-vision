"""Phonon Vision — headless job orchestrator (Phase 1).

    blender --background --python tools/pv_inner.py -- --job /path/to/job.json

NO --factory-startup, and NO second copy of the add-on: this RELIES on the
already-installed, enabled extension (Blender loads it at startup, so all
bundled wheels — phonon_core, PyYAML, h5py — are already on sys.path) and just
DRIVES it. It discovers the add-on package at runtime rather than hardcoding
'bl_ext.user_default.phonon_vision'.
"""
import sys
import os
import json
import importlib

import bpy


def _args_after_ddash():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def _parse_job_path(args):
    if "--job" in args:
        i = args.index("--job")
        if i + 1 < len(args):
            return args[i + 1]
    return None


def _discover_pkg():
    """Find the installed add-on package (exposing api + job) WITHOUT hardcoding
    its dotted name. Blender loads enabled extensions at startup as
    e.g. bl_ext.<repo>.phonon_vision (or 'phonon_vision' for a legacy add-on)."""
    candidates = []
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if (name == "phonon_vision" or name.endswith(".phonon_vision")) \
                and getattr(mod, "__path__", None):
            candidates.append(name)
    # Prefer a candidate whose .api imports cleanly.
    for name in candidates:
        try:
            api = importlib.import_module(name + ".api")
            job = importlib.import_module(name + ".job")
            return name, api, job
        except Exception:
            continue
    # Last resort: a plain top-level install.
    try:
        api = importlib.import_module("phonon_vision.api")
        job = importlib.import_module("phonon_vision.job")
        return "phonon_vision", api, job
    except Exception:
        return None, None, None


def _clear_factory_objects():
    """HEADLESS-ONLY: remove Blender's default startup objects (Cube/Light/
    Camera) before the add-on builds, so they don't leak into the render.

    Keeps every PV.* object — the add-on builds its own mesh, arrows, bonds,
    supercell box, camera (PV.CameraObj) and lights (PV.KeyLight, …). At call
    time (before build_scene_from_data) the scene contains only the factory
    trio, so the rule "remove pre-existing non-PV objects" clears exactly those.
    Each removal is guarded; a scene that's already clean is a no-op.
    """
    removed = []
    for obj in list(bpy.data.objects):
        if obj.name.startswith("PV."):
            continue   # never touch add-on objects
        try:
            name = obj.name
            bpy.data.objects.remove(obj, do_unlink=True)
            removed.append(name)
        except Exception as e:
            print(f"[pv] could not remove default object {obj.name!r}: {e}",
                  file=sys.stderr)
    if removed:
        print(f"[pv] cleared default startup objects: {', '.join(removed)}")


def main():
    args = _args_after_ddash()
    job_path = _parse_job_path(args)
    if not job_path:
        print("[pv] ERROR: pass --job <path/to/job.json> after '--'")
        sys.exit(2)

    print(f"[pv] reading job: {job_path}")
    with open(job_path, "r") as fh:
        job = json.load(fh)

    pkg, api, jobmod = _discover_pkg()
    if api is None:
        print("[pv] ERROR: could not find the installed Phonon Vision extension. "
              "Install + enable it in this Blender first (do NOT use "
              "--factory-startup).")
        sys.exit(1)
    print(f"[pv] driving installed extension: {pkg}")

    scene = bpy.context.scene

    # 4. load data (resets indices, populates species DEFAULTS)
    src = job["input"]["file"]
    print(f"[pv] load_file: {src}")
    api.load_file(scene, src)

    # 5. apply job AFTER load → overrides indices, species, supercell, mesh
    #    quality, amplitude, display, camera, render, overlay, loop props.
    print("[pv] apply_job_to_scene")
    jobmod.apply_job_to_scene(scene, job)

    # 5a. curated bonds (schema v2): the spec now sits in props.bonds_spec_json
    #     and the scene build below applies it. Log it so batch logs show
    #     whether a render used curated or default bonding.
    b = job.get("bonds")
    if isinstance(b, dict) and isinstance(b.get("bonds"), dict):
        src_info = b.get("source") or {}
        origin = f" (source: {src_info.get('filename')})" \
            if isinstance(src_info, dict) and src_info.get("filename") else ""
        print(f"[pv] curated bonds: {len(b['bonds'])} pair(s){origin}")

    # 5b. HEADLESS-ONLY: clear Blender's factory startup objects (Cube/Light/
    #     Camera) so they don't leak into the render. Runs BEFORE the build, so
    #     no PV object (mesh, arrows, bonds, supercell box, camera, lights) exists
    #     yet — only the factory trio can be removed here. The GUI never calls this.
    _clear_factory_objects()

    # 6. build geometry using the now-job-set build-time props
    print("[pv] build_scene_from_data")
    api.build_scene_from_data(scene)

    # 7. push species colors / vibrancy / arrow color to materials
    print("[pv] sync_appearance")
    api.sync_appearance(scene)

    # 8. frame the camera (needs the built bbox)
    cam_mode = job.get("camera", {}).get("mode", scene.phonon_vision.camera_view)
    print(f"[pv] apply_camera: {cam_mode}")
    api.apply_camera(scene, cam_mode)

    # 9. push render style / lighting / samples / engine from props
    print(f"[pv] apply_preset: {scene.phonon_vision.render_preset}")
    api.apply_preset(scene, scene.phonon_vision.render_preset)

    # 9b. select the render device (headless-only; CPU default = unchanged).
    rend = job.get("render", {})
    dev = rend.get("device", "CPU")
    back = rend.get("gpu_backend", "AUTO")
    summary = api.set_render_device(scene, dev, back)
    print(f"[pv] render device: {summary}")

    # Gotchas apply_preset may not cover — set explicitly from props.
    props = scene.phonon_vision
    try:
        scene.render.film_transparent = bool(props.transparent_background)
        scene.render.resolution_x = int(props.render_resolution_x)
        scene.render.resolution_y = int(props.render_resolution_y)
        scene.render.resolution_percentage = int(props.render_percentage)
    except Exception as e:
        print(f"[pv] note: could not set film/resolution explicitly: {e}")

    # 10. render
    out = job["output"]
    animation = bool(out.get("animation", False))
    out_path = out.get("path") or ""

    if animation:
        # Frames-first: render ONE seamless cycle as a PNG sequence (the outer
        # launcher assembles mp4/gif with ffmpeg). Use the SAME one-loop range
        # the GUI playback uses, so the seam frame isn't duplicated.
        from phonon_core.animation import recommended_render_settings
        props = scene.phonon_vision
        rs = recommended_render_settings(
            frames_per_cycle=int(props.frames_per_cycle),
            fps=int(props.fps_render),
            frame_start=1,
        )
        scene.frame_start = rs.frame_start
        scene.frame_end = rs.frame_end
        scene.render.fps = rs.fps
        n_frames = rs.frame_end - rs.frame_start + 1
        # Guard a video-extension path (Blender would mangle it into
        # 'name.mp4####.png'): render into a clean frames dir + 'frame_' prefix.
        stem, ext = os.path.splitext(out_path)
        if ext.lower() in (".mp4", ".gif", ".mov", ".mkv", ".webm", ".avi"):
            seq_prefix = os.path.join(stem + "_frames", "frame_")
        else:
            seq_prefix = out_path or "frame_"
        frames_dir = os.path.dirname(seq_prefix) or "."
        os.makedirs(frames_dir, exist_ok=True)
        print(f"[pv] animation: {n_frames} frames (1 cycle) @ {rs.fps}fps")
        api.render_headless(scene, animation=True, output_path=seq_prefix,
                            file_format="PNG")
        print(f"[pv] rendered {n_frames} frames -> {frames_dir}")
    else:
        out_fmt = out.get("format")
        print(f"[pv] render_headless: still -> {out_path}")
        written = api.render_headless(scene, animation=False,
                                      output_path=out_path, file_format=out_fmt)
        print(f"[pv] OK wrote: {written}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[pv] ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
