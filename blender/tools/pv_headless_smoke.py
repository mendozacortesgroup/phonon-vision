"""Headless acceptance test for the Phonon Vision add-on (Phase 0).

Proves the headless path end-to-end: load -> build -> camera -> render a PNG,
with NO GUI window. Also the skeleton of Phase 1's inner script.

Usage:
    blender --background --python tools/pv_headless_smoke.py -- <band.yaml> <out.png>

Self-contained: it adds the add-on directory to sys.path and registers the
add-on directly, so no install step is needed.
"""
import os
import sys
import importlib

import bpy


def _args_after_ddash():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


def main():
    args = _args_after_ddash()
    if len(args) < 2:
        print("usage: blender --background --python tools/pv_headless_smoke.py "
              "-- <band.yaml> <out.png>")
        sys.exit(2)
    in_path, out_path = args[0], args[1]

    # Make the add-on importable as a package, no install step needed. This file
    # lives in <addon>/tools/, so the add-on dir is one level up and its PARENT
    # must be on sys.path. When shipped, the add-on dir is named 'phonon_vision'
    # (the manifest id), so `pkg` resolves to that and matches the documented
    # `from phonon_vision import api`.
    addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent = os.path.dirname(addon_dir)
    pkg = os.path.basename(addon_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    addon = importlib.import_module(pkg)
    if hasattr(addon, "register"):
        try:
            addon.register()
        except Exception as e:
            # Already registered (e.g. installed) is fine; anything else is real.
            print(f"[smoke] register() note: {e}")

    scene = bpy.context.scene

    api = importlib.import_module(f"{pkg}.api")   # == `from phonon_vision import api`
    print(f"[smoke] loading {in_path}")
    api.load_file(scene, in_path)
    api.build_scene_from_data(scene)
    api.apply_camera(scene, "ISO")            # any preset; just proving the path

    # Keep it fast: low Cycles samples (EEVEE would also be fine).
    try:
        scene.cycles.samples = 16
    except Exception:
        pass

    print(f"[smoke] rendering still -> {out_path}")
    resolved = api.render_headless(scene, animation=False, output_path=out_path)
    print(f"[smoke] OK wrote: {resolved}")


if __name__ == "__main__":
    main()
