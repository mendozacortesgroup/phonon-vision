"""phonon_vision.api — the headless-safe contract.

A thin re-export of the entry points that work without a GUI window, so the
HPC CLI's inner script (Phase 1) imports one clean surface instead of reaching
into individual modules. Pure re-export — NO logic lives here.

All of these are driven by ``scene.phonon_vision`` props and are safe to call
under ``blender --background``:

    load_file(scene, path)                  -> load phonopy data (no scene build)
    build_scene_from_data(scene)            -> build atoms/arrows/cell
    apply_camera(scene, view)               -> place the camera
    sync_appearance(scene)                  -> write current appearance to materials
    render_headless(scene, *, animation=…)  -> synchronous, background-safe render
    set_render_device(scene, device, backend) -> select CPU/GPU (headless-only)
    apply_preset(scene, preset)             -> apply a render preset (engine/world/…)
"""
from __future__ import annotations

from .operators import load_file, apply_camera, render_headless, set_render_device
from .scene_setup import build_scene_from_data, sync_appearance
from .render_presets import apply_preset

__all__ = [
    "load_file",
    "build_scene_from_data",
    "apply_camera",
    "sync_appearance",
    "render_headless",
    "set_render_device",
    "apply_preset",
]
