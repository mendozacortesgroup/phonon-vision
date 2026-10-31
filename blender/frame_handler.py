"""Frame change handler — updates atom positions every frame.

Registers a callback on bpy.app.handlers.frame_change_pre that recomputes
displacements using the current frame's phase. Loop seamlessness comes from
phonon_core.animation.frame_to_phase wrapping correctly at frame == frames_per_cycle.
"""

from __future__ import annotations

import bpy

from .scene_setup import update_atom_positions, sync_appearance
from .state import _drain_redraws


def update_frame(scene) -> None:
    """Per-frame update — called both by the handler and by the UI when
    amplitude changes (to re-sync the current frame's visual)."""
    update_atom_positions(scene)


@bpy.app.handlers.persistent
def frame_change_handler(scene, depsgraph=None):
    """Called by Blender every frame change (also during playback / render)."""
    try:
        update_frame(scene)
    except Exception as e:
        print(f"[phonon-vision] frame update failed: {e}")
    # Drive appearance writes from the frame handler — the only thing guaranteed
    # to run every frame during playback (property callbacks / timers may not
    # surface mid-playback). Material writes ONLY (Base Color + Emission +
    # diffuse_color + update_tag); no object replacement, no node rebuild. The
    # depsgraph eval for this frame then shows the current colors.
    try:
        sync_appearance(scene)
    except Exception as e:
        print(f"[phonon-vision] appearance sync failed: {e}")
    # Static-case machinery retained: drain any pending redraw kicks once
    # (redraw kicks only — never object replacement).
    try:
        _drain_redraws()
    except Exception as e:
        print(f"[phonon-vision] redraw drain failed: {e}")


def register_handler() -> None:
    # Idempotent: remove any old reference first
    unregister_handler()
    bpy.app.handlers.frame_change_pre.append(frame_change_handler)


def unregister_handler() -> None:
    for h in list(bpy.app.handlers.frame_change_pre):
        if getattr(h, "__name__", "") == "frame_change_handler":
            bpy.app.handlers.frame_change_pre.remove(h)
