"""Mode info overlay — burns 'Mode N | q=(...) | ν=X.XXX THz' into renders.

We use Blender's built-in render stamp feature rather than the compositor's
text-image approach because stamps are simpler, work in every render engine
(Eevee, Eevee Next, Cycles), and survive across all output formats (PNG,
EXR, MP4, GIF, FFmpeg sequences).

The stamp is invisible in the viewport — that's the trade-off — but the Mode
Navigator panel already shows live mode info there, so there's no loss.

This module exposes a single function ``refresh_overlay_stamp(scene)`` which
the state callbacks invoke whenever:

- The user toggles ``show_mode_overlay``,
- The user changes the overlay's color/size/background,
- The user navigates to a different mode (q-point or band).
"""

from __future__ import annotations

import bpy

from .state import get_loaded_data
from phonon_core.units import format_frequency


def refresh_overlay_stamp(scene) -> None:
    """Configure scene.render.stamp_* based on ``scene.phonon_vision.show_mode_overlay``.

    When the overlay is on, all default-on stamps (time, date, frame number,
    camera, scene name, render time, lens info, filename, marker) are turned
    OFF — we only want the custom note we set, not Blender's metadata clutter.
    When off, ``use_stamp`` is turned off so nothing is burnt into the render.
    """
    props = scene.phonon_vision

    if not props.show_mode_overlay:
        scene.render.use_stamp = False
        return

    # Enable stamping and disable every default stamp source — only the
    # custom note should appear.
    r = scene.render
    r.use_stamp = True
    r.use_stamp_note = True
    for attr in (
        "use_stamp_time", "use_stamp_date", "use_stamp_frame",
        "use_stamp_camera", "use_stamp_scene", "use_stamp_render_time",
        "use_stamp_lens", "use_stamp_filename", "use_stamp_marker",
        "use_stamp_memory", "use_stamp_hostname", "use_stamp_sequencer_strip",
        "use_stamp_frame_range",
    ):
        if hasattr(r, attr):
            try:
                setattr(r, attr, False)
            except Exception:
                pass

    # Build the note text from the current mode.
    data = get_loaded_data(scene)
    if data is None:
        # No data loaded — clear text to avoid showing a stale stamp.
        r.stamp_note_text = ""
    else:
        try:
            q = int(props.q_index)
            b = int(props.band_index)
            mode = data.get_mode(q, b)
            freq_txt = format_frequency(mode.frequency, "THz", 3)
            r.stamp_note_text = (
                f"Mode {b + 1}  |  "
                f"q=({mode.q_point[0]:.3f}, {mode.q_point[1]:.3f}, {mode.q_point[2]:.3f})  |  "
                f"ν={freq_txt}"
            )
        except Exception as e:
            print(f"[phonon-vision] overlay note build failed: {e}")
            r.stamp_note_text = ""

    # Appearance: foreground = (r,g,b,1.0), background = (r,g,b,a).
    fg = props.overlay_text_color
    bg = props.overlay_background_color
    try:
        r.stamp_foreground = (float(fg[0]), float(fg[1]), float(fg[2]), 1.0)
        r.stamp_background = (float(bg[0]), float(bg[1]), float(bg[2]), float(bg[3]))
    except Exception:
        pass

    # Font size in pixels.
    try:
        r.stamp_font_size = int(props.overlay_text_size)
    except Exception:
        pass
