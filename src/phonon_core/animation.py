"""Animation utilities for seamlessly looping phonon visualizations.

Why this module exists
----------------------
A phonon mode is a periodic oscillation: at phase φ + 2π the displacement
is identical to at phase φ. To render a loop with no stutter or duplicated
frame, the rendered frames must sample phases in ``[0, 2π)`` — *not*
``[0, 2π]`` — so that the wrap from the last frame back to the first
advances by exactly one phase step.

The Blender frame handler in the addon does this implicitly with::

    phase = (frame % frames_per_cycle) / frames_per_cycle * 2π

At ``frame = frames_per_cycle``, the modulo wraps to 0, exactly the same as
``frame = 0``. So if the user renders frames 0..frames_per_cycle-1 (i.e.,
exactly frames_per_cycle frames) and loops them, the boundary is seamless.

Common off-by-one mistakes
--------------------------
1. Rendering frames 0..N inclusive (= N+1 frames) gives a duplicate first/last.
2. Sampling phases with ``np.linspace(0, 2π, N)`` (endpoint=True) also gives
   a duplicate. Use ``endpoint=False`` or equivalently sample N frames in
   ``[0, 2π)``.
3. Mixing frame ranges and phase counts: if Blender's scene runs from
   ``frame_start`` to ``frame_end`` inclusive, the rendered count is
   ``frame_end - frame_start + 1``. So to get N frames per cycle, set
   ``frame_end = frame_start + N - 1`` and ``frames_per_cycle = N``.

Use :func:`recommended_render_settings` to get the right Blender scene
parameters for a given desired loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def phase_grid(n_frames: int, endpoint: bool = False) -> np.ndarray:
    """Phases for ``n_frames`` evenly spaced samples in one loop.

    With ``endpoint=False`` (default) the samples are::

        0, 2π/N, 2·2π/N, ..., (N-1)·2π/N

    so wrapping back to 0 advances by 2π/N — the same step as between any
    two consecutive frames. This is what you want for a seamless GIF.

    With ``endpoint=True`` you get N samples in ``[0, 2π]``, where the last
    frame is identical to the first. Use this only if you specifically need
    that property (almost no one does).
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be ≥ 1; got {n_frames}")
    return np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=endpoint)


def frame_to_phase(
    frame_current: int,
    frames_per_cycle: int,
    *,
    frame_start: int = 0,
) -> float:
    """Compute the global phase φ for one specific frame in a periodic loop.

    Parameters
    ----------
    frame_current : int
        The current Blender (or other) frame number.
    frames_per_cycle : int
        How many frames make one full oscillation.
    frame_start : int
        Frame index treated as φ = 0 (default 0). Useful if your scene's
        timeline starts at frame 1 (Blender's default).

    Returns
    -------
    float
        Phase in radians, in ``[0, 2π)``.
    """
    if frames_per_cycle < 1:
        raise ValueError(f"frames_per_cycle must be ≥ 1; got {frames_per_cycle}")
    rel = (frame_current - frame_start) % frames_per_cycle
    return rel / frames_per_cycle * 2.0 * np.pi


@dataclass(frozen=True)
class RenderSettings:
    """Blender-compatible scene settings for a seamless loop render.

    Attributes
    ----------
    frame_start : int
        Set as ``scene.frame_start``.
    frame_end : int
        Set as ``scene.frame_end``. The total rendered frame count is
        ``frame_end - frame_start + 1``.
    fps : int
        Set as ``scene.render.fps``.
    frames_per_cycle : int
        How many frames represent one full oscillation. Equal to
        ``frame_end - frame_start + 1`` for a single-cycle loop.
    cycle_duration_seconds : float
        Wall-clock duration of one loop = ``frames_per_cycle / fps``.
    """

    frame_start: int
    frame_end: int
    fps: int
    frames_per_cycle: int
    cycle_duration_seconds: float


def recommended_render_settings(
    *,
    frames_per_cycle: int = 60,
    fps: int = 30,
    frame_start: int = 1,
) -> RenderSettings:
    """Recommended Blender scene settings for a seamless single-cycle loop.

    Defaults give a 2-second loop at 30 fps with 60 frames per cycle — a
    sweet spot for GIF quality and file size on most phonon modes.

    Notes
    -----
    *Why ``frame_start = 1``:* Blender's default scene starts at frame 1
    rather than 0. Setting it to 1 means rendering "all frames" (Blender's
    UI default) covers exactly one cycle.

    The frame at ``frame_end`` has phase ``(frames_per_cycle - 1) / frames_per_cycle * 2π``.
    The next frame, ``frame_end + 1`` (if it existed), would have phase 2π,
    which is identical to phase 0 at ``frame_start``. So when a player loops
    ``frame_end → frame_start``, the transition steps by exactly 2π/frames_per_cycle —
    just like every other inter-frame transition. No stutter.
    """
    if frames_per_cycle < 2:
        raise ValueError(f"frames_per_cycle must be ≥ 2; got {frames_per_cycle}")
    if fps < 1:
        raise ValueError(f"fps must be ≥ 1; got {fps}")
    return RenderSettings(
        frame_start=frame_start,
        frame_end=frame_start + frames_per_cycle - 1,
        fps=fps,
        frames_per_cycle=frames_per_cycle,
        cycle_duration_seconds=frames_per_cycle / fps,
    )
