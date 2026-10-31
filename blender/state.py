"""Scene state: holds the loaded phonon data and the active mode.

The state lives on the Scene as a PropertyGroup, plus a module-level
runtime cache (the loaded BandPath/MeshGrid objects, which aren't
serializable into Blender properties).

NOTE: this module intentionally does NOT use ``from __future__ import
annotations``. The PropertyGroup ``update=lambda self, ctx: _on_x(ctx)``
callbacks must be REAL lambdas whose ``__globals__`` is this module's namespace,
so they can resolve the module-level ``_on_*`` functions at call time. Under
PEP 563 the annotations become strings that are eval'd at registration in a
namespace lacking ``_on_*`` → the lambdas raise ``NameError: name '_on_*' is
not defined`` (filename ``<string>``) every time a prop is set. (Fixed in 0.7.1.)
"""

from typing import Optional, Union

import bpy
from bpy.props import (
    StringProperty, IntProperty, FloatProperty, BoolProperty,
    EnumProperty, PointerProperty, FloatVectorProperty, IntVectorProperty,
)

# Runtime cache — keyed by the scene's name, value is the loaded BandPath/MeshGrid
# (these contain numpy arrays which can't be stored as Blender properties).
_loaded_data: dict[str, object] = {}


def get_loaded_data(scene) -> Optional[object]:
    return _loaded_data.get(scene.name)


def set_loaded_data(scene, data) -> None:
    _loaded_data[scene.name] = data


def clear_loaded_data(scene) -> None:
    _loaded_data.pop(scene.name, None)


class PVSpeciesEntry(bpy.types.PropertyGroup):
    """Per-species visual overrides (color + radius).

    One entry per chemical species in the loaded material. Populated on YAML
    load with sensible defaults derived from atomic mass and number of species.
    The user can then customize via the Display panel — these become the
    single source of truth for both atom spheres and bond half-colors.
    """

    # `name` is inherited from PropertyGroup (used as the species symbol like "Sn")

    color: FloatVectorProperty(
        name="Color",
        description="Sphere color for this species. Also used for the bond half "
                    "anchored at atoms of this species.",
        subtype="COLOR",
        size=3,
        min=0.0, max=1.0,
        default=(0.5, 0.5, 0.5),
        update=lambda self, ctx: _on_species_change(ctx),
    )  # type: ignore
    radius: FloatProperty(
        name="Radius (Å)",
        description="Visual sphere radius for this species. Default is a mass-"
                    "scaled hint; override for clearer publication figures.",
        default=0.35, min=0.05, soft_max=3.0,
        update=lambda self, ctx: _on_species_change(ctx),
    )  # type: ignore


class PhononVisionProps(bpy.types.PropertyGroup):
    """Per-scene phonon visualization settings."""

    # --- Loaded data ---
    yaml_path: StringProperty(
        name="Data File",
        description="Path to a phonopy band.yaml/band.hdf5 or mesh.yaml/mesh.hdf5 file. "
                    "HDF5 is roughly 100× faster to load for large calculations",
        subtype="FILE_PATH",
        default="",
    )  # type: ignore
    data_kind: EnumProperty(
        name="Data Kind",
        items=[
            ("NONE",  "None loaded", "No data loaded"),
            ("BAND",  "Band path",  "1D path through q-space (from band.yaml)"),
            ("MESH",  "Mesh grid",  "3D q-grid (from mesh.yaml)"),
            ("ENGINE", "Engine (A2)",
             "Exact D(q) at ANY q from phonopy.yaml + FORCE_CONSTANTS "
             "(vendored pure-numpy engine; live q-scrubbing)"),
        ],
        default="NONE",
    )  # type: ignore

    # --- Mode selection ---
    # Internal indices are always 0-based (matches phonon-core).
    # UI displays them +1 via the panel labels.
    q_index: IntProperty(
        name="q-point index",
        description="Index of the current q-point in the loaded data (1-based in UI)",
        min=0, default=0,
        update=lambda self, ctx: _on_mode_change(ctx),
    )  # type: ignore
    band_index: IntProperty(
        name="Band index",
        description="Phonon branch index (1-based in UI; 1 = lowest acoustic, 3N = highest optical)",
        min=0, default=0,
        update=lambda self, ctx: _on_mode_change(ctx),
    )  # type: ignore

    # --- Engine mode (0.8.0): the LIVE fractional q (r.l.u.) ---
    # Dragging any component re-evaluates D(q) via the vendored A2 engine
    # (~0.07 ms/evaluation, architect-measured) and refreshes the mode
    # readout + animation source CONTINUOUSLY — no Evaluate button.
    q_frac_1: FloatProperty(
        name="q1",
        description="Fractional q, first reciprocal-lattice component (r.l.u.)",
        default=0.0, soft_min=-1.0, soft_max=1.0, step=1, precision=4,
        update=lambda self, ctx: _on_engine_q_change(ctx),
    )  # type: ignore
    q_frac_2: FloatProperty(
        name="q2",
        description="Fractional q, second reciprocal-lattice component (r.l.u.)",
        default=0.0, soft_min=-1.0, soft_max=1.0, step=1, precision=4,
        update=lambda self, ctx: _on_engine_q_change(ctx),
    )  # type: ignore
    q_frac_3: FloatProperty(
        name="q3",
        description="Fractional q, third reciprocal-lattice component (r.l.u.)",
        default=0.0, soft_min=-1.0, soft_max=1.0, step=1, precision=4,
        update=lambda self, ctx: _on_engine_q_change(ctx),
    )  # type: ignore

    # --- Path walking (0.8.5): one slider carries q along the band file's
    # declared path, with EXACT D(q) at every position (continuum evaluation,
    # not precomputed samples). Only active for an engine source that has a
    # band path (a same-material sibling band file). ---
    path_s: FloatProperty(
        name="Walk s",
        description="Position along the band file's declared path (0 = start, "
                    "1 = end). Drag to carry q CONTINUOUSLY, with exact D(q) "
                    "evaluated at every point. CAVEAT: Band N is the Nth SORTED "
                    "eigenvalue at each q — walking through a band crossing "
                    "follows the sort order, not the connected band "
                    "(band-following is future work).",
        default=0.0, min=0.0, max=1.0, precision=4,
        update=lambda self, ctx: _on_path_s_changed(ctx),
    )  # type: ignore

    # --- Animation / display ---
    amplitude: FloatProperty(
        name="Amplitude",
        description="Visual amplitude multiplier — eigenvectors are tiny, "
                    "typically use 3-10 (large values produce unphysically big motion)",
        default=4.0, min=0.0, soft_max=50.0,
        update=lambda self, ctx: _on_anim_change(ctx),
    )  # type: ignore
    frames_per_cycle: IntProperty(
        name="Frames per cycle",
        description="How many frames make one full oscillation. Loop is "
                    "seamless when scene frame_end - frame_start + 1 = this",
        default=60, min=2, soft_max=240,
    )  # type: ignore
    fps_render: IntProperty(
        name="FPS",
        description="Frames per second for the render",
        default=30, min=1, soft_max=120,
    )  # type: ignore

    # --- Supercell replication ---
    supercell_n: IntVectorProperty(
        name="Supercell N×M×K",
        description="How many unit cells to replicate along each lattice vector. "
                    "Larger supercells reveal long-wavelength modes but are slower "
                    "to render. Triggers a scene rebuild when changed.",
        size=3,
        default=(3, 3, 3),
        min=1, soft_max=20,
        update=lambda self, ctx: _on_supercell_change(ctx),
    )  # type: ignore

    # --- Per-species overrides (color + radius) — populated on YAML load ---
    species_overrides: bpy.props.CollectionProperty(
        type=PVSpeciesEntry,
        name="Species overrides",
    )  # type: ignore

    species_vibrancy: FloatProperty(
        name="Vibrancy",
        description="Self-emission strength of atom (and per-species bond) "
                    "materials in Material Preview / Rendered shading. Higher "
                    "values keep the species color saturated under bright "
                    "HDRI lighting; lower values give flatter shading with "
                    "more shadow detail. Affects appearance only, not physics.",
        default=0.30, min=0.0, max=1.0,
        update=lambda self, ctx: _on_species_change(ctx),
    )  # type: ignore

    arrow_color: FloatVectorProperty(
        name="Arrow color",
        description="Color of the displacement arrows. Applies live to all "
                    "arrows (one global color); appearance only, not physics.",
        subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(0.90, 0.10, 0.10),
        update=lambda self, ctx: _on_arrow_color_change(ctx),
    )  # type: ignore

    # --- Visuals ---
    show_arrows: BoolProperty(
        name="Show arrows",
        description="Draw displacement arrows on each atom",
        default=True,
        update=lambda self, ctx: _on_visual_change(ctx),
    )  # type: ignore
    show_unit_cell: BoolProperty(
        name="Show unit cell",
        description="Draw the cell outline",
        default=True,
        update=lambda self, ctx: _on_visual_change(ctx),
    )  # type: ignore
    unit_cell_kind: EnumProperty(
        name="Cell outline",
        items=[
            ("PRIMITIVE", "Primitive only",
             "Outline only the primitive unit cell at the origin (the original cell)"),
            ("GRID",      "All cells (grid)",
             "Outline every individual unit cell of the supercell, showing internal grid lines"),
            ("SUPERCELL", "Supercell box",
             "Outline only the outer supercell boundary (a single bounding box)"),
        ],
        default="SUPERCELL",
        update=lambda self, ctx: _on_visual_change(ctx),
    )  # type: ignore
    cell_thickness: FloatProperty(
        name="Cell edge thickness ×",
        description="Multiplier for cell edge thickness. Auto-base is "
                    "scaled to lattice size; this multiplies that base. "
                    "VESTA-style — use 0.5 for thin, 2-3 for very thick.",
        default=1.0, min=0.05, soft_max=10.0,
        update=lambda self, ctx: _on_visual_change(ctx),
    )  # type: ignore
    show_mode_overlay: BoolProperty(
        name="Burn mode info into render",
        description="Stamp 'Mode N | q=(...) | ν=X.XXX THz' into the corner of "
                    "each rendered frame using Blender's stamp feature. Visible "
                    "in MP4/GIF/PNG output (not in the viewport — use the Mode "
                    "Navigator card there).",
        default=False,
        update=lambda self, ctx: _on_overlay_change(ctx),
    )  # type: ignore
    overlay_text_size: IntProperty(
        name="Overlay text size",
        description="Font size of the burnt-in overlay text (pixels).",
        default=24, min=8, soft_max=96,
        update=lambda self, ctx: _on_overlay_change(ctx),
    )  # type: ignore
    overlay_text_color: FloatVectorProperty(
        name="Overlay color",
        description="Text color of the burnt-in overlay.",
        subtype="COLOR", size=3, min=0.0, max=1.0,
        default=(0.0, 0.0, 0.0),
        update=lambda self, ctx: _on_overlay_change(ctx),
    )  # type: ignore
    overlay_background_color: FloatVectorProperty(
        name="Overlay background",
        description="Background plate behind the overlay text (set alpha 0 to disable).",
        subtype="COLOR", size=4, min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0, 0.7),
        update=lambda self, ctx: _on_overlay_change(ctx),
    )  # type: ignore
    show_overlay: BoolProperty(
        name="(deprecated)",
        description="Deprecated — kept for project file compatibility. "
                    "Not shown in the UI; do not key or reference.",
        default=False,
        options={"HIDDEN"},
    )  # type: ignore

    imag_freq_threshold: FloatProperty(
        name="Imaginary threshold (THz)",
        description="Hide the 'imaginary' badge for modes whose |frequency| "
                    "is below this threshold. Phonopy can produce small "
                    "numerical-noise imaginary frequencies near Γ that aren't "
                    "physically meaningful; raise this to 0.05-0.1 THz to "
                    "suppress them.",
        default=0.01, min=0.0, soft_max=1.0,
    )  # type: ignore

    arrow_scale: FloatProperty(
        name="Arrow scale",
        description="Multiplier for arrow length. The arrow at each atom is "
                    "the current displacement direction times this factor. "
                    "Bump up (5-15) for clearly visible arrows.",
        default=5.0, min=0.0, soft_max=30.0,
        update=lambda self, ctx: _on_anim_change(ctx),
    )  # type: ignore
    use_canonical_phase: BoolProperty(
        name="Canonical phase",
        description="Rephase eigenvectors so the largest component is real at phase 0 — "
                    "makes snapshots reproducible across phonopy runs",
        default=True,
    )  # type: ignore

    # --- Bonds ---
    show_bonds: BoolProperty(
        name="Show bonds",
        description="Draw cylinders between atoms that are within covalent-bond distance",
        default=True,
        update=lambda self, ctx: _on_visual_change(ctx),
    )  # type: ignore
    bond_tolerance: FloatProperty(
        name="Bond tolerance",
        description="Bond cutoff is (R_i + R_j) × this. Use 1.0-1.3 for typical "
                    "covalent crystals; raise to 1.5-1.8 for ionic crystals where "
                    "you want to see ionic 'bonds'.",
        default=1.2, min=0.5, soft_max=2.5,
        update=lambda self, ctx: _on_pbc_change(ctx),
    )  # type: ignore
    bond_thickness: FloatProperty(
        name="Bond thickness",
        description="Bond cylinder thickness multiplier (1.0 = thin, "
                    "3.0 = chunky and easy to see).",
        default=3.0, min=0.05, soft_max=10.0,
        update=lambda self, ctx: _on_anim_change(ctx),
    )  # type: ignore
    bond_two_color: BoolProperty(
        name="Two-color bonds (per species)",
        description="Split each bond at its midpoint with the species color at "
                    "each end. Like VESTA — each half cylinder takes its "
                    "color from the atom it touches.",
        default=True,
        update=lambda self, ctx: _on_anim_change(ctx),
    )  # type: ignore
    bond_periodic: BoolProperty(
        name="Show atoms & bonds across PBC",
        description="When ON, adds image atoms in a one-unit-cell halo "
                    "around the supercell so PBC bonds terminate on "
                    "visible spheres. When OFF, only atoms strictly inside "
                    "the supercell are drawn and bonds that cross the "
                    "boundary are hidden — the 'as-is' view of the chosen "
                    "supercell. Triggers a scene rebuild because it "
                    "changes the atom slots.",
        default=False,
        update=lambda self, ctx: _on_pbc_change(ctx),
    )  # type: ignore

    # --- Per-frame bond visibility (stretch / compress thresholds) ---
    # Bonds are pre-computed from equilibrium atom positions. During the
    # phonon animation, atoms move; if a bond stretches beyond
    # max_factor × equilibrium_length, the visualization should hide that
    # bond (otherwise you see absurdly long bonds during large-amplitude
    # modes). Similarly for compression, though that's rarely triggered.
    # Hiding is implemented by collapsing the bond's cylinder verts to a
    # point — same trick as invisible arrows. No topology change per frame.
    bond_break_max_factor: FloatProperty(
        name="Max. bond distance",
        description="Hide the bond when its current length exceeds this "
                    "multiple of its equilibrium length. 1.5 means 'break "
                    "when the bond is 50% longer than at rest'. Set to 100 "
                    "to effectively disable the stretch break.",
        default=1.5, min=1.0, soft_max=5.0,
        update=lambda self, ctx: _on_anim_change(ctx),
    )  # type: ignore
    bond_break_min_factor: FloatProperty(
        name="Min. bond distance",
        description="Hide the bond when its current length falls below this "
                    "multiple of its equilibrium length. 0.5 means 'break "
                    "when the bond shrinks below 50% of rest length'. Rarely "
                    "triggered in real phonon modes. Set to 0 to disable.",
        default=0.5, min=0.0, soft_max=1.0,
        update=lambda self, ctx: _on_anim_change(ctx),
    )  # type: ignore

    # --- Curated bonds (navigator side-car, 0.7.9) ---
    # The FULL normalized "phonon-vision-bonds" spec as JSON (including the
    # provenance "source" block added at import), or "" for default bonding.
    # While non-empty, the spec REPLACES the tolerance criterion entirely
    # (bonds.build_bond_list). No update callback: the import/clear operators
    # rebuild explicitly, and job apply is followed by an explicit build.
    bonds_spec_json: StringProperty(
        name="Curated bonds spec",
        description="Imported bond side-car (JSON); empty = covalent default",
        default="",
        options={"HIDDEN"},
    )  # type: ignore

    # --- Amplitude clamping (keeps atoms from flying across the cell) ---
    clamp_amplitude: BoolProperty(
        name="Clamp displacement",
        description="If on, per-atom displacement is capped at "
                    "``max_displacement_angstroms`` so atoms don't fly across "
                    "the cell when amplitude is large. Useful for very high "
                    "amplitudes or near-degenerate modes.",
        default=False,
        update=lambda self, ctx: _on_anim_change(ctx),
    )  # type: ignore
    max_displacement: FloatProperty(
        name="Max displacement (Å)",
        description="Cap on per-atom displacement when 'Clamp displacement' is on.",
        default=0.8, min=0.05, soft_max=5.0,
        update=lambda self, ctx: _on_anim_change(ctx),
    )  # type: ignore

    # --- Camera (manual override) ---
    camera_auto_frame: BoolProperty(
        name="Auto-frame",
        description="If on, framing is computed automatically from atom bbox. "
                    "If off, you can manually set the camera ortho width and distance.",
        default=True,
    )  # type: ignore
    camera_ortho_scale: FloatProperty(
        name="Frame width (Å)",
        description="Manual horizontal extent of the camera frame in Å. "
                    "Only used when Auto-frame is off.",
        default=15.0, min=0.5, soft_max=200.0,
    )  # type: ignore
    camera_distance: FloatProperty(
        name="Camera distance (Å)",
        description="Distance from the supercell center to the camera. "
                    "Only used when Auto-frame is off.",
        default=30.0, min=1.0, soft_max=500.0,
    )  # type: ignore

    # --- Camera view mode (Side/Top/Iso presets + Spherical/Miller) ---
    camera_view: EnumProperty(
        name="Camera view",
        description="Camera placement mode. Presets (Side/Top/Iso) use fixed "
                    "framing. Spherical aims from an azimuth + polar angle (from +Z) at a "
                    "chosen radius. Miller aims along an (hkl) plane normal "
                    "(reciprocal-lattice direction; Ashcroft & Mermin Ch.5). "
                    "Target = structure bbox center; up = world +Z (fallback "
                    "+Y near vertical).",
        items=[
            ("SIDE", "Side", "a-b plane edge-on (along -Y)"),
            ("TOP", "Top", "down the c-axis (+Z, top-down)"),
            ("ISO", "Iso", "3/4 isometric view"),
            ("SPHERICAL", "Spherical",
             "Aim from azimuth + polar angle (from +Z) at radius. polar90/az0 = "
             "Side-like (-Y); polar0 = top-down (+Z)"),
            ("MILLER", "Miller",
             "Look along the (hkl) plane normal (reciprocal-lattice direction). "
             "Negative indices view from the opposite side"),
        ],
        default="SIDE",
        update=lambda self, ctx: _on_camera_view_change(ctx),
    )  # type: ignore
    # Spherical
    camera_azimuth: FloatProperty(
        name="Azimuth",
        description="Spherical azimuth in degrees. 0 = view from -Y (Side-like); "
                    "increases toward +X.",
        default=0.0, min=0.0, max=360.0,
        update=lambda self, ctx: _on_camera_param_change(ctx),
    )  # type: ignore
    camera_polar: FloatProperty(
        name="Polar (from +Z)",
        description="Spherical polar angle from +Z, in degrees. 0 = top-down "
                    "(+Z); 90 = horizontal (az 0 -> -Y, the Side view); "
                    "180 = bottom-up (-Z). Azimuth sweeps around +Z.",
        default=90.0, min=0.0, max=180.0,
        update=lambda self, ctx: _on_camera_param_change(ctx),
    )  # type: ignore
    camera_radius: FloatProperty(
        name="Radius (Å)",
        description="Spherical camera distance from center, in Å. NOTE: in the "
                    "orthographic camera this only positions the camera; zoom is "
                    "the Frame width / Auto-frame, not the radius.",
        default=30.0, min=1.0, soft_max=500.0,
        update=lambda self, ctx: _on_camera_param_change(ctx),
    )  # type: ignore
    # Miller (floats allowed for fine tuning, per ascii-phonons)
    camera_miller_h: FloatProperty(
        name="h", description="Miller index h (plane normal h·b1 + k·b2 + l·b3).",
        default=0.0,
        update=lambda self, ctx: _on_camera_param_change(ctx),
    )  # type: ignore
    camera_miller_k: FloatProperty(
        name="k", description="Miller index k.",
        default=1.0,
        update=lambda self, ctx: _on_camera_param_change(ctx),
    )  # type: ignore
    camera_miller_l: FloatProperty(
        name="l", description="Miller index l.",
        default=0.0,
        update=lambda self, ctx: _on_camera_param_change(ctx),
    )  # type: ignore
    # Movable look-at / orbit center (offset from the structure bbox center)
    camera_target_offset: FloatVectorProperty(
        name="Target offset",
        description="Offset (Å) added to the structure bbox center to move the "
                    "camera's look-at / orbit center. (0,0,0) = structure "
                    "center. A lateral offset re-centers the framing; in the "
                    "orthographic camera the along-view component has no effect.",
        size=3, subtype='TRANSLATION', unit='LENGTH',
        default=(0.0, 0.0, 0.0),
        update=lambda self, ctx: _on_camera_param_change(ctx),
    )  # type: ignore

    # --- Render preset ---
    render_preset: EnumProperty(
        name="Render style",
        items=[
            ("TOON",       "Toon (outlined)", "Cartoon style with Freestyle outlines — matches the model GIF"),
            ("EEVEE_PBR",  "Eevee PBR",       "Realistic shaders in Eevee (fast)"),
            ("CYCLES_PBR", "Cycles PBR",      "Realistic shaders in Cycles (slow, high quality)"),
        ],
        default="TOON",
    )  # type: ignore
    lighting_style: EnumProperty(
        name="Lighting",
        items=[
            ("STUDIO",     "Studio (3-point)",  "Three-point soft lighting with mild shadows. Good general default."),
            ("SCIENTIFIC", "Scientific (flat)", "Even lighting from 5 directions; minimal shadows. Recommended for publication."),
            ("DRAMATIC",   "Dramatic",          "Single hard light, deep shadows. Use for emphasis frames."),
        ],
        default="STUDIO",
    )  # type: ignore
    render_quality: EnumProperty(
        name="Mesh quality",
        items=[
            ("LOW",    "Low",    "Coarser spheres/arrows — fast viewport"),
            ("MEDIUM", "Medium", "Default tessellation"),
            ("HIGH",   "High",   "Very smooth spheres/arrows for close-ups and renders"),
        ],
        default="MEDIUM",
        update=lambda self, ctx: _on_quality_change(ctx),
    )  # type: ignore

    # --- Render samples / denoising (per-engine) ---
    eevee_samples: IntProperty(
        name="Eevee samples",
        description="TAA samples for Eevee renders. 32 is fast; 128 is "
                    "publication quality.",
        default=64, min=1, soft_max=512,
    )  # type: ignore
    cycles_samples: IntProperty(
        name="Cycles samples",
        description="Path-tracing samples for Cycles. 64-128 with denoising "
                    "gives a clean look; 256+ for hero shots.",
        default=128, min=1, soft_max=1024,
    )  # type: ignore
    use_denoising: BoolProperty(
        name="Denoising (Cycles)",
        description="Enable Cycles' built-in denoiser. Highly recommended at "
                    "low-to-medium sample counts.",
        default=True,
    )  # type: ignore

    # --- Output resolution & transparency ---
    render_resolution_x: IntProperty(
        name="Width (px)",
        description="Output image width in pixels (before percentage scaling).",
        default=1080, min=64, soft_max=7680,
    )  # type: ignore
    render_resolution_y: IntProperty(
        name="Height (px)",
        description="Output image height in pixels (before percentage scaling).",
        default=1080, min=64, soft_max=7680,
    )  # type: ignore
    render_percentage: IntProperty(
        name="Resolution %",
        description="Scale factor applied to width × height — 50% halves "
                    "render time at lower quality; 100% is full resolution.",
        default=100, min=10, max=400, subtype="PERCENTAGE",
    )  # type: ignore
    transparent_background: BoolProperty(
        name="Transparent background",
        description="Render with a transparent (alpha) background instead of "
                    "the world color. Useful for PNG sequences you want to "
                    "composite over a slide. NOTE: Toon-style outlines look "
                    "best with a solid background; transparency can fight with "
                    "Freestyle in some configurations.",
        default=False,
    )  # type: ignore

    # --- Export ---
    output_directory: StringProperty(
        name="Output Directory",
        description="Where to save rendered animations",
        subtype="DIR_PATH",
        default="//",
    )  # type: ignore
    output_format: EnumProperty(
        name="Output Format",
        items=[
            ("MP4", "MP4",      "H.264 MPEG-4 video"),
            ("GIF", "GIF",      "Animated GIF (requires ffmpeg in PATH)"),
            ("PNG", "PNG seq",  "PNG image sequence (no movie conversion)"),
        ],
        default="MP4",
    )  # type: ignore


# ============================================================
# Update callbacks — triggered when user changes mode / amplitude / visibility
# ============================================================
def _on_mode_change(context):
    """Called when q_index or band_index changes — rebuild displacements
    and refresh the burnt-in overlay text."""
    if _suppress_all_updates:
        return
    from .scene_setup import refresh_scene_for_current_mode
    from .overlay import refresh_overlay_stamp
    try:
        refresh_scene_for_current_mode(context.scene)
        refresh_overlay_stamp(context.scene)
    except Exception as e:
        print(f"[phonon-vision] mode change failed: {e}")


def _engine_evaluate_and_refresh(scene) -> None:
    """Re-evaluate D(q) at the CURRENT props q and refresh scene + overlay.
    THE single engine-mode evaluate+refresh path — shared verbatim by the
    direct-q callback and the path-walk callback (0.8.5), reused not
    duplicated (no evaluator change). No-op unless an engine source is active."""
    data = get_loaded_data(scene)
    if data is None or not hasattr(data, "set_q"):
        return                        # not an engine source: props are inert
    props = scene.phonon_vision
    from .scene_setup import refresh_scene_for_current_mode
    from .overlay import refresh_overlay_stamp
    try:
        data.set_q((float(props.q_frac_1), float(props.q_frac_2),
                    float(props.q_frac_3)))
        refresh_scene_for_current_mode(scene)
        refresh_overlay_stamp(scene)
    except Exception as e:
        print(f"[phonon-vision] engine q change failed: {e}")


def _on_engine_q_change(context):
    """Called when any q_frac_* component changes (engine source only) —
    re-evaluate D(q) and refresh. A DIRECT q change (the user dragging a q
    slider, or a jump button) is a departure from the parameterized walk, so it
    flips the off-path flag (one-way flow — q is never back-projected onto the
    self-intersecting path). The path slider writes q UNDER the suppress guard
    and refreshes via _engine_evaluate_and_refresh directly, so it never passes
    here. The _suppress_all_updates guard doubles as the recursion guard."""
    if _suppress_all_updates:
        return
    global _path_off
    _path_off = True                  # q was set directly -> off the path
    _engine_evaluate_and_refresh(context.scene)


# High-symmetry jump targets for the ENGINE navigator: remembered from the
# last BAND file loaded this session (label, q_frac triple). Session-scoped
# runtime state, like _loaded_data — never serialized.
_band_hsp: list = []
_band_hsp_provenance: str = ""

# Path-walk (0.8.5): the ordered segment table of the loaded band file's
# DECLARED path — [(label_start, label_end, q_start(3), q_end(3), length)] with
# length = |Δq @ reciprocal_lattice| (Cartesian, so the walk slider s tracks
# the plotted band-structure x-axis), plus the cumulative-length array
# _path_cum = [0, len0, len0+len1, …, total]. _path_off is the one-way-flow
# flag: True once q was set DIRECTLY (a q slider or a jump button), so the Walk
# readout says "off-path" — q is NEVER back-projected onto the path (it
# self-intersects at Γ; projection is ill-posed). Cleared by the lattice route.
_path_segments: list = []
_path_cum: list = []
_path_off: bool = False


def _prettify_point_label(lab) -> str:
    """Clean a phonopy BAND_LABELS high-symmetry point NAME for the jump row.

    Undoes the artifacts of the comma-separated BAND_LABELS / TeX syntax:
    strip a trailing ','/whitespace (segment-final labels carry one, e.g.
    ``'X,'``); strip the ``$`` math delimiters; ``\\Gamma`` -> ``Γ``,
    ``\\prime`` -> ``′``; collapse TeX sub/superscript braces
    (``$R_2$`` -> ``R2``, ``$X^{\\prime}$`` -> ``X′``). Anything unrecognised is
    left VERBATIM — never invent a name. Pure; unit-tested.

    NB this operates on a SINGLE point name from ``BandPath.labels``; it is
    deliberately NOT fed the compound ``'X,|Y'`` tick strings (those are plot
    furniture, not point names — the 0.8.3 field defect)."""
    s = str(lab).strip().rstrip(",").strip()
    s = s.strip("$")
    s = s.replace("\\Gamma", "\u0393").replace("\\prime", "\u2032")
    s = (s.replace("^{", "").replace("_{", "")
          .replace("^", "").replace("_", "")
          .replace("{", "").replace("}", ""))
    return s


def remember_band_hsp(data, *, provenance: str = "") -> None:
    """Capture (label, q_frac) jump targets from a loaded BandPath — a BAND
    file load or the engine navigator's Source 1 (a same-material sibling).

    Point NAMES come from ``BandPath.labels`` — the PER-SEGMENT (start, end)
    high-symmetry point names (the truth) — NEVER from ``tick_locations()``,
    whose matplotlib x-axis tick strings FUSE path discontinuities into
    compound ``'X,|Y'`` furniture (the 0.8.3 field defect: a button labelled
    "X|Y" that actually jumped to X). Each segment contributes BOTH endpoints,
    paired with the segment's start / end q; the unchanged
    ``assemble_jump_targets`` then dedups by wrapped q (first spelling wins, so
    primed variants absorb onto their unprimed points). ``provenance`` is the
    user-facing origin shown in the Data panel (e.g. ``"band.hdf5"``)."""
    global _band_hsp, _band_hsp_provenance
    try:
        qp = data.q_points
        seg_labels = data.labels
        pairs = []
        segs = []
        start = 0
        for i, n in enumerate(data.segment_nqpoint):
            lab_start, lab_end = seg_labels[i]
            ps = _prettify_point_label(lab_start)
            pe = _prettify_point_label(lab_end)
            q_s = [float(x) for x in qp[start]]
            q_e = [float(x) for x in qp[start + n - 1]]
            pairs.append((ps, q_s))
            pairs.append((pe, q_e))
            segs.append((ps, pe, q_s, q_e))
            start += n
        _band_hsp = pairs
        _band_hsp_provenance = provenance
        # 0.8.5: also stash the ordered segment table for path walking. RAW
        # per-segment labels (prettified; primes PRESERVED — a route's identity
        # is its direction), unlike the dedup'd jump row above.
        _set_path_segments(segs, data.material.reciprocal_lattice)
    except Exception as e:
        print(f"[phonon-vision] band HSP capture failed: {e}")
        _band_hsp = []
        _band_hsp_provenance = ""
        clear_path_segments()


def set_band_hsp(targets, provenance: str = "") -> None:
    """Set the jump-target store directly from a ``[(label, q), ...]`` list —
    the engine navigator's Source 2 (lattice-derived high-symmetry points),
    which has no BandPath to extract from."""
    global _band_hsp, _band_hsp_provenance
    _band_hsp = [(str(lab), [float(x) for x in q]) for lab, q in targets]
    _band_hsp_provenance = provenance
    clear_path_segments()   # the lattice route (Source 2) declares no path to walk


def get_band_hsp() -> list:
    return list(_band_hsp)


def get_hsp_provenance() -> str:
    return _band_hsp_provenance


# ============================================================
# Path walking (0.8.5) — a parameterization layer over engine mode
# ============================================================
def _set_path_segments(segments, reciprocal_lattice) -> None:
    """Build the ordered segment table + cumulative lengths from per-segment
    ``(label_start, label_end, q_start, q_end)`` tuples. length =
    |Δq @ reciprocal_lattice| (Cartesian, so s tracks the plotted band x-axis).
    A freshly built path starts ON-path (at s=0 = the path's first point)."""
    global _path_segments, _path_cum, _path_off
    import numpy as np
    R = np.asarray(reciprocal_lattice, dtype=float)
    table = []
    cum = [0.0]
    for lab_s, lab_e, q_s, q_e in segments:
        dq = np.asarray(q_e, dtype=float) - np.asarray(q_s, dtype=float)
        length = float(np.linalg.norm(dq @ R))
        table.append((str(lab_s), str(lab_e),
                      [float(x) for x in q_s], [float(x) for x in q_e], length))
        cum.append(cum[-1] + length)
    _path_segments = table
    _path_cum = cum
    _path_off = False


def clear_path_segments() -> None:
    global _path_segments, _path_cum
    _path_segments = []
    _path_cum = []


def get_path_segments() -> list:
    """The ordered [(lab_s, lab_e, q_s, q_e, length), ...] segment table."""
    return list(_path_segments)


def path_total() -> float:
    return float(_path_cum[-1]) if _path_cum else 0.0


def has_path(scene) -> bool:
    """True iff a declared band path exists AND an engine source is active —
    walking uses exact D(q) evaluation, which only the engine provides, and a
    path exists only via a same-material sibling band file (Source 1)."""
    return bool(_path_segments) and scene.phonon_vision.data_kind == "ENGINE"


def is_off_path() -> bool:
    return _path_off


def _locate_path_s(s):
    """(segment_index, t_within_segment) for normalized s in [0,1]; None if no
    path. At a segment boundary the LATER segment's start is chosen (t=0), so a
    shared endpoint is bitwise-seamless from both sides."""
    if not _path_segments:
        return None
    import bisect
    total = _path_cum[-1]
    s = max(0.0, min(1.0, float(s)))
    if total <= 0.0:
        return 0, 0.0
    target = s * total
    i = bisect.bisect_right(_path_cum, target) - 1
    i = max(0, min(i, len(_path_segments) - 1))
    length = _path_segments[i][4]
    t = 0.0 if length <= 0.0 else (target - _path_cum[i]) / length
    return i, max(0.0, min(1.0, t))


def q_at_path_s(s):
    """Fractional q at normalized path position s — the phonopy band
    convention: linear FRACTIONAL interpolation q = (1−t)·q_start + t·q_end
    within the located segment. None if no path."""
    loc = _locate_path_s(s)
    if loc is None:
        return None
    i, t = loc
    q_s = _path_segments[i][2]
    q_e = _path_segments[i][3]
    return [(1.0 - t) * q_s[k] + t * q_e[k] for k in range(3)]


def path_readout(s):
    """(label_start, label_end, percent_within_segment) for the Walk readout;
    None if no path."""
    loc = _locate_path_s(s)
    if loc is None:
        return None
    i, t = loc
    return _path_segments[i][0], _path_segments[i][1], int(round(100.0 * t))


def _on_path_s_changed(context):
    """Path-walk slider: s -> segment -> interpolated q -> write q1/q2/q3 UNDER
    the suppress guard (so the per-component q callback does not storm) -> ONE
    evaluation + refresh via the SHARED _engine_evaluate_and_refresh. Dragging
    the slider returns ON-path."""
    if _suppress_all_updates:
        return
    if not _path_segments:
        return
    scene = context.scene
    props = scene.phonon_vision
    q = q_at_path_s(float(props.path_s))
    if q is None:
        return
    global _path_off
    with _SuppressUpdates():
        props.q_frac_1, props.q_frac_2, props.q_frac_3 = q
    _path_off = False
    _engine_evaluate_and_refresh(scene)


def _wrap_q_component(x) -> float:
    """Map a fractional-q component to its image in [-0.5, 0.5) — the nearest
    reciprocal-lattice image. This is jump-target IDENTITY: a q and a q shifted
    by a reciprocal lattice vector are the same physical point (so +0.5 and
    -0.5 both map to -0.5, the [-0.5, 0.5) representative)."""
    return (float(x) + 0.5) % 1.0 - 0.5


def assemble_jump_targets(band_hsp):
    """Engine-mode jump targets, deduplicated by WRAPPED q — NOT label string.

    Returns ``[(label, (q1, q2, q3)), ...]`` starting with the standing Gamma
    ``("Γ", (0, 0, 0))``; each ``(label, q)`` from ``band_hsp`` is
    appended only if its wrapped q differs from every already-accepted
    target's wrapped q by more than 1e-6 in max-norm. Label plays NO part in
    identity and the FIRST spelling of a q wins — so the standing Gamma
    absorbs any path Gamma whatever its raw ``tick_locations()`` spelling.

    FIELD DEFECT (0.8.0 acceptance): the path start-Gamma arrives as
    ``"\\Gamma"`` while the standing Gamma is the literal ``"Γ"``, so the
    old label-string dedup missed it and rendered TWO Gamma buttons. q is
    identity now. Pure / bpy-free — directly tested in
    tools/test_jump_targets.py.
    """
    targets = [("\u0393", (0.0, 0.0, 0.0))]
    wrapped = [tuple(_wrap_q_component(c) for c in targets[0][1])]
    for lab, q in band_hsp:
        wq = tuple(_wrap_q_component(c) for c in q)
        if all(max(abs(wq[i] - w[i]) for i in range(3)) > 1e-6
               for w in wrapped):
            targets.append((str(lab), tuple(float(c) for c in q)))
            wrapped.append(wq)
    return targets


def _on_anim_change(context):
    """Called when amplitude or arrow_scale changes — re-run frame at current frame."""
    if _suppress_all_updates:
        return
    from .frame_handler import update_frame
    try:
        update_frame(context.scene)
    except Exception as e:
        print(f"[phonon-vision] anim refresh failed: {e}")


def _on_camera_view_change(context):
    """Camera mode selected — (re)apply the camera. Creates PV.CameraObj if
    needed; presets use direct rotation, Spherical/Miller use a TRACK_TO target."""
    if _suppress_all_updates:
        return
    from .operators import apply_camera
    try:
        apply_camera(context.scene, context.scene.phonon_vision.camera_view)
    except Exception:
        import traceback
        traceback.print_exc()


def _on_camera_param_change(context):
    """A Spherical/Miller parameter changed — live-update an EXISTING camera
    only (don't spawn one from a stray slider edit before a mode is chosen)."""
    if _suppress_all_updates:
        return
    if bpy.data.objects.get("PV.CameraObj") is None:
        return
    from .operators import apply_camera
    try:
        apply_camera(context.scene, context.scene.phonon_vision.camera_view)
    except Exception:
        import traceback
        traceback.print_exc()


def _on_visual_change(context):
    """Toggle visibility, or rebuild cell when its kind changes."""
    if _suppress_all_updates:
        return
    from .scene_setup import refresh_visibility, refresh_cell
    try:
        refresh_cell(context.scene)
        refresh_visibility(context.scene)
    except Exception as e:
        print(f"[phonon-vision] visibility refresh failed: {e}")


# Deferred REDRAW kicks (view_layer.update + tag_redraw — NO material write, NO
# object replacement). Drained by an idle timer (static case) and, as a playback
# fallback, by the frame-change handler (bpy.app.timers are STARVED during
# continuous playback). Keyed so a swatch drag coalesces to one kick per key.
_pending_redraws: dict = {}
_redraw_timer_armed: bool = False


def _drain_redraws():
    """Run all pending redraw kicks once and clear. Idempotent and safe to call
    from both the idle timer and the frame-change handler (only one will have
    work; the other finds the queue empty)."""
    global _redraw_timer_armed
    _redraw_timer_armed = False
    if not _pending_redraws:
        return None
    fns = list(_pending_redraws.values())
    _pending_redraws.clear()
    for fn in fns:
        try:
            fn()
        except Exception:
            import traceback
            traceback.print_exc()
    return None   # one-shot timer


def _schedule_redraw(key: str, redraw_fn) -> None:
    """Queue a deferred redraw kick (debounced per key) and arm the idle timer."""
    global _redraw_timer_armed
    _pending_redraws[key] = redraw_fn
    if not _redraw_timer_armed:
        _redraw_timer_armed = True
        bpy.app.timers.register(_drain_redraws, first_interval=0.0)


# Deferred GEOMETRY / object-replacement (radius -> sphere-mesh rebuild + swap).
# Timer-only and debounced; NEVER run synchronously in a callback and NEVER
# per-frame (object replacement mid-playback is out of scope and unsafe).
_pending_geometry: set = set()


def _schedule_geometry(key: str, refresh_fn) -> None:
    if key in _pending_geometry:
        return
    _pending_geometry.add(key)

    def _deferred():
        _pending_geometry.discard(key)
        try:
            refresh_fn(bpy.context.scene)
        except Exception:
            import traceback
            traceback.print_exc()
        return None   # one-shot

    bpy.app.timers.register(_deferred, first_interval=0.0)


def _on_arrow_color_change(context):
    """Live arrow-color update.

    Writes PV.ArrowMat SYNCHRONOUSLY (material + update_tag) so it is correct
    even during playback, when timers are starved. Defers ONLY the redraw kick
    (view_layer.update + tag_redraw) for the static case; during playback the
    per-frame handler's depsgraph eval picks up the dirtied material.
    """
    if _suppress_all_updates:
        return
    from .scene_setup import write_arrow_color_sync, redraw_kick
    try:
        write_arrow_color_sync(context.scene)
    except Exception:
        import traceback
        traceback.print_exc()
    _schedule_redraw("arrow_color", redraw_kick)


def _on_supercell_change(context):
    """Rebuild the atoms+arrows+cell when supercell_n changes."""
    if _suppress_all_updates:
        return
    from .scene_setup import build_scene_from_data
    try:
        build_scene_from_data(context.scene)
    except Exception as e:
        print(f"[phonon-vision] supercell rebuild failed: {e}")


def _on_bond_change(context):
    """Rebuild the bond list (e.g. when bond_tolerance changes)."""
    if _suppress_all_updates:
        return
    from .scene_setup import refresh_bonds
    try:
        refresh_bonds(context.scene)
    except Exception as e:
        print(f"[phonon-vision] bond rebuild failed: {e}")


def _on_pbc_change(context):
    """Toggling bond_periodic changes the atom supercell (adds/removes image
    atoms), so we need a full rebuild — atoms, arrows, bonds, cell."""
    if _suppress_all_updates:
        return
    from .scene_setup import build_scene_from_data
    try:
        build_scene_from_data(context.scene)
    except Exception as e:
        print(f"[phonon-vision] PBC change failed: {e}")


def _on_species_change(context):
    """Called when the user changes a species color or radius.

    Updates only the affected species: rebuilds its template sphere mesh
    (radius) and refreshes its material (color). Also refreshes bond
    half-cylinders since they take their colors from the species materials.

    Guarded by ``_suppress_species_updates`` so that bulk operations
    (e.g. ``_populate_species_overrides`` during a Reset-to-VESTA click)
    can write many properties at once and call a single
    ``refresh_species_visuals`` at the end, instead of firing one refresh
    per (color, radius) assignment. Without this guard, the second
    consecutive VESTA click hit a race where mesh datablocks were being
    swapped while a prior refresh was still iterating them.
    """
    if _suppress_all_updates or _suppress_species_updates:
        return
    from .scene_setup import (apply_species_colors_sync, refresh_species_visuals,
                              redraw_kick)
    # SYNCHRONOUS color/vibrancy write (+ update_tag): callback-safe and works
    # during playback (timers are starved then). Geometry (radius/DupliVerts) is
    # NOT touched here.
    try:
        apply_species_colors_sync(context.scene)
    except Exception:
        import traceback
        traceback.print_exc()
    # Deferred redraw kick (static case); frame handler drains it during playback.
    _schedule_redraw("species", redraw_kick)
    # Deferred, timer-only: heavy object-replacement for radius changes. Stays
    # off the synchronous and per-frame paths by design.
    _schedule_geometry("species", refresh_species_visuals)


# Set to True to make _on_species_change a no-op temporarily. The bulk
# operations that flip this MUST flip it back in a try/finally so a stuck
# True state can't blackhole all future species edits.
_suppress_species_updates: bool = False


class _SuppressSpeciesUpdates:
    """Context manager that suppresses _on_species_change.

    Use:
        with _SuppressSpeciesUpdates():
            for ... :
                entry.color = ...    # no refresh fires
                entry.radius = ...   # no refresh fires
        # one explicit refresh here, after the loop
    """
    def __enter__(self):
        global _suppress_species_updates
        _suppress_species_updates = True
        return self

    def __exit__(self, *exc):
        global _suppress_species_updates
        _suppress_species_updates = False
        return False  # never swallow exceptions


# Generalized: suppress ALL phonon_vision prop update callbacks. Every _on_*
# callback early-returns when this is set. Wrap bulk PROGRAMMATIC prop-setting
# (load_file, apply_job_to_scene, and GUI bulk operators) so that setting many
# props doesn't fire N redundant refreshes — and so headless runs stay quiet.
_suppress_all_updates: bool = False


class _SuppressUpdates:
    """Context manager that makes every _on_* update callback a no-op.

    Nesting-safe (restores the previous value, not unconditionally False) and
    restores in __exit__ even on exception, so a raised error can't leave all
    future prop updates blackholed.
    """
    def __enter__(self):
        global _suppress_all_updates
        self._prev = _suppress_all_updates
        _suppress_all_updates = True
        return self

    def __exit__(self, *exc):
        global _suppress_all_updates
        _suppress_all_updates = self._prev
        return False  # never swallow exceptions


def _on_quality_change(context):
    """Rebuild atoms (and arrows on next frame) at the new tessellation."""
    if _suppress_all_updates:
        return
    from .scene_setup import build_scene_from_data
    if context.scene.phonon_vision.data_kind == "NONE":
        return
    try:
        build_scene_from_data(context.scene)
    except Exception as e:
        print(f"[phonon-vision] quality refresh failed: {e}")


def _on_overlay_change(context):
    """Push show_mode_overlay + colors into scene.render's stamp settings,
    and refresh the stamp note text from the current mode."""
    if _suppress_all_updates:
        return
    from .overlay import refresh_overlay_stamp
    try:
        refresh_overlay_stamp(context.scene)
    except Exception as e:
        print(f"[phonon-vision] overlay refresh failed: {e}")
