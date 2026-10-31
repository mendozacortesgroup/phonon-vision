"""UI panels.

Four panels in the N-Panel (sidebar) of the 3D View, under a "Phonon" tab:

1. **Data** — load YAML, clear, show what's loaded, manual rebuild.
2. **Mode Navigator** — pick (q, band) with sliders and prev/next buttons.
3. **Display** — visibility toggles, amplitude, supercell size, cell mode.
4. **Render** — preset, camera (auto or manual), loop, output, big Render button.
"""

from __future__ import annotations

import os

import bpy

# ============================================================
# phonon-core imports — module-top with stale-wheel guard
# ------------------------------------------------------------
# Doing these imports inside Panel.draw() (as 0.2.8 did) means a stale
# bundled wheel causes ImportError at draw time, which Blender swallows
# silently and TRUNCATES THE PANEL below the failing line. The user sees
# missing UI elements with no error message.
#
# Putting the imports at module top makes any mismatch fail loudly at
# extension-load time (the user gets a clear ImportError dialog). The
# try/except below is a defensive belt-and-braces: if the import does
# somehow fail at load time but the module loads anyway, stub fallbacks
# keep the rest of the UI drawable and the Data panel shows a banner.
# ============================================================
try:
    from phonon_core.units import format_frequency, prettify_label
    _PC_IMPORT_OK = True
    _PC_IMPORT_ERR: str | None = None
except ImportError as e:
    _PC_IMPORT_OK = False
    _PC_IMPORT_ERR = str(e)

    def format_frequency(freq, unit="THz", precision=3):
        """Stub fallback — used only if the bundled phonon-core wheel is stale."""
        try:
            return f"{float(freq):.{int(precision)}f} {unit}"
        except Exception:
            return str(freq)

    def prettify_label(label):
        """Stub fallback — used only if the bundled phonon-core wheel is stale."""
        return label if isinstance(label, str) else ""

from . import bonds as bondcore
from .state import get_loaded_data


# Helper to wrap each draw() so a single panel crash can't blank the entire
# Phonon tab. Prints the traceback to console; user sees a tiny inline marker.
def _safe_draw(method):
    """Decorator: wrap Panel.draw so exceptions become an inline label, not a silent void."""
    def wrapped(self, context):
        try:
            method(self, context)
        except Exception as e:
            import traceback
            print(f"[phonon-vision] panel {self.bl_idname} draw failed: {e}")
            traceback.print_exc()
            row = self.layout.row()
            row.alert = True
            row.label(text=f"draw failed: {type(e).__name__} (see console)", icon="ERROR")
    return wrapped


class PV_PT_data(bpy.types.Panel):
    bl_label = "Data"
    bl_idname = "PV_PT_data"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Phonon"

    @_safe_draw
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.phonon_vision
        data = get_loaded_data(scene)

        # If phonon-core was stale at import time, show a clear banner.
        # The rest of the UI still draws (with stub formatting) so the user
        # isn't faced with a blank tab.
        if not _PC_IMPORT_OK:
            box = layout.box()
            box.alert = True
            box.label(text="phonon-core wheel is out of date", icon="ERROR")
            box.label(text="Uninstall + reinstall the Phonon Vision addon")
            box.label(text=f"({_PC_IMPORT_ERR[:50] if _PC_IMPORT_ERR else 'unknown'})")
            layout.separator()

        col = layout.column(align=True)
        col.prop(props, "yaml_path", text="")
        row = col.row(align=True)
        row.operator("phonon_vision.load_yaml", icon="FILE_TICK")
        row.operator("phonon_vision.clear", icon="TRASH", text="")

        layout.separator()
        if data is None:
            layout.label(text="No data loaded.", icon="INFO")
        else:
            box = layout.box()
            box.label(
                text=f"{props.data_kind}: {data.material.num_atoms} atoms",
                icon="MATSPHERE",
            )
            box.label(text=f"Species: {', '.join(data.material.unique_species)}")
            if props.data_kind == "ENGINE":
                box.label(text="ENGINE (A2 numpy) — exact D(q) at any q",
                          icon="PHYSICS")
                box.label(text=f"FC: {os.path.basename(getattr(data, 'fc_path', '?'))}")
                box.label(text=f"any q × {data.material.num_bands} bands")
                # provenance of the jump-row targets (0.8.2): sibling band file
                # or the reciprocal-lattice symmetry — stays visible.
                from .state import get_hsp_provenance
                _prov = get_hsp_provenance()
                if _prov:
                    box.label(text=f"Jump targets: {_prov}")
            else:
                box.label(text=f"{data.num_qpoints} q-points × {data.material.num_bands} bands")

            # Manual rebuild — fallback if a property update didn't refresh
            layout.separator()
            layout.operator("phonon_vision.rebuild_scene", icon="FILE_REFRESH")


class PV_PT_navigator(bpy.types.Panel):
    bl_label = "Mode Navigator"
    bl_idname = "PV_PT_navigator"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Phonon"

    @classmethod
    def poll(cls, ctx):
        return get_loaded_data(ctx.scene) is not None

    @_safe_draw
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.phonon_vision
        data = get_loaded_data(scene)
        if data is None:
            return

        n_b = data.material.num_bands
        n_q = data.num_qpoints

        # Mode info card
        try:
            mode = data.get_mode(int(props.q_index), int(props.band_index))
            box = layout.box()
            box.label(
                text=f"Mode {props.band_index + 1} / {n_b}",
                icon="ANIM",
            )
            box.label(
                text=f"q = ({mode.q_point[0]:.4f}, {mode.q_point[1]:.4f}, {mode.q_point[2]:.4f})"
            )
            if props.data_kind == "ENGINE":
                box.label(text="exact engine mode \u2014 q = "
                               f"({mode.q_point[0]:.4f}, {mode.q_point[1]:.4f}, "
                               f"{mode.q_point[2]:.4f}) r.l.u.")
            freq_label = format_frequency(mode.frequency, "THz", 3)
            row = box.row()
            row.label(text=f"Freq: {freq_label}")
            # Only flag truly imaginary modes — suppress sub-threshold noise
            if mode.is_imaginary and abs(mode.frequency) > float(props.imag_freq_threshold):
                row.label(text="imaginary", icon="ERROR")
        except Exception:
            pass

        col = layout.column(align=True)
        if props.data_kind == "ENGINE":
            # LIVE q (r.l.u.) — dragging any slider re-evaluates D(q)
            # continuously (~0.07 ms/evaluation); no Evaluate button, no
            # fake 2D-path position is ever shown for an engine q (the
            # navigator Stage-3 honesty rule).
            col.label(text="exact engine mode — q (r.l.u.):")
            col.prop(props, "q_frac_1", text="q1", slider=True)
            col.prop(props, "q_frac_2", text="q2", slider=True)
            col.prop(props, "q_frac_3", text="q3", slider=True)
            col.separator()
            col.label(text="Jump to:")
            from .state import get_band_hsp, assemble_jump_targets
            # dedup by WRAPPED q, not label string (0.8.1): the standing Gamma
            # absorbs any path Gamma whatever its raw tick_locations() spelling.
            targets = assemble_jump_targets(get_band_hsp())
            row = col.row(align=True)
            for i, (lab, q) in enumerate(targets):
                if i > 0 and i % 4 == 0:
                    row = col.row(align=True)
                try:
                    text = prettify_label(lab) or lab
                except Exception:
                    text = lab
                op = row.operator("phonon_vision.engine_jump", text=text)
                op.label = lab
                op.q = q
            # Walk-path row (0.8.5): one slider carries q CONTINUOUSLY along the
            # band file's declared path, exact D(q) at every point. Only when a
            # path exists (a same-material sibling band, via Source 1). Moving a
            # q slider or clicking a jump above flips to "off-path" (one-way
            # flow); grabbing this slider resumes the walk.
            from .state import has_path, is_off_path, path_readout
            if has_path(scene):
                col.separator()
                col.label(text="Walk path:")
                col.prop(props, "path_s", text="s", slider=True)
                if is_off_path():
                    col.label(text="off-path — grab the slider to resume",
                              icon="INFO")
                else:
                    ro = path_readout(props.path_s)
                    if ro:
                        col.label(text=f"{ro[0]} → {ro[1]} · {ro[2]}%")
        else:
            col.label(text=f"q-point  ({props.q_index + 1} / {n_q}):")
            row = col.row(align=True)
            row.operator("phonon_vision.prev_q", text="", icon="TRIA_LEFT")
            row.prop(props, "q_index", text="")
            row.operator("phonon_vision.next_q", text="", icon="TRIA_RIGHT")

        col.label(text=f"Band  ({props.band_index + 1} / {n_b}):")
        row = col.row(align=True)
        row.operator("phonon_vision.prev_mode", text="", icon="TRIA_LEFT")
        row.prop(props, "band_index", text="")
        row.operator("phonon_vision.next_mode", text="", icon="TRIA_RIGHT")

        if props.data_kind == "BAND" and hasattr(data, "segment_nqpoint"):
            col.separator()
            col.label(text="Jump to high-symmetry:")
            # Wrapped in try/except: a malformed label set should never break
            # the entire Mode Navigator panel.
            try:
                positions, labels = data.tick_locations()
                # 4 per row keeps buttons visible at typical sidebar widths.
                # Very long compound labels (e.g. 'T₂′|U₂′') need real width.
                BUTTONS_PER_ROW = 4
                row = col.row(align=True)
                for i, lab in enumerate(labels):
                    if i > 0 and i % BUTTONS_PER_ROW == 0:
                        row = col.row(align=True)
                    try:
                        text = prettify_label(lab) or f"#{i+1}"
                    except Exception:
                        text = f"#{i+1}"
                    op = row.operator("phonon_vision.jump_high_sym", text=text)
                    op.point_index = i
            except Exception as e:
                col.label(text=f"(labels unavailable: {e})", icon="ERROR")


class PV_PT_display(bpy.types.Panel):
    bl_label = "Display"
    bl_idname = "PV_PT_display"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Phonon"

    @_safe_draw
    def draw(self, context):
        layout = self.layout
        props = context.scene.phonon_vision

        # Shading switcher
        col = layout.column(align=True)
        col.label(text="Viewport shading:")
        row = col.row(align=True)
        op = row.operator("phonon_vision.set_viewport_shading", text="Solid")
        op.mode = "SOLID"
        op = row.operator("phonon_vision.set_viewport_shading", text="Material")
        op.mode = "MATERIAL"
        op = row.operator("phonon_vision.set_viewport_shading", text="Rendered")
        op.mode = "RENDERED"

        layout.separator()
        col = layout.column(align=True)
        col.prop(props, "amplitude")
        col.prop(props, "arrow_scale")
        col.prop(props, "arrow_color")
        col.prop(props, "use_canonical_phase")
        col.prop(props, "imag_freq_threshold")

        col.separator()
        col.label(text="Supercell N × M × K:")
        row = col.row(align=True)
        row.prop(props, "supercell_n", text="")

        col.separator()
        col.label(text="Visibility:")
        col.prop(props, "show_arrows", toggle=True)
        row = col.row(align=True)
        row.prop(props, "show_unit_cell", text="Show cell", toggle=True)
        sub = row.row(align=True)
        sub.enabled = props.show_unit_cell
        sub.prop(props, "unit_cell_kind", text="")
        sub2 = col.row(align=True)
        sub2.enabled = props.show_unit_cell
        sub2.prop(props, "cell_thickness", text="Edge thickness")

        # Bonds
        col.separator()
        col.prop(props, "show_bonds", toggle=True)
        sub = col.column(align=True)
        sub.enabled = props.show_bonds
        # bond_tolerance is fully bypassed while a curated spec is loaded
        # (replace semantics) — grey it out so the UI says so.
        tol_row = sub.row(align=True)
        tol_row.enabled = not bool(props.bonds_spec_json)
        tol_row.prop(props, "bond_tolerance")
        # bond_thickness governs only bonds WITHOUT a curated radius_A.
        # When every curated pair carries radius_A (any navigator >= 0.6.76
        # export), the slider drives nothing drawn under replace semantics —
        # grey it out, mirroring bond_tolerance above.
        thick_row = sub.row(align=True)
        thick_row.enabled = not bondcore.all_pairs_have_radius_A(
            bondcore.spec_from_json(props.bonds_spec_json))
        thick_row.prop(props, "bond_thickness")
        sub.prop(props, "bond_two_color")
        sub.prop(props, "bond_periodic")
        # Per-frame break thresholds — bonds hide when stretched/compressed
        # past these multiples of equilibrium length
        sub.prop(props, "bond_break_max_factor")
        sub.prop(props, "bond_break_min_factor")
        # Curated bonds (navigator side-car). While a spec is loaded it
        # REPLACES the tolerance criterion (greyed out above).
        sub.separator()
        row = sub.row(align=True)
        row.operator("phonon_vision.import_bonds", text="Import bonds JSON",
                     icon="IMPORT")
        row.operator("phonon_vision.clear_bonds", text="", icon="X")
        if props.bonds_spec_json:
            try:
                import json as _json
                _spec = _json.loads(props.bonds_spec_json)
                _src = (_spec.get("source") or {}).get("filename", "")
                _n = len(_spec.get("bonds", {}))
                sub.label(text=f"Curated: {_n} pair(s)"
                               + (f" — {_src}" if _src else ""),
                          icon="CHECKMARK")
            except Exception:
                sub.label(text="Curated bonds: (unreadable spec)",
                          icon="ERROR")

        # Amplitude clamp
        col.separator()
        col.prop(props, "clamp_amplitude", toggle=True)
        sub = col.column(align=True)
        sub.enabled = props.clamp_amplitude
        sub.prop(props, "max_displacement")


class PV_PT_render(bpy.types.Panel):
    bl_label = "Render"
    bl_idname = "PV_PT_render"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Phonon"

    @_safe_draw
    def draw(self, context):
        layout = self.layout
        props = context.scene.phonon_vision

        col = layout.column(align=True)
        col.label(text="Style:")
        col.prop(props, "render_preset", text="")
        col.label(text="Lighting:")
        col.prop(props, "lighting_style", text="")
        col.label(text="Mesh quality:")
        col.prop(props, "render_quality", text="")

        col.separator()
        col.label(text="Samples:")
        col.prop(props, "eevee_samples")
        col.prop(props, "cycles_samples")
        col.prop(props, "use_denoising")
        col.operator("phonon_vision.apply_render_preset", icon="MATERIAL")

        col.separator()
        col.label(text="Camera view:")
        col.prop(props, "camera_view", text="")
        if props.camera_view == "SPHERICAL":
            col.label(text="Azimuth / Polar / Radius:")
            row = col.row(align=True)
            row.prop(props, "camera_azimuth", text="Az")
            row.prop(props, "camera_polar", text="Polar")
            row.prop(props, "camera_radius", text="R")
        elif props.camera_view == "MILLER":
            col.label(text="Miller index (h k l):")
            row = col.row(align=True)
            row.prop(props, "camera_miller_h", text="h")
            row.prop(props, "camera_miller_k", text="k")
            row.prop(props, "camera_miller_l", text="l")
        op_cam = col.operator("phonon_vision.setup_camera",
                              text="Setup / Update Camera", icon="CAMERA_DATA")
        op_cam.view = props.camera_view

        # Movable look-at / orbit center
        col.label(text="Target offset X/Y/Z (Å):")
        row = col.row(align=True)
        row.prop(props, "camera_target_offset", index=0, text="X")
        row.prop(props, "camera_target_offset", index=1, text="Y")
        row.prop(props, "camera_target_offset", index=2, text="Z")
        col.operator("phonon_vision.center_camera_target",
                     text="Center on structure", icon="PIVOT_BOUNDBOX")

        # Camera framing — auto vs manual
        col.separator()
        col.prop(props, "camera_auto_frame", text="Auto-frame", toggle=True)
        sub = col.column(align=True)
        sub.enabled = not props.camera_auto_frame
        sub.prop(props, "camera_ortho_scale")
        sub.prop(props, "camera_distance")
        if props.camera_auto_frame:
            col.label(text="(showing auto-computed values)", icon="INFO")

        col.separator()
        col.label(text="Loop:")
        col.prop(props, "frames_per_cycle")
        col.prop(props, "fps_render")
        col.operator("phonon_vision.configure_loop", icon="TIME")

        col.separator()
        col.label(text="Resolution:")
        row = col.row(align=True)
        row.prop(props, "render_resolution_x", text="W")
        row.prop(props, "render_resolution_y", text="H")
        col.prop(props, "render_percentage")
        col.prop(props, "transparent_background")
        # Quick-set buttons (two rows of 2 to avoid label squashing)
        row = col.row(align=True)
        for label, key in (("1080²", "SQUARE_1080"), ("1920×1080", "FULL_HD")):
            op = row.operator("phonon_vision.set_resolution_preset", text=label)
            op.preset = key
        row = col.row(align=True)
        for label, key in (("1920²", "LARGE_SQUARE"), ("4K", "UHD_4K")):
            op = row.operator("phonon_vision.set_resolution_preset", text=label)
            op.preset = key

        col.separator()
        col.label(text="Overlay (burnt into render):")
        col.prop(props, "show_mode_overlay", toggle=True)
        sub = col.column(align=True)
        sub.enabled = props.show_mode_overlay
        sub.prop(props, "overlay_text_size", text="Size")
        sub.prop(props, "overlay_text_color", text="Color")
        sub.prop(props, "overlay_background_color", text="Background")

        col.separator()
        col.label(text="Export:")
        col.prop(props, "output_directory", text="")
        col.prop(props, "output_format", text="")

        big = layout.row()
        big.scale_y = 1.6
        big.operator("phonon_vision.render_animation", icon="RENDER_ANIMATION")

        # Headless authoring: export current settings as a job file for the CLI.
        layout.operator("phonon_vision.export_job", icon="EXPORT")


class PV_PT_species(bpy.types.Panel):
    bl_label = "Species (colors & radii)"
    bl_idname = "PV_PT_species"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Phonon"

    @classmethod
    def poll(cls, ctx):
        return get_loaded_data(ctx.scene) is not None

    @_safe_draw
    def draw(self, context):
        layout = self.layout
        props = context.scene.phonon_vision
        if len(props.species_overrides) == 0:
            layout.label(text="No species (reload data?)")
            return

        for entry in props.species_overrides:
            box = layout.box()
            box.label(text=entry.name, icon="MATSPHERE")
            row = box.row(align=True)
            row.prop(entry, "color", text="")
            row.prop(entry, "radius", text="r (Å)")

        layout.separator()
        layout.prop(props, "species_vibrancy", slider=True)

        layout.separator()
        layout.label(text="Reset to defaults:")
        # Four one-click palette buttons in a 2×2 grid. Each is the single
        # PV_OT_reset_palette operator pre-set to a palette value, so every
        # button shares the one crash-fixed refresh path. Buttons (not a
        # dropdown) keep the valued single-click instant-apply behaviour.
        col = layout.column(align=True)
        row = col.row(align=True)
        row.operator("phonon_vision.reset_palette",
                     text="Auto (mass-based)", icon="LOOP_BACK").palette = "auto"
        row.operator("phonon_vision.reset_palette",
                     text="Jmol (CPK)", icon="COLOR").palette = "jmol"
        row = col.row(align=True)
        row.operator("phonon_vision.reset_palette",
                     text="VESTA", icon="COLOR").palette = "vesta"
        # VMD palette deferred to 0.5.4 (no authoritative source reachable in
        # the build sandbox). The slot is kept disabled so the 2×2 layout and
        # the user's muscle memory are preserved; it lights up the moment a
        # "vmd" entry is added to phonon-core's PALETTES registry.
        sub = row.row(align=True)
        sub.enabled = False
        sub.operator("phonon_vision.reset_palette",
                     text="VMD (0.5.4)", icon="COLOR").palette = "vmd"


# ============================================================
PANELS = (
    PV_PT_data,
    PV_PT_navigator,
    PV_PT_display,
    PV_PT_species,
    PV_PT_render,
)


def register():
    for c in PANELS:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(PANELS):
        bpy.utils.unregister_class(c)
