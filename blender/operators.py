"""Operators — the actions exposed in the UI.

PV_OT_load_yaml          — load a band.yaml / mesh.yaml
PV_OT_clear              — remove all PhononVision objects from the scene
PV_OT_prev_mode          — band_index - 1 (with q-wrap)
PV_OT_next_mode          — band_index + 1 (with q-wrap)
PV_OT_prev_q             — q_index - 1
PV_OT_next_q             — q_index + 1
PV_OT_jump_to_freq       — find the mode closest to a target frequency
PV_OT_apply_render_preset — apply the selected render preset
PV_OT_setup_camera       — quick top/side/iso camera setup
PV_OT_configure_loop     — set scene.frame_* + fps to make a perfect loop
PV_OT_render_animation   — render one cycle to MP4/GIF/PNG-sequence
"""

from __future__ import annotations

import os
import subprocess
import shutil
import math

import bpy
from bpy.props import (StringProperty, FloatProperty, EnumProperty,
                       FloatVectorProperty)
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Vector, Matrix

import numpy as np

from phonon_core import load_band_yaml, load_mesh_yaml
from phonon_core.animation import recommended_render_settings

from .state import set_loaded_data, clear_loaded_data, get_loaded_data
from .scene_setup import build_scene_from_data, clear_scene, refresh_scene_for_current_mode
from .render_presets import apply_preset
from .frame_handler import update_frame


# ============================================================
# Data loading
# ============================================================
def _default_species_color(symbol: str, masses_by_species: dict,
                            all_species: list) -> tuple:
    """Match the historical default color logic (binary lighter=yellow,
    heavier=blue; otherwise from the categorical palette)."""
    n = len(all_species)
    if n == 2:
        sorted_sp = sorted(all_species, key=lambda s: (masses_by_species[s], s))
        if symbol == sorted_sp[0]:
            return (0.95, 0.80, 0.20)
        else:
            return (0.30, 0.50, 0.72)
    from phonon_core.species import assign_species_palette
    palette = assign_species_palette(all_species)
    return tuple(palette[symbol])


def _default_species_radius(mass: float) -> float:
    """Same hint used by scene_setup._atomic_radius_hint."""
    return 0.35 * (mass / 50.0) ** (1.0 / 3.0)


def _populate_species_overrides(props, material, *,
                                 preserve_overrides: bool = True,
                                 palette: str = "auto") -> None:
    """Populate ``props.species_overrides`` from the loaded material.

    Parameters
    ----------
    preserve_overrides : bool, default True
        If True (the YAML-load path), existing user-customized entries are
        preserved when the species name matches. If False (an explicit
        reset), all entries are restored to the chosen default scheme.
    palette : str, default "auto"
        Colour scheme for species without a preserved override:
          - "auto"  → mass-based categorical palette (lighter→warm,
                      heavier→cool); the default.
          - any key in ``phonon_core.radii.PALETTES`` (e.g. "jmol",
                      "vesta") → per-element table lookup via that accessor.
        An unrecognised palette name falls back to "auto". Radii are always
        mass-scaled, independent of the colour palette.
    """
    # Resolve the colour function once. "auto" is categorical (not an
    # element table) so it is handled separately from the PALETTES registry.
    color_fn = None
    if palette != "auto":
        from phonon_core.radii import PALETTES
        color_fn = PALETTES.get(palette)
        # Unknown palette → fall back to auto rather than raising mid-reset.
        # (Each accessor itself already returns mid-grey for unknown elements.)
    from .state import _SuppressSpeciesUpdates
    # Optionally remember any prior settings, keyed by species symbol
    prior: dict = {}
    if preserve_overrides:
        for entry in props.species_overrides:
            prior[entry.name] = (tuple(entry.color), float(entry.radius))

    # Suppress per-property refresh callbacks during the bulk write —
    # we'll fire a single refresh at the end. This avoids O(2N) refreshes
    # during population and the race condition on the second consecutive
    # Reset click (mesh datablocks were getting swapped mid-iteration by
    # a prior refresh that was still running).
    with _SuppressSpeciesUpdates():
        props.species_overrides.clear()

        masses_by_sp: dict = {}
        for sp, m in zip(material.symbols, material.masses):
            masses_by_sp.setdefault(sp, float(m))
        all_species = list(material.unique_species)

        for sp in all_species:
            entry = props.species_overrides.add()
            entry.name = sp
            if sp in prior:
                color, radius = prior[sp]
                entry.color = color
                entry.radius = radius
            else:
                if color_fn is not None:
                    entry.color = color_fn(sp)
                else:
                    entry.color = _default_species_color(sp, masses_by_sp, all_species)
                sp_masses = [float(material.masses[i])
                             for i, s in enumerate(material.symbols) if s == sp]
                entry.radius = _default_species_radius(float(np.mean(sp_masses)))

    # NOTE: this helper is intentionally side-effect-free now. It writes
    # the species_overrides collection and returns. Callers (both reset
    # operators, and the YAML loader → build_scene_from_data path) handle
    # any required refresh themselves. In 0.4.2 this function called
    # refresh_species_visuals at the end, but the reset operators also
    # called it explicitly — the resulting double-refresh invalidated
    # template-sphere objects on the second pass (ReferenceError:
    # StructRNA of type Object has been removed). See 0.5.1 changelog.


def load_file(scene, path):
    """Load phonopy band/mesh data into the scene's loaded-data slot.

    Headless-safe DATA load: validates, sniffs format, dispatches to the
    phonon_core loader, stores the result, resets indices, and populates the
    per-species overrides. It does NOT build the scene and does NOT touch any
    viewport — the caller does those (the GUI operator, or the future CLI).

    Raises with the same human-readable messages the GUI shows:
      - ValueError for a bad path / empty file / unrecognized extension, and
        (relayed from the parser) for malformed phonopy files;
      - ImportError for a missing dependency (e.g. h5py for .hdf5);
      - other Exception for unknown loader failures.

    Returns the loaded data object.
    """
    props = scene.phonon_vision
    if not path:
        raise ValueError("No file path set — pick a band/mesh file first")
    if not os.path.isfile(path):
        raise ValueError(f"File not found: {path}")
    try:
        size_bytes = os.path.getsize(path)
    except OSError:
        size_bytes = 0
    if size_bytes == 0:
        raise ValueError(f"File is empty: {path}")

    # Sniff: extension first, then filename heuristic (identical to the GUI).
    name = os.path.basename(path).lower()
    ext = os.path.splitext(name)[1]
    is_hdf5 = ext in (".hdf5", ".h5")
    is_yaml = ext in (".yaml", ".yml")
    if not (is_hdf5 or is_yaml):
        raise ValueError(
            f"Unrecognized extension '{ext}'. Expected .yaml, .yml, "
            f".hdf5, or .h5 (phonopy band.yaml/band.hdf5/mesh.yaml/mesh.hdf5)."
        )

    if is_hdf5:
        from phonon_core import load_band_hdf5, load_mesh_hdf5
        if "mesh" in name:
            data = load_mesh_hdf5(path)  # auto-detects sibling phonopy.yaml
            props.data_kind = "MESH"
        elif "band" in name:
            data = load_band_hdf5(path)
            props.data_kind = "BAND"
        else:
            try:
                data = load_band_hdf5(path)
                props.data_kind = "BAND"
            except Exception:
                data = load_mesh_hdf5(path)
                props.data_kind = "MESH"
    else:
        from . import motion
        if motion.sniff_phonopy_yaml(path):
            # 0.8.0 ENGINE MODE: a top-level `supercell_matrix:` key marks a
            # phonopy.yaml STRUCTURE file (band.yaml/mesh.yaml never carry
            # one) — route to the vendored A2 engine BEFORE the band/mesh
            # attempts. Errors (missing FC, the BORN refusal, parse failure)
            # propagate with the engine's own message.
            data = motion.load_engine_source(path)
            props.data_kind = "ENGINE"
        elif "mesh" in name:
            data = load_mesh_yaml(path)
            props.data_kind = "MESH"
        elif "band" in name:
            data = load_band_yaml(path)
            props.data_kind = "BAND"
        else:
            try:
                data = load_band_yaml(path)
                props.data_kind = "BAND"
            except Exception:
                data = load_mesh_yaml(path)
                props.data_kind = "MESH"

    set_loaded_data(scene, data)
    # Suppress prop-update callbacks for the post-load writes: q_index/band_index
    # reset and the species-collection population would otherwise each fire a
    # callback (refresh/rebuild) before anything is built. The caller builds
    # explicitly afterward.
    from .state import _SuppressUpdates, remember_band_hsp
    with _SuppressUpdates():
        props.q_index = 0
        props.band_index = 0
        # engine source: q starts at Gamma (the source itself evaluated
        # Gamma on load); the props must agree with the engine cache.
        props.q_frac_1 = 0.0
        props.q_frac_2 = 0.0
        props.q_frac_3 = 0.0
        _populate_species_overrides(props, data.material)
    if props.data_kind == "BAND":
        # session-remember the high-symmetry targets: the ENGINE navigator
        # offers them as jump buttons (Gamma is always offered).
        remember_band_hsp(data)
    elif props.data_kind == "ENGINE":
        # 0.8.2: engine jump targets come from the MATERIAL, never session
        # load order — a same-material sibling band file (Source 1) or, if
        # none, the reciprocal-lattice symmetry (Source 2). Always refreshes.
        _set_engine_jump_targets(data)
    return data


def _set_engine_jump_targets(data) -> None:
    """Populate the engine navigator's jump-target store from the loaded
    MATERIAL (field defect 0.8.2 — never from session load order).

    Source 1 (authoritative): a same-material sibling band file next to
    phonopy.yaml. Source 2 (fallback): high-symmetry points derived from the
    reciprocal lattice. This ALWAYS overwrites the store, so a previously
    loaded material's targets can never leak into this engine navigator.
    A bad sibling never blocks the (already succeeded) engine load — the
    discovery is non-fatal and falls through to the lattice route.
    """
    from . import hsp
    from .state import remember_band_hsp, set_band_hsp
    yaml_path = getattr(data, "yaml_path", None)
    if yaml_path:
        band, prov = hsp.discover_sibling_band(yaml_path, data.material)
        if band is not None:
            remember_band_hsp(band, provenance=prov)          # Source 1
            return
    recip = data.material.reciprocal_lattice                  # Source 2
    # slab vs orthogonal-3D-bulk decides the HONESTY of the provenance, not the
    # points (0.8.3): the supercell_matrix third row [0,0,1] marks a slab.
    is_slab = hsp.is_slab_supercell(yaml_path) if yaml_path else None
    targets = hsp.lattice_jump_targets(recip, is_slab=is_slab)
    prov = hsp.lattice_provenance(recip, is_slab=is_slab)
    set_band_hsp(targets, provenance=prov)


class PV_OT_load_yaml(Operator):
    """Load a phonopy band/mesh file. Accepts .yaml (band.yaml, mesh.yaml)
    or .hdf5 (band.hdf5, mesh.hdf5). The HDF5 path is roughly two orders
    of magnitude faster on a typical multi-MB calculation."""
    bl_idname = "phonon_vision.load_yaml"
    bl_label = "Load File"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        props = scene.phonon_vision
        path = bpy.path.abspath(props.yaml_path)
        # Path/extension guards stay in the operator so the GUI messages are
        # byte-identical. load_file() re-validates with the same messages for
        # headless callers, so these are a no-op pre-check on the GUI path.
        if not path:
            self.report({"ERROR"}, "No file path set — pick a band/mesh file first")
            return {"CANCELLED"}
        if not os.path.isfile(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}

        # Pre-flight: empty file is a sign of a partial download or a wrong
        # path; phonopy never emits zero-byte band/mesh files.
        try:
            size_bytes = os.path.getsize(path)
        except OSError:
            size_bytes = 0
        if size_bytes == 0:
            self.report({"ERROR"}, f"File is empty: {path}")
            return {"CANCELLED"}

        # Sniff: extension first, then filename heuristic
        name = os.path.basename(path).lower()
        ext = os.path.splitext(name)[1]
        is_hdf5 = ext in (".hdf5", ".h5")
        is_yaml = ext in (".yaml", ".yml")
        if not (is_hdf5 or is_yaml):
            self.report(
                {"ERROR"},
                f"Unrecognized extension '{ext}'. Expected .yaml, .yml, "
                f".hdf5, or .h5 (phonopy band.yaml/band.hdf5/mesh.yaml/mesh.hdf5).",
            )
            return {"CANCELLED"}

        # If the user picked a large YAML and an HDF5 with the same stem
        # exists next to it, hint that HDF5 is much faster. (GUI-only sugar.)
        if is_yaml and size_bytes > 5 * 1024 * 1024:
            stem = os.path.splitext(path)[0]
            for candidate in (stem + ".hdf5", stem + ".h5"):
                if os.path.isfile(candidate):
                    self.report(
                        {"INFO"},
                        f"Tip: {os.path.basename(candidate)} is next to this YAML "
                        f"— loading that instead is ~100× faster.",
                    )
                    break

        # DATA load (validate/sniff/dispatch/store) now lives in load_file();
        # relay each error class exactly as before.
        try:
            data = load_file(scene, path)
        except ImportError as e:
            # Typically h5py missing for an .hdf5 load
            self.report(
                {"ERROR"},
                f"Missing dependency: {e}. Reinstall the addon — the bundled "
                f"wheel may be incomplete.",
            )
            return {"CANCELLED"}
        except ValueError as e:
            # Parser raised — phonopy-side issue (missing eigenvectors,
            # missing material info on mesh.hdf5, malformed file, etc.).
            self.report({"ERROR"}, f"Parse failed: {e}")
            return {"CANCELLED"}
        except Exception as e:
            # Unknown failure mode — show type + message for debugging
            self.report({"ERROR"}, f"Failed to load file ({type(e).__name__}): {e}")
            return {"CANCELLED"}

        # Build scene. Capture failure but DON'T early-return — we still
        # want to apply the orthographic-projection default below, since
        # the data load itself succeeded and the user will keep working
        # with the scene either way.
        build_err: str | None = None
        try:
            build_scene_from_data(scene)
        except Exception as e:
            build_err = str(e)
            # Print full traceback to the system console so we can diagnose
            # build regressions (the operator report only shows the message).
            import traceback
            print(f"[phonon-vision] build_scene_from_data raised:")
            traceback.print_exc()

        # Switch all 3D viewports to orthographic projection. This is the
        # scientifically-correct default for crystal-structure visualization:
        # perspective foreshortening creates apparent bond-length distortions
        # that look like real but spurious displacement. The user can still
        # toggle back via Numpad 5.
        # IMPORTANT: this runs regardless of build_scene success, because
        # the data load itself succeeded and the user will want orthographic
        # framing whether or not the geometry built cleanly.
        try:
            for area in context.window.screen.areas:
                if area.type != "VIEW_3D":
                    continue
                for sp in area.spaces:
                    if sp.type == "VIEW_3D":
                        sp.region_3d.view_perspective = "ORTHO"
        except Exception as e:
            # Non-fatal; the data is loaded and the scene is built
            print(f"[phonon-vision] orthographic toggle failed: {e}")

        if build_err is not None:
            self.report({"WARNING"},
                        f"Data loaded but scene setup failed: {build_err}")
            return {"FINISHED"}

        n_q = data.num_qpoints
        n_b = data.material.num_bands
        fmt = "HDF5" if is_hdf5 else "YAML"
        self.report(
            {"INFO"},
            f"Loaded {props.data_kind} ({fmt}): {data.material.num_atoms} atoms, "
            f"{n_q} q-points, {n_b} bands"
        )
        return {"FINISHED"}


class PV_OT_clear(Operator):
    """Remove all PhononVision objects from the scene."""
    bl_idname = "phonon_vision.clear"
    bl_label = "Clear Scene"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        clear_scene(scene)
        clear_loaded_data(scene)
        scene.phonon_vision.data_kind = "NONE"
        return {"FINISHED"}


# ============================================================
# Mode navigation
# ============================================================
class PV_OT_prev_mode(Operator):
    bl_idname = "phonon_vision.prev_mode"
    bl_label = "Previous Band"
    def execute(self, context):
        scene = context.scene
        data = get_loaded_data(scene)
        if data is None:
            return {"CANCELLED"}
        props = scene.phonon_vision
        n_b = data.material.num_bands
        new_b = props.band_index - 1
        if new_b < 0:
            new_b = n_b - 1
        props.band_index = new_b
        return {"FINISHED"}


class PV_OT_next_mode(Operator):
    bl_idname = "phonon_vision.next_mode"
    bl_label = "Next Band"
    def execute(self, context):
        scene = context.scene
        data = get_loaded_data(scene)
        if data is None:
            return {"CANCELLED"}
        props = scene.phonon_vision
        n_b = data.material.num_bands
        props.band_index = (props.band_index + 1) % n_b
        return {"FINISHED"}


class PV_OT_prev_q(Operator):
    bl_idname = "phonon_vision.prev_q"
    bl_label = "Previous q"
    def execute(self, context):
        scene = context.scene
        data = get_loaded_data(scene)
        if data is None:
            return {"CANCELLED"}
        props = scene.phonon_vision
        n_q = data.num_qpoints
        props.q_index = (props.q_index - 1) % n_q
        return {"FINISHED"}


class PV_OT_next_q(Operator):
    bl_idname = "phonon_vision.next_q"
    bl_label = "Next q"
    def execute(self, context):
        scene = context.scene
        data = get_loaded_data(scene)
        if data is None:
            return {"CANCELLED"}
        props = scene.phonon_vision
        n_q = data.num_qpoints
        props.q_index = (props.q_index + 1) % n_q
        return {"FINISHED"}


class PV_OT_jump_to_high_symmetry(Operator):
    """For BandPath data: jump to a high-symmetry point by index.

    The tooltip shown on each button is computed dynamically from the
    loaded data, distinguishing:

      * a clean single label (e.g. "Γ"): just states the q-index;
      * a real path discontinuity (e.g. "X|Y"): warns that q+1 is on a
        different physical mode set because bands are sorted by frequency
        at each q-point — phonopy band-connection only tracks continuity
        within a segment, not across discontinuities;
      * a comma-artifact spurious "X|X" (typically "Γ|Γ"): explains the
        '|' comes from a comma in the band.conf BAND_LABELS line and
        both sides are the same physical point.
    """
    bl_idname = "phonon_vision.jump_high_sym"
    bl_label = "Jump to High-Symmetry Point"

    point_index: bpy.props.IntProperty(default=0)  # type: ignore

    @classmethod
    def description(cls, context, properties):
        """Compute the per-button tooltip from the loaded BandPath."""
        # Fall back to the docstring if anything goes wrong — the tooltip
        # should never raise, that breaks button rendering in Blender.
        try:
            from .state import get_loaded_data
            data = get_loaded_data(context.scene)
            if data is None or not hasattr(data, "segment_nqpoint"):
                return cls.__doc__ or ""

            # Resolve which q-index this button jumps to (same logic as execute)
            stops = [0]
            cum = 0
            for n in data.segment_nqpoint:
                cum += n
                stops.append(cum - 1)
            idx = max(0, min(int(properties.point_index), len(stops) - 1))
            q_idx = stops[idx]

            _, labels = data.tick_locations()
            raw = labels[idx] if idx < len(labels) else ""

            # Import here so a stale wheel can't break tooltip rendering
            try:
                from phonon_core.units import prettify_label
            except ImportError:
                def prettify_label(s):
                    return s

            if "|" in raw:
                parts = raw.split("|", 1)
                left = prettify_label(parts[0])
                right = prettify_label(parts[1]) if len(parts) > 1 else ""
                if left == right:
                    return (
                        f"Jump to q-point #{q_idx + 1} ({left}). "
                        f"The '|' marker in this label comes from a trailing "
                        f"comma in your band.conf BAND_LABELS line — both "
                        f"sides are {left}, so this is a single physical "
                        f"point. To remove the marker, drop the comma from "
                        f"BAND_LABELS and re-run phonopy."
                    )
                else:
                    return (
                        f"Jump to q-point #{q_idx + 1} ({left}). This is a "
                        f"path discontinuity: q+1 lands on {right}, a "
                        f"different point in the BZ. Band indices are sorted "
                        f"by frequency at each q-point, so band N at {left} "
                        f"is NOT the same physical mode as band N at "
                        f"{right}. BAND_CONNECTION tracks modes within a "
                        f"segment only."
                    )
            else:
                return f"Jump to q-point #{q_idx + 1} ({prettify_label(raw)})."
        except Exception:
            return cls.__doc__ or ""

    def execute(self, context):
        scene = context.scene
        data = get_loaded_data(scene)
        if data is None or not hasattr(data, "segment_nqpoint"):
            return {"CANCELLED"}
        props = scene.phonon_vision

        # Compute q-index of each high-symmetry stop
        stops = [0]
        cum = 0
        for n in data.segment_nqpoint:
            cum += n
            stops.append(cum - 1)
        idx = max(0, min(self.point_index, len(stops) - 1))
        props.q_index = stops[idx]
        return {"FINISHED"}


# ============================================================
# Render presets and camera
# ============================================================
class PV_OT_set_resolution_preset(Operator):
    """Quick-set render resolution to a common output size."""
    bl_idname = "phonon_vision.set_resolution_preset"
    bl_label = "Resolution preset"
    bl_options = {"REGISTER", "UNDO"}

    preset: EnumProperty(
        items=[
            ("SQUARE_1080", "1080² (Instagram)", "1080 × 1080 — square, social-friendly"),
            ("FULL_HD",     "1920 × 1080",       "16:9 Full HD — talks and presentations"),
            ("LARGE_SQUARE","1920²",             "1920 × 1920 — large square for papers"),
            ("UHD_4K",      "3840 × 2160 (4K)",  "16:9 4K — hero shots"),
        ],
        default="SQUARE_1080",
    )  # type: ignore

    def execute(self, context):
        props = context.scene.phonon_vision
        if self.preset == "SQUARE_1080":
            props.render_resolution_x, props.render_resolution_y = 1080, 1080
        elif self.preset == "FULL_HD":
            props.render_resolution_x, props.render_resolution_y = 1920, 1080
        elif self.preset == "LARGE_SQUARE":
            props.render_resolution_x, props.render_resolution_y = 1920, 1920
        elif self.preset == "UHD_4K":
            props.render_resolution_x, props.render_resolution_y = 3840, 2160
        return {"FINISHED"}


class PV_OT_apply_render_preset(Operator):
    bl_idname = "phonon_vision.apply_render_preset"
    bl_label = "Apply Render Preset"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        try:
            apply_preset(scene, scene.phonon_vision.render_preset)
        except Exception as e:
            self.report({"ERROR"}, f"Preset failed: {e}")
            return {"CANCELLED"}

        # Re-apply lighting if the bbox is known (so lighting_style takes effect
        # immediately, not just when the user re-clicks a camera button).
        # Wrapped in try/except so any custom-property read issues don't
        # abort the entire Apply Preset action — the user's chosen render
        # style has already been applied above.
        try:
            from .scene_setup import compute_supercell_bbox
            bbox = compute_supercell_bbox(scene, fallback_to_lattice=False)
            if bbox is not None:
                center, _size_xyz, max_size = bbox
                _ensure_lighting(scene, center, max_size)
        except Exception as e:
            # Lighting refresh is best-effort; the preset itself succeeded
            print(f"[phonon-vision] lighting refresh skipped: {e}")

        self.report({"INFO"}, f"Applied preset: {scene.phonon_vision.render_preset}")
        return {"FINISHED"}


class PV_OT_setup_camera(Operator):
    """Add a camera positioned for the loaded supercell, set as scene camera."""
    bl_idname = "phonon_vision.setup_camera"
    bl_label = "Setup Camera"
    bl_options = {"REGISTER", "UNDO"}

    view: EnumProperty(
        items=[
            ("SIDE", "Side (a-b plane edge-on)", "Camera along +Y looking at -Y"),
            ("TOP",  "Top (down c-axis)",        "Camera along +Z looking down"),
            ("ISO",  "Isometric",                "3/4 view"),
            ("SPHERICAL", "Spherical",           "Azimuth/elevation at radius"),
            ("MILLER", "Miller",                 "Along the (hkl) plane normal"),
        ],
        default="SIDE",
    )  # type: ignore

    def execute(self, context):
        scene = context.scene
        if get_loaded_data(scene) is None:
            self.report({"ERROR"}, "Load data first")
            return {"CANCELLED"}
        ok = apply_camera(scene, self.view)
        if not ok:
            self.report({"ERROR"},
                        "Camera setup failed (check Miller indices aren't 0,0,0 "
                        "and that data is loaded)")
            return {"CANCELLED"}
        return {"FINISHED"}


# ============================================================
# Camera placement — shared by the operator and the live property callbacks
# ============================================================
def _reciprocal(a1, a2, a3):
    """Reciprocal-lattice vectors (Ashcroft & Mermin, Solid State Physics Ch.5,
    1976). Ported from ascii-phonons camera.py::reciprocal(). The 2π and the
    cell-volume scalar wash out under normalization — we only use the direction.
    """
    denom = a1.dot(a2.cross(a3)) / (2.0 * math.pi)
    b1 = a2.cross(a3) / denom
    b2 = a3.cross(a1) / denom
    b3 = a1.cross(a2) / denom
    return b1, b2, b3


def _camera_direction(view, props, lattice):
    """Unit vector pointing FROM the structure center TO the camera.

    SPHERICAL: azimuth + POLAR angle from +Z. polar 0 -> +Z (top);
    polar 90 / az 0 -> -Y (Side preset); polar 180 -> -Z (bottom); azimuth
    sweeps around +Z.
    MILLER:    the (hkl) plane normal = h·b1 + k·b2 + l·b3 (reciprocal lattice).
    Returns None if the direction is degenerate (e.g. Miller 0,0,0).
    """
    if view == "SPHERICAL":
        th = math.radians(float(props.camera_polar))     # polar from +Z
        ph = math.radians(float(props.camera_azimuth))
        d = Vector((math.sin(th) * math.sin(ph),
                    -math.sin(th) * math.cos(ph),
                    math.cos(th)))
        return d.normalized() if d.length > 1e-9 else None
    # MILLER — lattice rows are a1,a2,a3 (phonopy row convention; same source
    # used to build PV.Cell).
    a1 = Vector((float(lattice[0][0]), float(lattice[0][1]), float(lattice[0][2])))
    a2 = Vector((float(lattice[1][0]), float(lattice[1][1]), float(lattice[1][2])))
    a3 = Vector((float(lattice[2][0]), float(lattice[2][1]), float(lattice[2][2])))
    b1, b2, b3 = _reciprocal(a1, a2, a3)
    d = (float(props.camera_miller_h) * b1
         + float(props.camera_miller_k) * b2
         + float(props.camera_miller_l) * b3)
    return d.normalized() if d.length > 1e-9 else None


def _clear_camera_track(cam_obj):
    """Remove any TRACK_TO constraint and the PV.CameraTarget empty.

    0.6.7 aims both camera paths with a directly-computed look-at rotation
    (pole-safe), so the constraint+target machinery from 0.6.6 is gone. This
    cleans up stragglers left in scenes saved under 0.6.6. Object removal only
    (the empty has no mesh) — clear of the bpy.data.*.remove() bug class.
    """
    if cam_obj is not None:
        for c in list(cam_obj.constraints):
            if c.type == 'TRACK_TO':
                cam_obj.constraints.remove(c)
    tgt = bpy.data.objects.get("PV.CameraTarget")
    if tgt is not None:
        bpy.data.objects.remove(tgt, do_unlink=True)


def _look_at_euler(location, center):
    """Deterministic, pole-safe look-at: returns rotation_euler aiming the
    camera's −Z at ``center`` with +Z up (falls back to +Y up when the view is
    near-vertical and +Z degenerates)."""
    forward = (Vector(center) - Vector(location)).normalized()   # −Z aims at center
    up = Vector((0.0, 0.0, 1.0))
    if abs(forward.dot(up)) > 0.999:
        up = Vector((0.0, 1.0, 0.0))
    right = forward.cross(up).normalized()
    true_up = right.cross(forward).normalized()
    return Matrix((
        (right.x, true_up.x, -forward.x),
        (right.y, true_up.y, -forward.y),
        (right.z, true_up.z, -forward.z),
    )).to_euler()



def apply_camera(scene, view) -> bool:
    """Place PV.Camera for ``view``. Returns True on success.

    SIDE/TOP/ISO: the original fixed location + rotation_euler framing
    (UNCHANGED). SPHERICAL/MILLER: place at center + distance*direction and aim
    with a pole-safe look-at rotation. Orthographic throughout: distance only POSITIONS the
    camera; zoom is ortho_scale (Auto-frame / Frame width), never the radius.
    """
    data = get_loaded_data(scene)
    if data is None:
        return False
    props = scene.phonon_vision

    from .scene_setup import compute_supercell_bbox
    bbox = compute_supercell_bbox(scene, fallback_to_lattice=True)
    if bbox is None:
        return False
    center, bbox_size, _max_size = bbox
    center_v = Vector((float(center[0]), float(center[1]), float(center[2])))
    # Movable look-at / orbit center: structure center + user offset (0.6.8).
    off = props.camera_target_offset
    eff_center = center_v + Vector((float(off[0]), float(off[1]), float(off[2])))

    cam = bpy.data.cameras.get("PV.Camera")
    if cam is None:
        cam = bpy.data.cameras.new("PV.Camera")
    cam_obj = bpy.data.objects.get("PV.CameraObj")
    if cam_obj is None:
        cam_obj = bpy.data.objects.new("PV.CameraObj", cam)
        scene.collection.objects.link(cam_obj)
    cam_obj.data = cam
    cam.type = "ORTHO"
    # Both paths aim with rotation_euler now; strip any 0.6.6 TRACK_TO + target.
    _clear_camera_track(cam_obj)

    if view in ("SIDE", "TOP", "ISO"):
        # ===== preset framing — UNCHANGED from 0.6.5 =====
        if props.camera_auto_frame:
            pad = 1.20
            if view == "TOP":
                ortho_w = float(max(bbox_size[0], bbox_size[1])) * pad
                dist = max(float(bbox_size[2]), 1.0) * 2.0
            elif view == "SIDE":
                ortho_w = float(max(bbox_size[0], bbox_size[2])) * pad
                dist = max(float(bbox_size[1]), 1.0) * 2.0
            else:  # ISO
                ortho_w = float(np.sqrt(bbox_size[0] ** 2 + bbox_size[1] ** 2)) * pad
                dist = float(np.linalg.norm(bbox_size)) * 1.2
            ortho_w = max(ortho_w, 3.0)
            props.camera_ortho_scale = ortho_w
            props.camera_distance = dist
        else:
            ortho_w = float(props.camera_ortho_scale)
            dist = float(props.camera_distance)
        if view == "TOP":
            location = (eff_center[0], eff_center[1], eff_center[2] + dist)
            rotation = (0.0, 0.0, 0.0)
        elif view == "SIDE":
            location = (eff_center[0], eff_center[1] - dist, eff_center[2])
            rotation = (1.5708, 0.0, 0.0)
        else:  # ISO
            location = (eff_center[0] + dist * 0.6, eff_center[1] - dist * 0.6,
                        eff_center[2] + dist * 0.5)
            rotation = (1.10, 0.0, 0.785)
        cam.ortho_scale = ortho_w
        cam_obj.location = location
        cam_obj.rotation_euler = rotation
    else:
        # ===== SPHERICAL / MILLER =====
        direction = _camera_direction(view, props, data.material.lattice)
        if direction is None:
            return False   # degenerate (e.g. Miller 0,0,0)
        # Zoom (ortho_scale): full bbox diagonal fits from any angle.
        if props.camera_auto_frame:
            ortho_w = max(float(np.linalg.norm(bbox_size)) * 1.10, 3.0)
            props.camera_ortho_scale = ortho_w
        else:
            ortho_w = float(props.camera_ortho_scale)
        # Distance only positions the (ortho) camera.
        if view == "SPHERICAL":
            distance = float(props.camera_radius)
        else:  # MILLER -> the existing camera_distance (auto or manual)
            if props.camera_auto_frame:
                distance = max(float(np.linalg.norm(bbox_size)) * 2.0, 1.0)
                props.camera_distance = distance
            else:
                distance = float(props.camera_distance)
        cam.ortho_scale = ortho_w
        cam_obj.location = eff_center + distance * direction
        # Deterministic, pole-safe look-at at the (possibly offset) center.
        cam_obj.rotation_euler = _look_at_euler(cam_obj.location, eff_center)

    scene.camera = cam_obj
    _ensure_lighting(scene, center, float(bbox_size.max()))
    return True


def _ensure_lighting(scene, center, size: float) -> None:
    """Build the lighting rig for the scene, based on props.lighting_style.

    Three styles:

    - STUDIO:     Three-point soft lighting (default). Good for general
                  presentation renders. Mild shadows.
    - SCIENTIFIC: Flat, even ambient lighting from many directions. No
                  hard shadows — atoms read cleanly without the shadows
                  visually mis-suggesting which is "in front". Recommended
                  for publication figures and physics-accurate renders.
    - DRAMATIC:   Single hard key + dark fill. High-contrast shadows for
                  emphasis. Useful for talks and emphasis frames.

    Lights live in the PhononVision collection so the user can delete or
    customize them.
    """
    coll = bpy.data.collections.get("PhononVision")
    if coll is None:
        return

    style = scene.phonon_vision.lighting_style
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

    # Always clean prior PV lights so style switching works.
    #
    # Lifecycle note (same bug class as the 0.5.2 species-mesh fix): removing
    # a light OBJECT with do_unlink=True does NOT delete its light DATA — it
    # orphans it (a mesh-object can't hold NULL data, so Blender keeps the
    # datablock alive at 0 users), exactly like mesh data after object removal.
    # If we stop here, the next _new_light()'s `bpy.data.lights.new(name +
    # "_data")` name-collides with that surviving orphan and gets auto-suffixed
    # to "..._data.001", ".002", ... — a slow orphan leak across style switches.
    # The fix mirrors build-new → swap → remove-old-by-name: remove the OBJECT
    # first (the safe direction — never cascades to other objects, and drops
    # the light data to 0 users), THEN re-fetch that data by name and remove it
    # while it is unbound. Removing the data first would cascade-delete the live
    # object (the StructRNA crash class); removing it after, at 0 users, cannot.
    # Reaping every switch keeps exactly zero orphans, so the data name stays
    # suffix-free ("PV.KeyLight_data") indefinitely.
    for name in ("PV.KeyLight", "PV.FillLight", "PV.BackLight",
                 "PV.AmbN", "PV.AmbS", "PV.AmbE", "PV.AmbW", "PV.AmbTop"):
        old = bpy.data.objects.get(name)
        if old is not None:
            data_name = old.data.name if old.data is not None else None
            bpy.data.objects.remove(old, do_unlink=True)
            if data_name is not None:
                old_data = bpy.data.lights.get(data_name)
                if old_data is not None and old_data.users == 0:
                    bpy.data.lights.remove(old_data)

    def _new_light(name, light_type, energy, location, rotation=(0, 0, 0), area_size=None):
        light = bpy.data.lights.new(name=name + "_data", type=light_type)
        obj = bpy.data.objects.new(name=name, object_data=light)
        coll.objects.link(obj)
        obj.location = location
        obj.rotation_euler = rotation
        obj.data.energy = energy
        if area_size is not None and light_type == "AREA":
            obj.data.size = area_size
        return obj

    if style == "SCIENTIFIC":
        # Five area lights pointing inward from N, S, E, W and Top.
        # Energy per light is modest; the sum gives even illumination with
        # very soft, almost imperceptible shadows.
        d = size * 1.6
        big = size * 3.0    # very large area light → very soft shadows
        per_light_energy = size * size * 4.0     # area-scaled
        # N (back: +y), looking toward -y
        _new_light("PV.AmbN", "AREA", per_light_energy,
                   location=(cx, cy + d, cz),
                   rotation=(-1.5708, 0.0, 0.0),
                   area_size=big)
        # S (front: -y), looking toward +y
        _new_light("PV.AmbS", "AREA", per_light_energy,
                   location=(cx, cy - d, cz),
                   rotation=(1.5708, 0.0, 0.0),
                   area_size=big)
        # E (+x), looking toward -x
        _new_light("PV.AmbE", "AREA", per_light_energy,
                   location=(cx + d, cy, cz),
                   rotation=(0.0, 1.5708, 0.0),
                   area_size=big)
        # W (-x), looking toward +x
        _new_light("PV.AmbW", "AREA", per_light_energy,
                   location=(cx - d, cy, cz),
                   rotation=(0.0, -1.5708, 0.0),
                   area_size=big)
        # Top (+z), looking down
        _new_light("PV.AmbTop", "AREA", per_light_energy * 1.3,
                   location=(cx, cy, cz + d),
                   rotation=(0.0, 0.0, 0.0),
                   area_size=big)

    elif style == "DRAMATIC":
        # One hard sun key (bright), one weak area fill
        _new_light("PV.KeyLight", "SUN", energy=5.0,
                   location=(cx + size, cy - size, cz + size * 1.5),
                   rotation=(0.785, 0.0, 0.785))
        _new_light("PV.FillLight", "AREA", energy=size * size * 0.3,
                   location=(cx - size * 1.5, cy - size, cz + size),
                   rotation=(0.785, 0.0, -0.785),
                   area_size=size * 2.0)

    else:  # STUDIO (default)
        _new_light("PV.KeyLight", "SUN", energy=3.0,
                   location=(cx + size, cy - size, cz + size * 1.5),
                   rotation=(0.785, 0.0, 0.785))
        _new_light("PV.FillLight", "AREA", energy=size * size * 0.8,
                   location=(cx - size, cy - size, cz + size),
                   rotation=(0.785, 0.0, -0.785),
                   area_size=size * 1.5)
        _new_light("PV.BackLight", "AREA", energy=size * size * 0.4,
                   location=(cx, cy + size, cz + size * 0.5),
                   rotation=(-0.785, 0.0, 0.0),
                   area_size=size * 1.5)


# ============================================================
# Loop / render configuration
# ============================================================
class PV_OT_configure_loop(Operator):
    """Set scene.frame_start/end/fps to produce a seamless one-cycle loop."""
    bl_idname = "phonon_vision.configure_loop"
    bl_label = "Configure Loop"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        props = scene.phonon_vision
        s = recommended_render_settings(
            frames_per_cycle=int(props.frames_per_cycle),
            fps=int(props.fps_render),
            frame_start=1,
        )
        scene.frame_start = s.frame_start
        scene.frame_end = s.frame_end
        scene.render.fps = s.fps
        # Note: resolution is controlled separately by the user via the
        # Render panel's resolution / transparency fields, applied through
        # 'Apply Render Preset'.
        self.report(
            {"INFO"},
            f"Loop set: frames {s.frame_start}-{s.frame_end} @ {s.fps}fps "
            f"({s.cycle_duration_seconds:.2f}s/loop)"
        )
        return {"FINISHED"}


# ============================================================
# Render animation
# ============================================================
# A render_complete handler trampoline for the non-blocking render path.
# When the user clicks "Render Animation", we:
#   1. Configure scene/preset/output
#   2. Register a one-shot handler that converts MP4→GIF when render finishes
#   3. Trigger bpy.ops.render.render('INVOKE_DEFAULT', ...) so Blender's
#      modal render window opens and the UI stays responsive
#
# The handler must be a module-level function (Blender's persistent-handler
# decorator requires it) but it pulls its parameters from a small dict.
_pending_gif_jobs: list[dict] = []


def _do_ffmpeg_gif(mp4_path: str, gif_path: str, out_dir: str,
                   base_name: str) -> tuple[bool, str]:
    """Run the two-pass ffmpeg palettegen + paletteuse pipeline.

    Returns (success, message). Used by the render_complete handler.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False, f"ffmpeg not on PATH; left MP4 at {mp4_path}"
    palette = os.path.join(out_dir, f"{base_name}_palette.png")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", mp4_path,
             "-vf", "fps=30,palettegen", palette],
            check=True, capture_output=True,
        )
        subprocess.run(
            [ffmpeg, "-y", "-i", mp4_path, "-i", palette,
             "-filter_complex", "fps=30[x];[x][1:v]paletteuse", gif_path],
            check=True, capture_output=True,
        )
        try:
            os.remove(palette)
        except OSError:
            pass
        return True, f"Wrote {gif_path}"
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="ignore")[:200] if e.stderr else "(no stderr)"
        return False, f"ffmpeg failed: {err}"


@bpy.app.handlers.persistent
def _on_render_complete(scene, depsgraph=None):
    """Trampoline run by Blender after every render_complete event.

    We pop any pending GIF-conversion jobs and run them. Other handlers
    on the same event are not affected. Self-unregisters once the job
    queue is empty so we don't run on unrelated future renders.
    """
    while _pending_gif_jobs:
        job = _pending_gif_jobs.pop(0)
        ok, msg = _do_ffmpeg_gif(
            job["mp4"], job["gif"], job["out_dir"], job["base"],
        )
        # Print to console — operator's self.report() context is gone by now.
        print(f"[phonon-vision] GIF post-process: {msg}")

    # Unregister so this handler doesn't pile up across renders.
    handlers = bpy.app.handlers.render_complete
    if _on_render_complete in handlers:
        try:
            handlers.remove(_on_render_complete)
        except ValueError:
            pass


class PV_OT_render_animation(Operator):
    """Render the current scene as a one-cycle looped video.

    Non-blocking: opens Blender's standard render-progress window so the
    UI stays responsive and the user can cancel mid-render. GIF output
    triggers an automatic ffmpeg conversion via a render_complete handler
    when the underlying MP4 render finishes.
    """
    bl_idname = "phonon_vision.render_animation"
    bl_label = "Render Animation"

    def execute(self, context):
        scene = context.scene
        props = scene.phonon_vision
        data = get_loaded_data(scene)
        if data is None:
            self.report({"ERROR"}, "Load data first")
            return {"CANCELLED"}

        # Configure loop
        bpy.ops.phonon_vision.configure_loop()
        # Apply preset (sets engine, world, freestyle)
        try:
            apply_preset(scene, props.render_preset)
        except Exception as e:
            self.report({"WARNING"}, f"Preset failed: {e}")

        # Output filepath
        out_dir = bpy.path.abspath(props.output_directory)
        os.makedirs(out_dir, exist_ok=True)
        base = f"PV_q{props.q_index:04d}_b{props.band_index:02d}"

        fmt = props.output_format
        if fmt == "PNG":
            scene.render.image_settings.file_format = "PNG"
            scene.render.filepath = os.path.join(out_dir, f"{base}_")
            target_msg = os.path.join(out_dir, f"{base}_####.png")
            gif_target = None
        elif fmt == "MP4":
            scene.render.image_settings.file_format = "FFMPEG"
            scene.render.ffmpeg.format = "MPEG4"
            scene.render.ffmpeg.codec = "H264"
            scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
            scene.render.filepath = os.path.join(out_dir, f"{base}.mp4")
            target_msg = scene.render.filepath
            gif_target = None
        elif fmt == "GIF":
            # Render to MP4 first, post-convert with ffmpeg in render_complete
            scene.render.image_settings.file_format = "FFMPEG"
            scene.render.ffmpeg.format = "MPEG4"
            scene.render.ffmpeg.codec = "H264"
            scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
            scene.render.filepath = os.path.join(out_dir, f"{base}.mp4")
            gif_target = os.path.join(out_dir, f"{base}.gif")
            target_msg = gif_target
        else:
            self.report({"ERROR"}, f"Unknown format: {fmt}")
            return {"CANCELLED"}

        # Schedule the GIF post-process if needed, and arm the handler
        if fmt == "GIF":
            _pending_gif_jobs.append({
                "mp4": scene.render.filepath,
                "gif": gif_target,
                "out_dir": out_dir,
                "base": base,
            })
            if _on_render_complete not in bpy.app.handlers.render_complete:
                bpy.app.handlers.render_complete.append(_on_render_complete)

        # Fire the render in INVOKE mode so Blender's render window opens
        # and the UI thread isn't blocked. The user can cancel from the
        # standard render dialog.
        try:
            bpy.ops.render.render("INVOKE_DEFAULT", animation=True)
        except Exception as e:
            self.report({"ERROR"}, f"Render failed: {e}")
            # If we armed the handler, disarm it
            if fmt == "GIF" and _pending_gif_jobs:
                _pending_gif_jobs.clear()
                if _on_render_complete in bpy.app.handlers.render_complete:
                    bpy.app.handlers.render_complete.remove(_on_render_complete)
            return {"CANCELLED"}

        self.report({"INFO"}, f"Render started → {target_msg}")
        return {"FINISHED"}


def set_render_device(scene, device: str = "CPU", backend: str = "AUTO") -> str:
    """Configure the Cycles render device for THIS headless process only.

    Returns a short human summary for the batch log. HEADLESS-ONLY: it edits the
    in-memory Cycles addon prefs + scene.cycles.device and NEVER calls
    save_userpref(), so it cannot leak into the user's GUI preferences.

    - "CPU"          -> scene.cycles.device = 'CPU'.
    - "GPU" / "AUTO" -> try backends (AUTO = OPTIX then CUDA); enable the GPU
      devices for the first backend that has any, set device='GPU'. If none:
      AUTO silently falls back to CPU; explicit GPU warns (but never crashes).
    """
    if device == "CPU":
        try:
            scene.cycles.device = 'CPU'
        except Exception:
            pass
        return "CPU"

    # GPU or AUTO
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except (KeyError, AttributeError):
        try:
            scene.cycles.device = 'CPU'
        except Exception:
            pass
        print("[phonon-vision] WARNING: 'cycles' addon prefs unavailable; using CPU")
        return "CPU (cycles prefs unavailable)"

    backends = ["OPTIX", "CUDA"] if backend == "AUTO" else [backend]
    for b in backends:
        try:
            prefs.compute_device_type = b   # enum; raises if unsupported in build
        except Exception:
            continue
        # Refresh the device list (get_devices is the documented call; some 4.x
        # builds prefer refresh_devices).
        try:
            prefs.get_devices()
        except Exception:
            try:
                prefs.refresh_devices()
            except Exception:
                pass
        gpu_devs = [d for d in prefs.devices if d.type == b]
        if gpu_devs:
            for d in gpu_devs:
                d.use = True
            scene.cycles.device = 'GPU'
            return f"GPU ({b}: {len(gpu_devs)}x {gpu_devs[0].name})"

    # No GPU/backend available — fall back to CPU, never crash.
    try:
        scene.cycles.device = 'CPU'
    except Exception:
        pass
    if device == "AUTO":
        return "CPU (no GPU found; AUTO fell back)"
    print("[phonon-vision] WARNING: GPU requested but NONE available; using CPU")
    return "CPU (GPU requested but NONE available)"


def render_headless(scene, *, animation: bool, output_path: str | None = None,
                    file_format: str | None = None) -> str:
    """Render the scene SYNCHRONOUSLY in EXEC context (background-safe).

    Unlike the GUI operator, this does NOT open a render window, uses no
    render_complete handler, configures no loop, and applies no preset — the
    caller sets style/samples/camera/frame-range/etc. on the scene first.

    - animation=False: a single still (current frame) as PNG; output_path is the
      .png file. Returns the resolved file path.
    - animation=True : a PNG FRAME SEQUENCE (frames-first). output_path is the
      sequence prefix (e.g. '<dir>/frame_'); Blender writes frame_0001.png … .
      Encoding to MP4/GIF is done by the OUTER launcher (pv_render) with ffmpeg,
      so a render can never be lost to a bad in-Blender container. file_format
      is ignored here for animation — frames are always PNG. Returns the prefix.

    No overlay burn-in and no GPU device selection here.
    """
    r = scene.render

    if not animation:
        r.image_settings.file_format = "PNG"
        if output_path:
            r.filepath = output_path
        bpy.ops.render.render(write_still=True)   # EXEC, synchronous; current frame
        return bpy.path.abspath(r.filepath)

    # Frames-first: always a PNG sequence. The caller set the frame range.
    r.image_settings.file_format = "PNG"
    if output_path:
        r.filepath = output_path                  # sequence prefix
    bpy.ops.render.render(animation=True)         # EXEC, synchronous; no INVOKE
    return bpy.path.abspath(r.filepath)


class PV_OT_export_job(Operator, ExportHelper):
    """Export the current GUI settings as a headless job file (JSON).

    Dial in the look in the GUI, export, then batch that job headless with
    tools/pv_inner.py. The written job round-trips back through
    job.apply_job_to_scene.
    """
    bl_idname = "phonon_vision.export_job"
    bl_label = "Export Job File"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})  # type: ignore

    def execute(self, context):
        import json
        from .job import scene_to_job
        try:
            job = scene_to_job(context.scene)
            with open(self.filepath, "w") as fh:
                json.dump(job, fh, indent=2)
        except Exception as e:
            self.report({"ERROR"}, f"Export failed: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Wrote job: {self.filepath}")
        return {"FINISHED"}


# ============================================================
# Rebuild — force a full rebuild of all PhononVision objects
# ============================================================
class PV_OT_rebuild_scene(Operator):
    """Manually rebuild atoms / arrows / cell from the loaded data.

    Use this when a property update didn't refresh visually for any reason
    (e.g., changing supercell size while playback is paused on a frame
    Blender hasn't redrawn). The math is recomputed from scratch.
    """
    bl_idname = "phonon_vision.rebuild_scene"
    bl_label = "Rebuild Scene"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        if get_loaded_data(scene) is None:
            self.report({"ERROR"}, "Load data first")
            return {"CANCELLED"}
        try:
            build_scene_from_data(scene)
        except Exception as e:
            # Print full traceback so we can diagnose regressions from the
            # system console — the operator's report() truncates and
            # doesn't help track down where the failure actually originated.
            import traceback
            print(f"[phonon-vision] rebuild_scene raised:")
            traceback.print_exc()
            self.report({"ERROR"}, f"Rebuild failed: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, "Scene rebuilt")
        return {"FINISHED"}


class PV_OT_reset_palette(Operator):
    """Reset species colours to a chosen palette (radii stay mass-scaled).

    One operator, one refresh path. The ``palette`` property selects the
    scheme; the four UI buttons are just this operator pre-set to each value
    (Blender passes operator properties through ``layout.operator(...).prop``).
    Routing every reset through a single ``execute`` keeps the crash-fixed
    0.5.2 refresh path single-sourced — there is exactly one call site for
    ``refresh_species_visuals`` here, so the build-new→swap→remove-old-by-name
    mesh lifecycle can never diverge between palettes.

      - "auto"  → categorical / mass-based assignment (unchanged behaviour).
      - "jmol"  → Jmol CPK element colours (carbon gray). This is the exact
                  palette the old "VESTA (Jmol palette)" button applied; only
                  its label and the underlying function name changed.
      - "vesta" → real VESTA element colours (carbon brown), genuinely
                  distinct from Jmol.
    (A "vmd" value will slot in here unchanged once phonon-core ships a VMD
    table — see PALETTES registry; deferred to 0.5.4 pending a source.)
    """
    bl_idname = "phonon_vision.reset_palette"
    bl_label = "Reset palette"
    bl_options = {"REGISTER", "UNDO"}

    palette: bpy.props.EnumProperty(
        name="Palette",
        items=[
            ("auto", "Auto (mass-based)", "Mass-based categorical colours"),
            ("jmol", "Jmol (CPK)", "Jmol CPK element colours (carbon gray)"),
            ("vesta", "VESTA", "Real VESTA element colours (carbon brown)"),
            # "vmd" is a valid enum value so the (currently disabled) VMD UI
            # button can set it at draw time without raising. Its accessor is
            # not yet in PALETTES, so execute() falls back to "auto" — but the
            # button is disabled, so this path can't be reached until 0.5.4.
            ("vmd", "VMD", "VMD element colours (deferred to 0.5.4)"),
        ],
        default="auto",
    )

    @classmethod
    def description(cls, context, properties):
        return {
            "auto":  "Reset to auto (mass-based) colours; lighter→warm, heavier→cool",
            "jmol":  "Reset to Jmol CPK element colours (carbon gray)",
            "vesta": "Reset to real VESTA element colours (carbon brown)",
        }.get(properties.palette, "Reset species colours")

    def execute(self, context):
        scene = context.scene
        data = get_loaded_data(scene)
        if data is None:
            return {"CANCELLED"}
        _populate_species_overrides(scene.phonon_vision, data.material,
                                     preserve_overrides=False,
                                     palette=self.palette)
        from .scene_setup import refresh_species_visuals
        refresh_species_visuals(scene)
        return {"FINISHED"}


# ============================================================
# Viewport shading switch
# ============================================================
class PV_OT_set_viewport_shading(Operator):
    """Set the active 3D viewport to Material Preview shading.

    Atom colors only show in Material Preview or Rendered modes — Solid
    shading shows everything in a flat default color regardless of material.
    This operator finds the active 3D viewport and switches it.
    """
    bl_idname = "phonon_vision.set_viewport_shading"
    bl_label = "Show colors (Material Preview)"
    bl_options = {"REGISTER", "UNDO"}

    mode: bpy.props.EnumProperty(
        items=[
            ("MATERIAL", "Material Preview", "Realtime materials with default HDRI"),
            ("RENDERED", "Rendered", "Live render preview (slower)"),
            ("SOLID",    "Solid (no colors)", "Default solid shading (atoms appear grey)"),
        ],
        default="MATERIAL",
    )  # type: ignore

    def execute(self, context):
        n = 0
        for area in context.window.screen.areas:
            if area.type == "VIEW_3D":
                for sp in area.spaces:
                    if sp.type == "VIEW_3D":
                        sp.shading.type = self.mode
                        n += 1
        if n == 0:
            self.report({"WARNING"}, "No 3D viewport found")
            return {"CANCELLED"}
        return {"FINISHED"}


# ============================================================
# Registration
# ============================================================
class PV_OT_center_camera_target(Operator):
    """Reset the camera target offset to the structure center (0,0,0)."""
    bl_idname = "phonon_vision.center_camera_target"
    bl_label = "Center on structure"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        # Setting the offset fires _on_camera_param_change, which live-re-aims
        # an existing camera back onto the structure center.
        context.scene.phonon_vision.camera_target_offset = (0.0, 0.0, 0.0)
        return {"FINISHED"}


class PV_OT_import_bonds(Operator, ImportHelper):
    """Import a navigator bond side-car ("phonon-vision-bonds" JSON).

    Curate bonds once in the web navigator (its Bonds panel exports the
    file), import here: the file's pair table REPLACES the covalent-
    tolerance criterion entirely — pairs absent from the file form no
    bonds. Formula/natoms mismatches warn but apply (legitimate cross-
    calculation reuse). Rejected files (bad JSON, wrong format, missing
    bonds table) leave the current state untouched.
    """
    bl_idname = "phonon_vision.import_bonds"
    bl_label = "Import Bonds JSON"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})  # type: ignore

    def execute(self, context):
        import datetime
        import json
        from . import bonds as bondcore
        from .scene_setup import refresh_bonds
        scene = context.scene
        props = scene.phonon_vision
        try:
            with open(self.filepath, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as e:
            self.report({"ERROR"}, f"Could not read {self.filepath}: {e}")
            return {"CANCELLED"}
        try:
            spec, warnings = bondcore.parse_sidecar(text)
        except bondcore.SidecarError as e:
            # Contract: rejection leaves state untouched.
            self.report({"ERROR"}, f"Rejected bond side-car: {e}")
            return {"CANCELLED"}

        data = get_loaded_data(scene)
        if data is not None:
            formula = "".join(data.material.symbols)
            warnings += bondcore.structure_warnings(
                spec, formula, len(data.material.symbols))

        # Provenance travels with the spec (and into schema-v2 jobs).
        spec["source"] = {"filename": os.path.basename(self.filepath),
                          "imported": datetime.date.today().isoformat()}
        props.bonds_spec_json = json.dumps(spec)
        refresh_bonds(scene)   # bond list + geometry; atom slots unchanged
        for w in warnings:
            self.report({"WARNING"}, w)
        self.report({"INFO"},
                    f"Imported {len(spec['bonds'])} bond pair(s) from "
                    f"{os.path.basename(self.filepath)}")
        return {"FINISHED"}


class PV_OT_clear_bonds(Operator):
    """Clear the imported bond side-car — the covalent bond-tolerance
    criterion takes over again."""
    bl_idname = "phonon_vision.clear_bonds"
    bl_label = "Clear Imported Bonds"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.phonon_vision
        if not props.bonds_spec_json:
            self.report({"INFO"}, "No imported bond spec to clear")
            return {"CANCELLED"}
        props.bonds_spec_json = ""
        from .scene_setup import refresh_bonds
        refresh_bonds(context.scene)
        self.report({"INFO"}, "Cleared imported bonds — default criterion restored")
        return {"FINISHED"}


class PV_OT_engine_jump(Operator):
    """Jump the engine q to a named target (Gamma or a remembered
    high-symmetry point from the last loaded band file)."""
    bl_idname = "phonon_vision.engine_jump"
    bl_label = "Jump to q"

    label: StringProperty(default="")
    q: FloatVectorProperty(size=3, default=(0.0, 0.0, 0.0))

    def execute(self, context):
        scene = context.scene
        data = get_loaded_data(scene)
        if data is None or not hasattr(data, "set_q"):
            return {"CANCELLED"}
        props = scene.phonon_vision
        from .state import _SuppressUpdates
        with _SuppressUpdates():          # ONE refresh, not three
            props.q_frac_1 = float(self.q[0])
            props.q_frac_2 = float(self.q[1])
            props.q_frac_3 = float(self.q[2])
        from .state import _on_engine_q_change   # fire the single refresh
        _on_engine_q_change(context)
        return {"FINISHED"}


CLASSES = (
    PV_OT_load_yaml,
    PV_OT_clear,
    PV_OT_rebuild_scene,
    PV_OT_reset_palette,
    PV_OT_prev_mode,
    PV_OT_next_mode,
    PV_OT_prev_q,
    PV_OT_next_q,
    PV_OT_jump_to_high_symmetry,
    PV_OT_engine_jump,
    PV_OT_set_resolution_preset,
    PV_OT_apply_render_preset,
    PV_OT_setup_camera,
    PV_OT_center_camera_target,
    PV_OT_set_viewport_shading,
    PV_OT_configure_loop,
    PV_OT_render_animation,
    PV_OT_export_job,
    PV_OT_import_bonds,
    PV_OT_clear_bonds,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
