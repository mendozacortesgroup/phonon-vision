"""Render presets — apply a coherent look across world / engine / materials.

Three presets:
- TOON       : Eevee + Freestyle outlines + emissive-mixed atom shaders.
               Closest to the ascii-phonons / phonon-toolbox aesthetic the
               user wants for publication figures.
- EEVEE_PBR  : Plain principled BSDF in Eevee. Faster, more realistic.
- CYCLES_PBR : Cycles raytracing. Slower, photorealistic.

All three iterate over the per-species atom materials (``PV.Mat.<sp>``)
and the arrow material (``PV.ArrowMat``), updating their node graphs in
place. Materials' viewport_color is preserved so Solid shading still
shows the species color.
"""

from __future__ import annotations

import bpy
import numpy as np


def apply_preset(scene, preset: str) -> None:
    if preset == "TOON":
        _apply_toon(scene)
    elif preset == "EEVEE_PBR":
        _apply_eevee_pbr(scene)
    elif preset == "CYCLES_PBR":
        _apply_cycles_pbr(scene)
    else:
        raise ValueError(f"Unknown preset: {preset}")

    _ensure_render_lighting(scene)
    _apply_output_settings(scene)


def _iter_atom_materials():
    """Yield each per-species atom material (PV.Mat.<sp>)."""
    for mat in bpy.data.materials:
        if mat.name.startswith("PV.Mat.") and mat.use_nodes:
            yield mat


# ============================================================
# TOON
# ============================================================
def _apply_toon(scene) -> None:
    props = scene.phonon_vision
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"

    # Eevee samples (the property name differs between EEVEE and EEVEE_NEXT;
    # try both safely)
    _set_eevee_samples(scene, int(props.eevee_samples))

    _set_world_color(scene, (1.0, 1.0, 1.0, 1.0), strength=1.5)
    # Note: film_transparent is handled by _apply_output_settings, which
    # respects the user's `transparent_background` toggle. Toon-style
    # outlines work better with a solid background, but the user gets
    # the final say.

    # Freestyle outlines
    scene.render.use_freestyle = True
    view_layer = scene.view_layers[0]
    view_layer.use_freestyle = True
    fs = view_layer.freestyle_settings
    fs.mode = "EDITOR"
    fs.crease_angle = 2.094395  # 120°

    while len(fs.linesets):
        fs.linesets.remove(fs.linesets[0])
    ls = fs.linesets.new("PV.Outline")
    ls.select_silhouette = True
    ls.select_border = True
    ls.select_contour = True
    ls.select_crease = False
    ls.select_edge_mark = False
    ls.linestyle.thickness = 2.5
    ls.linestyle.color = (0.0, 0.0, 0.0)

    # Atom materials: BSDF + slight emission for color visibility
    for mat in _iter_atom_materials():
        _set_atom_shader_toon(mat)

    # Arrow material: bright red, slightly emissive
    arrow_mat = bpy.data.materials.get("PV.ArrowMat")
    if arrow_mat is not None and arrow_mat.use_nodes:
        _set_arrow_shader(arrow_mat, emit=True)


def _set_atom_shader_toon(mat) -> None:
    """Atom shader = MixShader(BSDF, Emission) at 35% emission so atoms stay
    bright and toon-y even with minimal lighting. Reads base color from
    material's existing diffuse_color (set at material creation)."""
    rgb = (mat.diffuse_color[0], mat.diffuse_color[1], mat.diffuse_color[2])
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    n_out = nt.nodes.new("ShaderNodeOutputMaterial")
    n_out.location = (500, 0)
    n_bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    n_bsdf.location = (50, 200)
    n_bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    n_bsdf.inputs["Roughness"].default_value = 0.45
    for sname in ("Specular IOR Level", "Specular"):
        if sname in n_bsdf.inputs:
            n_bsdf.inputs[sname].default_value = 0.3
            break
    n_emit = nt.nodes.new("ShaderNodeEmission")
    n_emit.location = (50, -100)
    n_emit.inputs["Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    n_emit.inputs["Strength"].default_value = 0.6
    n_mix = nt.nodes.new("ShaderNodeMixShader")
    n_mix.location = (300, 0)
    n_mix.inputs["Fac"].default_value = 0.35

    nt.links.new(n_bsdf.outputs["BSDF"], n_mix.inputs[1])
    nt.links.new(n_emit.outputs["Emission"], n_mix.inputs[2])
    nt.links.new(n_mix.outputs["Shader"], n_out.inputs["Surface"])


def _set_arrow_shader(mat, *, emit: bool) -> None:
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    n_out = nt.nodes.new("ShaderNodeOutputMaterial")
    n_out.location = (400, 0)
    if emit:
        n_emit = nt.nodes.new("ShaderNodeEmission")
        n_emit.location = (100, 0)
        n_emit.inputs["Color"].default_value = (0.90, 0.10, 0.10, 1.0)
        n_emit.inputs["Strength"].default_value = 1.5
        nt.links.new(n_emit.outputs["Emission"], n_out.inputs["Surface"])
    else:
        n_bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        n_bsdf.location = (100, 0)
        n_bsdf.inputs["Base Color"].default_value = (0.85, 0.10, 0.10, 1.0)
        n_bsdf.inputs["Roughness"].default_value = 0.6
        nt.links.new(n_bsdf.outputs["BSDF"], n_out.inputs["Surface"])


# ============================================================
# Eevee PBR
# ============================================================
def _apply_eevee_pbr(scene) -> None:
    props = scene.phonon_vision
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    _set_eevee_samples(scene, int(props.eevee_samples))
    _set_world_color(scene, (0.05, 0.05, 0.05, 1.0), strength=0.6)
    scene.render.use_freestyle = False
    view_layer = scene.view_layers[0]
    view_layer.use_freestyle = False

    for mat in _iter_atom_materials():
        _set_atom_shader_pbr(mat)

    arrow_mat = bpy.data.materials.get("PV.ArrowMat")
    if arrow_mat is not None and arrow_mat.use_nodes:
        _set_arrow_shader(arrow_mat, emit=False)


def _set_atom_shader_pbr(mat) -> None:
    rgb = (mat.diffuse_color[0], mat.diffuse_color[1], mat.diffuse_color[2])
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    n_out = nt.nodes.new("ShaderNodeOutputMaterial")
    n_out.location = (300, 0)
    n_bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    n_bsdf.location = (0, 0)
    n_bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    n_bsdf.inputs["Roughness"].default_value = 0.35
    nt.links.new(n_bsdf.outputs["BSDF"], n_out.inputs["Surface"])


# ============================================================
# Cycles PBR
# ============================================================
def _apply_cycles_pbr(scene) -> None:
    props = scene.phonon_vision
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(props.cycles_samples)
    scene.cycles.use_denoising = bool(props.use_denoising)
    _set_world_color(scene, (0.05, 0.05, 0.05, 1.0), strength=0.6)
    scene.render.use_freestyle = False
    view_layer = scene.view_layers[0]
    view_layer.use_freestyle = False

    for mat in _iter_atom_materials():
        _set_atom_shader_pbr(mat)

    arrow_mat = bpy.data.materials.get("PV.ArrowMat")
    if arrow_mat is not None and arrow_mat.use_nodes:
        _set_arrow_shader(arrow_mat, emit=False)


# ============================================================
# Lighting
# ============================================================
def _ensure_render_lighting(scene) -> None:
    """Add a sun light if none exists in the PhononVision collection."""
    coll = bpy.data.collections.get("PhononVision")
    if coll is None:
        return

    if bpy.data.objects.get("PV.KeyLight") is not None:
        return

    # Compute a reasonable position from the supercell bbox
    from .scene_setup import compute_supercell_bbox
    bbox = compute_supercell_bbox(scene, fallback_to_lattice=False)
    if bbox is None:
        return
    center, _size_xyz, size = bbox

    light = bpy.data.lights.new(name="PV.KeyLight_data", type="SUN")
    obj = bpy.data.objects.new(name="PV.KeyLight", object_data=light)
    coll.objects.link(obj)
    obj.location = (
        float(center[0]) + size,
        float(center[1]) - size,
        float(center[2]) + size * 1.5,
    )
    obj.rotation_euler = (0.785, 0.0, 0.785)
    obj.data.energy = 3.0


# ============================================================
# Helpers
# ============================================================
def _set_world_color(scene, color, strength=1.0) -> None:
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("PV.World")
        scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    if bg is None:
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        bg = nt.nodes.new("ShaderNodeBackground")
        out = nt.nodes.new("ShaderNodeOutputWorld")
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs["Color"].default_value = color
    bg.inputs["Strength"].default_value = strength


def _set_eevee_samples(scene, n: int) -> None:
    """Set Eevee render samples, tolerating either the old or new API."""
    e = getattr(scene, "eevee", None)
    if e is None:
        return
    # EEVEE_NEXT (Blender 4.2+) uses `samples`
    if hasattr(e, "taa_render_samples"):
        e.taa_render_samples = n
    if hasattr(e, "samples"):
        try:
            e.samples = n
        except Exception:
            pass


def _apply_output_settings(scene) -> None:
    """Apply resolution + transparency from the user's properties.

    Called by each preset so changes take effect on the next 'Apply Render
    Preset' click. Resolution percentage scales the width/height multiplicatively
    (matches Blender's own convention).
    """
    props = scene.phonon_vision
    scene.render.resolution_x = int(props.render_resolution_x)
    scene.render.resolution_y = int(props.render_resolution_y)
    scene.render.resolution_percentage = int(props.render_percentage)

    # Transparency:
    #   When transparent_background is on, ALL presets honor it — Toon's
    #   white-world setup is overridden in favor of alpha.
    #   For Eevee/Cycles PBR, transparent vs world-color is independent of
    #   the engine.
    scene.render.film_transparent = bool(props.transparent_background)
