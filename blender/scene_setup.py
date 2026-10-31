"""Build and update the Blender scene from phonon-core data.

Architecture
------------
Atoms are rendered via **per-species vertex meshes + DupliVerts**, not
Geometry Nodes. This gives reliable colored display in *all* viewport
modes (Solid, Material Preview, Rendered) without depending on GN's
realize-instances quirks.

Scene contents (all under the ``PhononVision`` collection):

- ``PV.Atoms.<Species>``  : one vertex mesh per chemical species (e.g.
                            PV.Atoms.W, PV.Atoms.Se). Vertices = atom
                            positions in the current frame. Has a template
                            sphere child (PV.Sphere.<Species>) instanced
                            via ``instance_type = 'VERTS'``.
- ``PV.Sphere.<Species>`` : the template sphere mesh that gets instanced
                            on every vertex of its parent. Carries the
                            material with the species color.
- ``PV.Arrows``           : Mesh built directly in Python (compound
                            shaft+head per atom), rebuilt each frame so
                            arrows track the current displacement direction.
- ``PV.Cell``             : Cell outline. Mode = PRIMITIVE | GRID | SUPERCELL.
"""

from __future__ import annotations

from typing import Optional
import json

import bpy
import numpy as np
from mathutils import Vector

from . import bonds as bondcore
from . import motion

from .state import get_loaded_data, PhononVisionProps

COLLECTION_NAME = "PhononVision"
ATOMS_PREFIX = "PV.Atoms."     # one per species, e.g. PV.Atoms.W
SPHERE_PREFIX = "PV.Sphere."   # template spheres
ARROWS_OBJ = "PV.Arrows"
BONDS_OBJ = "PV.Bonds"
CELL_OBJ = "PV.Cell"


# ============================================================
# Collection helpers
# ============================================================
def get_or_create_collection(scene) -> bpy.types.Collection:
    coll = bpy.data.collections.get(COLLECTION_NAME)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION_NAME)
        scene.collection.children.link(coll)
    return coll


# ============================================================
# Shared geometry helper: supercell bounding box
# ============================================================
def compute_supercell_bbox(scene, *, fallback_to_lattice: bool = True
                            ) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Compute the axis-aligned bounding box of all PV.Atoms.<sp> objects.

    Used by:
      - camera framing (PV_OT_setup_camera) — needs center + per-axis size
      - lighting placement (apply_render_preset → _ensure_lighting) — needs
        center + max size
      - any future code that wants to know how big the scene is

    Reads the stashed equilibrium positions from each species object's
    custom property ``base_pos`` (flat list of 3N floats) and
    ``n_atoms_species`` (int). These are populated in
    ``create_atom_objects``; if no atom objects exist yet and
    ``fallback_to_lattice`` is True, we fall back to the supercell lattice
    diagonal.

    Parameters
    ----------
    scene : bpy.types.Scene
    fallback_to_lattice : bool, default True
        If no PV.Atoms.<sp> objects are present, fall back to computing
        the bbox from the supercell lattice vectors. Set False to get
        ``None`` instead.

    Returns
    -------
    (center, size_xyz, max_size) tuple or None
        ``center`` : (3,) ndarray, midpoint of the bbox (Å, world space)
        ``size_xyz`` : (3,) ndarray, per-axis extent
        ``max_size`` : float, max(size_xyz), guaranteed ≥ 1.0
    """
    positions: list[np.ndarray] = []
    for obj in bpy.data.objects:
        if not obj.name.startswith(ATOMS_PREFIX):
            continue
        if "base_pos" not in obj.keys() or "n_atoms_species" not in obj.keys():
            continue
        try:
            n = int(obj["n_atoms_species"])
            arr = np.asarray(list(obj["base_pos"]), dtype=np.float64)
            if arr.size == 3 * n:
                positions.append(arr.reshape(n, 3))
        except Exception:
            # Skip a species whose custom-property read failed rather than
            # bringing down camera/lighting framing entirely.
            continue

    if positions:
        pos = np.vstack(positions)
        bbox_min = pos.min(axis=0)
        bbox_max = pos.max(axis=0)
        center = 0.5 * (bbox_min + bbox_max)
        size_xyz = bbox_max - bbox_min
        max_size = max(float(size_xyz.max()), 1.0)
        return center, size_xyz, max_size

    if not fallback_to_lattice:
        return None

    # Lattice-based fallback (no atom objects yet)
    data = get_loaded_data(scene)
    if data is None:
        return None
    nx, ny, nz = scene.phonon_vision.supercell_n[:]
    lat = np.asarray(data.material.lattice, dtype=np.float64)
    diag = nx * lat[0] + ny * lat[1] + nz * lat[2]
    center = 0.5 * diag
    size_xyz = np.abs(diag)
    max_size = max(float(size_xyz.max()), 1.0)
    return center, size_xyz, max_size


# ============================================================
# Color and size logic
# ============================================================
def _species_color_rgb(symbol: str, masses_by_species: dict[str, float],
                       all_species: list[str]) -> tuple[float, float, float]:
    n = len(all_species)
    if n == 2:
        sorted_sp = sorted(all_species, key=lambda s: (masses_by_species[s], s))
        if symbol == sorted_sp[0]:
            return (0.95, 0.80, 0.20)    # bright yellow — lighter
        else:
            return (0.30, 0.50, 0.72)    # blue — heavier
    from phonon_core.species import assign_species_palette
    palette = assign_species_palette(all_species)
    return tuple(palette[symbol])


def _atomic_radius_hint(mass: float) -> float:
    return 0.35 * (mass / 50.0) ** (1.0 / 3.0)


# ============================================================
# Per-species template sphere + material
# ============================================================
def _get_or_build_species_material(species: str, rgb: tuple[float, float, float],
                                   emission_strength: float = 0.30) -> bpy.types.Material:
    """Build the per-species shader material.

    Looks good in:
      - Solid shading mode (uses ``diffuse_color`` directly)
      - Material Preview mode (uses Principled BSDF with the tuned settings below)
      - Rendered mode (same BSDF, with proper lighting from the render presets)

    The tuning targets a "publication-style molecular render" look (cf. VESTA
    exports, ascii-phonons output): saturated base color, slightly glossy
    highlight, with a faint emission tinted to the base color so that
    Material Preview's HDRI indirect light doesn't wash out species identity.
    """
    name = f"PV.Mat.{species}"
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    n_out = nt.nodes.new("ShaderNodeOutputMaterial")
    n_out.location = (300, 0)
    n_bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    n_bsdf.location = (0, 0)
    n_bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    # Lower roughness → sharper specular highlight → reads as more saturated
    n_bsdf.inputs["Roughness"].default_value = 0.30

    # Faint base-color-tinted self-emission so that Material Preview's
    # default HDRI (which adds a lot of indirect white light) doesn't wash
    # the species color out. Controlled by the "Vibrancy" slider in the
    # Species panel (species_vibrancy prop); default 0.30. Higher = more
    # self-glow / saturation, lower = flatter / more shadow detail.
    # Input names changed between Blender 3.x and 4.x; try both for safety.
    for emission_color_name in ("Emission Color", "Emission"):
        if emission_color_name in n_bsdf.inputs:
            n_bsdf.inputs[emission_color_name].default_value = (
                rgb[0], rgb[1], rgb[2], 1.0,
            )
            break
    if "Emission Strength" in n_bsdf.inputs:
        n_bsdf.inputs["Emission Strength"].default_value = float(emission_strength)

    nt.links.new(n_bsdf.outputs["BSDF"], n_out.inputs["Surface"])

    # CRITICAL: diffuse_color is what Solid-shading mode displays.
    mat.diffuse_color = (rgb[0], rgb[1], rgb[2], 1.0)
    return mat


def _quality_settings(scene) -> tuple[int, int, int]:
    """Return (sphere_rings, sphere_segs, arrow_sides) for the current quality level."""
    q = scene.phonon_vision.render_quality
    if q == "LOW":
        return 8, 12, 8
    if q == "HIGH":
        return 24, 32, 24
    return 12, 16, 16   # MEDIUM (default)


def _create_template_sphere(species: str, radius: float,
                            rgb: tuple[float, float, float],
                            n_rings: int = 12, n_segs: int = 16,
                            emission_strength: float = 0.30) -> bpy.types.Object:
    """Create (or replace) the template UV sphere for a species."""
    name = SPHERE_PREFIX + species
    existing = bpy.data.objects.get(name)
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)
    mesh_name = f"PV.SphereMesh.{species}"
    existing_mesh = bpy.data.meshes.get(mesh_name)
    if existing_mesh is not None:
        bpy.data.meshes.remove(existing_mesh)

    # Build a UV sphere manually (no bpy.ops dependency)
    mesh = bpy.data.meshes.new(mesh_name)
    verts = []
    faces = []
    verts.append((0.0, 0.0, radius))   # top pole
    for ring in range(1, n_rings):
        theta = np.pi * ring / n_rings
        z = radius * np.cos(theta)
        r = radius * np.sin(theta)
        for seg in range(n_segs):
            phi = 2 * np.pi * seg / n_segs
            verts.append((r * np.cos(phi), r * np.sin(phi), z))
    verts.append((0.0, 0.0, -radius))  # bottom pole
    bottom_idx = len(verts) - 1

    for s in range(n_segs):
        nx_seg = (s + 1) % n_segs
        faces.append((0, 1 + s, 1 + nx_seg))
    for ring in range(n_rings - 2):
        ring_start = 1 + ring * n_segs
        next_ring_start = 1 + (ring + 1) * n_segs
        for s in range(n_segs):
            nx_seg = (s + 1) % n_segs
            faces.append((
                ring_start + s, next_ring_start + s,
                next_ring_start + nx_seg, ring_start + nx_seg,
            ))
    last_ring_start = 1 + (n_rings - 2) * n_segs
    for s in range(n_segs):
        nx_seg = (s + 1) % n_segs
        faces.append((bottom_idx, last_ring_start + nx_seg, last_ring_start + s))

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    for poly in mesh.polygons:
        poly.use_smooth = True

    mat = _get_or_build_species_material(species, rgb,
                                         emission_strength=emission_strength)
    obj.data.materials.append(mat)

    coll = bpy.data.collections.get(COLLECTION_NAME)
    coll.objects.link(obj)
    # IMPORTANT: do NOT set hide_render=True here. In DupliVerts the
    # *template* must be renderable for its instances to render — Blender
    # hides the template's own bare appearance via the *parent's*
    # show_instancer_for_render/viewport flags, not the template's hide_*.
    # We only mark the template hidden in the outliner so it doesn't
    # clutter the scene; this doesn't affect instance rendering.
    obj.hide_set(True)            # Outliner-only hide; instances still render
    return obj


# ============================================================
# Per-species atom vertex meshes + DupliVerts
# ============================================================
def create_atom_objects(scene) -> dict[str, bpy.types.Object]:
    """Build a vertex mesh per species, with a child template sphere.

    When ``bond_periodic`` is on, also appends image-atom slots — periodic-
    image copies of supercell atoms that have at least one bond reaching back
    into the real supercell. These show up as visible spheres at the ends of
    PBC stub bonds (rather than bonds dangling into empty space). Image atoms
    inherit the Bloch-phase machinery automatically via their extended
    uc_indices.
    """
    data = get_loaded_data(scene)
    if data is None:
        raise RuntimeError("No phonon data loaded")
    props = scene.phonon_vision

    nx, ny, nz = props.supercell_n[:]
    # Single unified expansion (phonon-core 0.1.4+). The "Show atoms & bonds
    # across PBC" toggle (bond_periodic) selects the boundary mode:
    #   ON  → "halo": strict supercell + iterative image atoms (PBC bonds
    #         terminate on visible spheres). OFF → "strict": supercell only.
    # The old two-step generate_supercell_slots + generate_image_atom_slots is
    # now this one call; the halo path is bit-identical to 0.5.3.
    from phonon_core.expand import expand_cell
    mode = "halo" if bool(props.bond_periodic) else "strict"
    try:
        expanded = expand_cell(data.material, nx, ny, nz, mode=mode,
                               bond_tolerance=float(props.bond_tolerance))
    except Exception as e:
        # Resilience parity with 0.5.3: if image generation ever fails, fall
        # back to the plain supercell rather than aborting the whole build.
        print(f"[phonon-vision] expand_cell(mode={mode!r}) failed: {e}; "
              f"falling back to strict")
        expanded = expand_cell(data.material, nx, ny, nz, mode="strict")
    uc_idx = expanded.n_int
    atom_idx = expanded.j
    base_pos = expanded.r_cart

    n_slots = len(atom_idx)

    material = data.material
    masses_by_sp: dict[str, float] = {}
    for sp, m in zip(material.symbols, material.masses):
        masses_by_sp.setdefault(sp, float(m))
    all_species = material.unique_species

    by_species: dict[str, list[int]] = {sp: [] for sp in all_species}
    for slot, atom_i in enumerate(atom_idx):
        sp = material.symbols[atom_i]
        by_species[sp].append(slot)

    # Clean up old per-species objects
    for sp in all_species:
        for prefix in (ATOMS_PREFIX, SPHERE_PREFIX):
            old = bpy.data.objects.get(prefix + sp)
            if old is not None:
                bpy.data.objects.remove(old, do_unlink=True)

    coll = get_or_create_collection(scene)
    result: dict[str, bpy.types.Object] = {}

    # Build a lookup from props.species_overrides — the user's source of truth
    overrides = {entry.name: entry for entry in scene.phonon_vision.species_overrides}

    # Custom-prop pack of the shared geometry (C-order, pinned in motion.py;
    # unpack_geometry at the animate site inverts it exactly).
    flat_pos, flat_uc = motion.pack_geometry(base_pos, uc_idx)

    n_rings, n_segs, _ = _quality_settings(scene)
    for sp in all_species:
        entry = overrides.get(sp)
        if entry is not None:
            rgb = (float(entry.color[0]), float(entry.color[1]), float(entry.color[2]))
            radius = float(entry.radius)
        else:
            # Fallback if overrides weren't populated (shouldn't happen post-load)
            rgb = _species_color_rgb(sp, masses_by_sp, all_species)
            species_masses = [float(material.masses[i]) for i, s in enumerate(material.symbols) if s == sp]
            radius = _atomic_radius_hint(float(np.mean(species_masses)))
        template = _create_template_sphere(
            sp, radius, rgb, n_rings=n_rings, n_segs=n_segs,
            emission_strength=float(props.species_vibrancy),
        )

        slot_indices = by_species[sp]
        positions = base_pos[slot_indices]

        atoms_name = ATOMS_PREFIX + sp
        old_mesh = bpy.data.meshes.get(f"PV.AtomsMesh.{sp}")
        if old_mesh is not None:
            bpy.data.meshes.remove(old_mesh)
        mesh = bpy.data.meshes.new(f"PV.AtomsMesh.{sp}")
        mesh.from_pydata([tuple(p) for p in positions], [], [])
        mesh.update()
        obj = bpy.data.objects.new(atoms_name, mesh)
        coll.objects.link(obj)

        # DupliVerts: parent the template to this vertex mesh and set instance_type.
        # CRITICAL render fix: hide the parent's BARE vertex cloud from both
        # viewport and render, but its DUPLI INSTANCES will still appear.
        # Setting hide_render on the template breaks instance rendering — the
        # correct flags are these on the parent (the instancer).
        template.parent = obj
        obj.instance_type = "VERTS"
        obj.show_instancer_for_viewport = False
        obj.show_instancer_for_render = False

        # Stash per-species and shared metadata
        obj["slot_indices"] = slot_indices
        obj["base_pos"] = positions.flatten().tolist()
        obj["n_atoms_species"] = len(slot_indices)
        obj["uc_indices"] = flat_uc
        obj["atom_indices"] = atom_idx.tolist()
        obj["n_slots_total"] = int(n_slots)
        obj["all_base_pos"] = flat_pos
        obj["species"] = sp

        result[sp] = obj

    return result


# ============================================================
# Cell outline — three modes, built as real cylinder geometry
# ============================================================
def _cylinder_along(p0: np.ndarray, p1: np.ndarray, radius: float, n_sides: int = 8):
    """Build a closed cylinder between two points.

    Returns (verts, faces) where verts is an (N,3) ndarray and faces is
    a list of vertex-index tuples.
    """
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return np.zeros((0, 3)), []
    axis = direction / length

    # Orthonormal basis perpendicular to axis
    if abs(axis[2]) < 0.9:
        helper = np.array([0.0, 0.0, 1.0])
    else:
        helper = np.array([1.0, 0.0, 0.0])
    perp1 = np.cross(axis, helper)
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)

    angles = np.linspace(0.0, 2 * np.pi, n_sides, endpoint=False)
    ring = (np.cos(angles)[:, None] * perp1[None, :] +
            np.sin(angles)[:, None] * perp2[None, :])

    ring_p0 = p0[None, :] + radius * ring
    ring_p1 = p1[None, :] + radius * ring

    verts = np.vstack([ring_p0, ring_p1])
    faces = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        faces.append((i, j, n_sides + j, n_sides + i))
    # Close the ends (single n-gons)
    faces.append(tuple(range(n_sides)))
    faces.append(tuple(range(2 * n_sides - 1, n_sides - 1, -1)))
    return verts, faces


def create_cell_object(scene) -> bpy.types.Object:
    """Cell outline. Mode set via props.unit_cell_kind:

    - PRIMITIVE: outline only the primitive UC at the origin.
    - GRID:      outline every individual UC in the supercell (internal grid).
    - SUPERCELL: outline only the outer supercell bounding box.

    Built as 12 cylinders per box for reliable rendering in all viewport
    modes. No wireframe modifier dependency.
    """
    data = get_loaded_data(scene)
    if data is None:
        raise RuntimeError("No phonon data loaded")
    props = scene.phonon_vision

    # Clean up previous
    existing = bpy.data.objects.get(CELL_OBJ)
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)
    old_mesh = bpy.data.meshes.get("PV.CellMesh")
    if old_mesh is not None:
        bpy.data.meshes.remove(old_mesh)

    lat = np.asarray(data.material.lattice, dtype=np.float64)
    nx, ny, nz = props.supercell_n[:]
    kind = props.unit_cell_kind

    # Edge thickness: auto-scaled to lattice size, then × user multiplier
    avg_edge = float(np.linalg.norm(lat[0]) + np.linalg.norm(lat[1]) + np.linalg.norm(lat[2])) / 3
    auto_radius = max(0.025, avg_edge * 0.012) * 0.5     # matched to what looked good for SnSe2 at 0.5
    edge_radius = auto_radius * float(props.cell_thickness)

    # Each entry: (start_xyz, end_xyz)
    edge_segments: list[tuple[np.ndarray, np.ndarray]] = []

    def box_edges(origin: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        """Return the 12 edges (as start/end pairs) of a parallelepiped."""
        # 8 corners indexed by (fa, fb, fc) ∈ {0,1}^3
        corners = {}
        for fa in (0, 1):
            for fb in (0, 1):
                for fc in (0, 1):
                    corners[(fa, fb, fc)] = origin + fa * a + fb * b + fc * c
        out = []
        for fb in (0, 1):
            for fc in (0, 1):
                out.append((corners[(0, fb, fc)], corners[(1, fb, fc)]))     # a-direction edges
        for fa in (0, 1):
            for fc in (0, 1):
                out.append((corners[(fa, 0, fc)], corners[(fa, 1, fc)]))     # b-direction edges
        for fa in (0, 1):
            for fb in (0, 1):
                out.append((corners[(fa, fb, 0)], corners[(fa, fb, 1)]))     # c-direction edges
        return out

    if kind == "PRIMITIVE":
        edge_segments.extend(box_edges(np.zeros(3), lat[0], lat[1], lat[2]))
    elif kind == "GRID":
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    origin = ix * lat[0] + iy * lat[1] + iz * lat[2]
                    edge_segments.extend(box_edges(origin, lat[0], lat[1], lat[2]))
    else:  # SUPERCELL
        edge_segments.extend(box_edges(np.zeros(3), nx * lat[0], ny * lat[1], nz * lat[2]))

    # Dedupe identical edges (GRID has many shared edges between neighbours)
    seen = set()
    unique_segments = []
    for p0, p1 in edge_segments:
        key1 = (tuple(np.round(p0, 4)), tuple(np.round(p1, 4)))
        key2 = (tuple(np.round(p1, 4)), tuple(np.round(p0, 4)))
        if key1 in seen or key2 in seen:
            continue
        seen.add(key1)
        unique_segments.append((p0, p1))

    # Build merged mesh of cylinders
    all_verts: list[np.ndarray] = []
    all_faces: list[tuple] = []
    offset = 0
    for p0, p1 in unique_segments:
        verts, faces = _cylinder_along(p0, p1, radius=edge_radius, n_sides=8)
        if len(verts) == 0:
            continue
        all_verts.append(verts)
        for f in faces:
            all_faces.append(tuple(v + offset for v in f))
        offset += len(verts)

    mesh = bpy.data.meshes.new("PV.CellMesh")
    if all_verts:
        merged = np.vstack(all_verts)
        mesh.from_pydata(list(map(tuple, merged)), [], all_faces)
        mesh.update()
        for poly in mesh.polygons:
            poly.use_smooth = True

    obj = bpy.data.objects.new(CELL_OBJ, mesh)
    coll = get_or_create_collection(scene)
    coll.objects.link(obj)

    # Material: emissive black so it's always visible
    mat = bpy.data.materials.get("PV.CellMat")
    if mat is None:
        mat = bpy.data.materials.new("PV.CellMat")
        mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out_n = nt.nodes.new("ShaderNodeOutputMaterial")
    emit_n = nt.nodes.new("ShaderNodeEmission")
    emit_n.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    emit_n.inputs["Strength"].default_value = 1.0
    nt.links.new(emit_n.outputs["Emission"], out_n.inputs["Surface"])
    mat.diffuse_color = (0.0, 0.0, 0.0, 1.0)
    obj.data.materials.append(mat)

    return obj


def refresh_cell(scene) -> None:
    data = get_loaded_data(scene)
    if data is None:
        return
    try:
        create_cell_object(scene)
        props = scene.phonon_vision
        obj = bpy.data.objects.get(CELL_OBJ)
        if obj is not None:
            obj.hide_viewport = not props.show_unit_cell
            obj.hide_render = not props.show_unit_cell
    except Exception as e:
        print(f"[phonon-vision] cell rebuild failed: {e}")


def _apply_arrow_color(mat, rgba) -> None:
    """Write arrow color to BOTH shading paths of ``mat``.

    The ONLY place arrow color is written: the Principled BSDF "Base Color"
    input (Material Preview / Rendered) AND ``mat.diffuse_color`` (Solid). Sets
    input values only — never removes or recreates the material or its nodes.
    ``rgba`` must be a 4-tuple (r, g, b, 1.0).
    """
    if mat.use_nodes and mat.node_tree is not None:
        for n in mat.node_tree.nodes:
            if n.type == 'BSDF_PRINCIPLED':
                n.inputs["Base Color"].default_value = rgba
    mat.diffuse_color = rgba


def _apply_species_color(mat, rgb, emission_strength) -> None:
    """Color-only, in-place write to an existing PV.Mat.<sp> — no node rebuild,
    no object/mesh replacement, so it is callback-safe (including during
    playback). Mirrors the color/emission/diffuse fields that
    _get_or_build_species_material sets, but only assigns values on the existing
    Principled BSDF; it never touches geometry.
    """
    rgba = (rgb[0], rgb[1], rgb[2], 1.0)
    if mat.use_nodes and mat.node_tree is not None:
        for n in mat.node_tree.nodes:
            if n.type == 'BSDF_PRINCIPLED':
                n.inputs["Base Color"].default_value = rgba
                for em in ("Emission Color", "Emission"):
                    if em in n.inputs:
                        n.inputs[em].default_value = rgba
                        break
                if "Emission Strength" in n.inputs:
                    n.inputs["Emission Strength"].default_value = float(emission_strength)
    mat.diffuse_color = rgba


def _invalidate_and_redraw(mats) -> None:
    """Make an in-place material edit actually show in the viewport.

    ``tag_redraw`` alone repaints the *cached* material — the depsgraph must also
    be told the material changed, or Blender redraws the unchanged result. So:
    ``mat.update_tag()`` each edited material (marks it dirty; covers
    diffuse_color/Solid AND Base Color/Material+Rendered), then re-evaluate the
    depsgraph (``view_layer.update()``), then tag the 3D viewports to repaint.

    This is the depsgraph re-evaluation the palette operators get for free by
    running in ``execute()`` context. Property-update callbacks must trigger it
    explicitly AND deferred (Blender can defer/ignore a depsgraph update issued
    mid-callback) — see state._schedule_redraw / _drain_redraws.
    """
    for mat in mats:
        if mat is not None:
            mat.update_tag()
    bpy.context.view_layer.update()
    wm = bpy.context.window_manager
    for window in (wm.windows if wm else []):
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def redraw_kick() -> None:
    """Deferred-only redraw: re-evaluate the depsgraph and repaint the 3D
    viewports. Materials are already update_tag'd by the synchronous writers,
    so this only needs the depsgraph kick + repaint (the part that is ignored
    mid-callback and must run deferred / on a frame change). NO material write,
    NO object replacement — safe to drain from the frame-change handler.
    """
    bpy.context.view_layer.update()
    wm = bpy.context.window_manager
    for window in (wm.windows if wm else []):
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def write_arrow_color_sync(scene) -> None:
    """SYNCHRONOUS arrow-color write — callback-safe, works during playback.

    Writes PV.ArrowMat (Base Color + diffuse_color) and marks it dirty
    (update_tag) immediately, so the material data is correct even mid-playback
    when bpy.app.timers are starved. The actual repaint (view_layer.update +
    tag_redraw) is the deferred redraw_kick (static) or the per-frame depsgraph
    eval (playback).
    """
    mat = bpy.data.materials.get("PV.ArrowMat")
    if mat is None:
        return  # arrows not built yet; initial build reads arrow_color
    ac = tuple(scene.phonon_vision.arrow_color)
    rgba = (ac[0], ac[1], ac[2], 1.0)
    _apply_arrow_color(mat, rgba)
    mat.update_tag()


def apply_species_colors_sync(scene) -> None:
    """SYNCHRONOUS per-species color/vibrancy write — callback-safe, works during
    playback. Writes each PV.Mat.<sp> color + emission in place and marks it
    dirty. Geometry (radius / DupliVerts mesh) is NOT touched here — that stays
    on the deferred/operator path (refresh_species_visuals).
    """
    vib = float(scene.phonon_vision.species_vibrancy)
    for entry in scene.phonon_vision.species_overrides:
        sp = entry.name
        rgb = (float(entry.color[0]), float(entry.color[1]), float(entry.color[2]))
        mat = bpy.data.materials.get(f"PV.Mat.{sp}")
        if mat is None:
            continue
        _apply_species_color(mat, rgb, vib)
        mat.update_tag()


def sync_appearance(scene) -> None:
    """Write all current appearance props to the existing materials, in place.

    Material writes ONLY — PV.ArrowMat and each PV.Mat.<sp> Base Color +
    Emission + diffuse_color from current props, plus update_tag. No object
    replacement, no node rebuild, so it is cheap enough to call every frame.

    Called from the frame-change handler (frame_change_pre) so appearance edits
    surface during playback: the per-frame handler is the only thing guaranteed
    to run every frame, and writing here means the depsgraph eval for that frame
    sees the dirtied materials. Harmless when nothing changed (idempotent
    re-write of the same values).
    """
    write_arrow_color_sync(scene)
    apply_species_colors_sync(scene)


def refresh_arrow_color(scene) -> None:
    """Apply arrow_color to PV.ArrowMat in place, invalidate it, and redraw.

    Appearance only — calls _apply_arrow_color (no datablock removal/recreation)
    so it is safe live, including mid-animation (the per-frame handler only
    rewrites vertex coordinates, never the material).
    """
    mat = bpy.data.materials.get("PV.ArrowMat")
    if mat is None:
        return  # arrows not built yet; initial build will read the color
    ac = tuple(scene.phonon_vision.arrow_color)
    rgba = (ac[0], ac[1], ac[2], 1.0)
    _apply_arrow_color(mat, rgba)
    _invalidate_and_redraw([mat])


# ============================================================
# Arrows
# ============================================================
def _arrow_vertices(start: np.ndarray, vec: np.ndarray, radius: float,
                    head_length_frac: float, head_radius_mult: float,
                    n_sides: int) -> tuple[np.ndarray, list]:
    length = float(np.linalg.norm(vec))
    if length < 1e-9:
        return np.zeros((0, 3)), []
    direction = vec / length
    if abs(direction[2]) < 0.9:
        helper = np.array([0.0, 0.0, 1.0])
    else:
        helper = np.array([1.0, 0.0, 0.0])
    perp1 = np.cross(direction, helper)
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)

    head_len = length * head_length_frac
    shaft_len = length - head_len
    head_r = radius * head_radius_mult

    angles = np.linspace(0.0, 2 * np.pi, n_sides, endpoint=False)
    ring_unit = (np.cos(angles)[:, None] * perp1[None, :] +
                 np.sin(angles)[:, None] * perp2[None, :])

    base = start[None, :] + radius * ring_unit
    top_shaft = start[None, :] + direction * shaft_len + radius * ring_unit
    base_head = start[None, :] + direction * shaft_len + head_r * ring_unit
    tip = (start + direction * length)[None, :]

    verts = np.vstack([base, top_shaft, base_head, tip])
    faces = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        faces.append([i, j, n_sides + j, n_sides + i])
    for i in range(n_sides):
        j = (i + 1) % n_sides
        faces.append([n_sides + i, n_sides + j, 2 * n_sides + j, 2 * n_sides + i])
    tip_idx = 3 * n_sides
    for i in range(n_sides):
        j = (i + 1) % n_sides
        faces.append([2 * n_sides + i, 2 * n_sides + j, tip_idx])
    return verts, faces


def create_arrow_object(scene) -> bpy.types.Object:
    """Allocate the arrows mesh with fixed topology for ``n_slots`` atoms.

    Each atom gets ``3 * n_sides + 1`` vertex slots and ``3 * n_sides``
    faces, dummy positions. Per-frame updates only rewrite vertex
    coordinates via foreach_set — roughly 10× faster than rebuilding the
    mesh from_pydata each frame (same trick as bonds).

    Invisible arrows (|displacement| < 0.02 Å) get their verts collapsed
    to the atom's position so the degenerate faces disappear visually.
    """
    existing = bpy.data.objects.get(ARROWS_OBJ)
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)
    old_mesh = bpy.data.meshes.get("PV.ArrowsMesh")
    if old_mesh is not None:
        bpy.data.meshes.remove(old_mesh)

    mesh = bpy.data.meshes.new(name="PV.ArrowsMesh")
    obj = bpy.data.objects.new(ARROWS_OBJ, mesh)
    coll = get_or_create_collection(scene)
    coll.objects.link(obj)

    # Allocate topology for all atom slots in the current supercell.
    # Read n_slots from any species object's stash (they all carry it).
    species_objs = _find_species_objects()
    if species_objs:
        first = next(iter(species_objs.values()))
        n_slots = int(first.get("n_slots_total", 0))
    else:
        n_slots = 0

    _, _, n_sides = _quality_settings(scene)

    if n_slots > 0:
        verts_per_arrow = 3 * n_sides + 1
        n_verts_total = n_slots * verts_per_arrow
        # Dummy positions; real positions written per frame
        dummy_verts = [(0.0, 0.0, 0.0)] * n_verts_total
        faces = _make_arrow_face_indices(n_slots, n_sides)
        mesh.from_pydata(dummy_verts, [], faces)
        mesh.update()
        for poly in mesh.polygons:
            poly.use_smooth = True

    props = scene.phonon_vision
    ac = tuple(props.arrow_color)        # default (0.90, 0.10, 0.10) == old red
    arrow_rgba = (ac[0], ac[1], ac[2], 1.0)

    mat = bpy.data.materials.get("PV.ArrowMat")
    if mat is None:
        mat = bpy.data.materials.new("PV.ArrowMat")
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out_n = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf_n = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf_n.inputs["Roughness"].default_value = 0.55
        nt.links.new(bsdf_n.outputs["BSDF"], out_n.inputs["Surface"])
    # Color on EVERY build, not only at creation: otherwise a rebuild leaves
    # the BSDF Base Color (Material Preview / Rendered) stuck at the first
    # color set, while only diffuse_color (Solid) updates. Single source of
    # truth for arrow color.
    _apply_arrow_color(mat, arrow_rgba)
    obj.data.materials.append(mat)
    return obj


# ============================================================
# Bonds — static topology, dynamic per-frame endpoint update
# ============================================================
# The bond CRITERION (which pairs bond, strict + PBC-image paths, side-car
# per-pair cutoffs with replace semantics) lives in the bpy-free
# bonds.build_bond_list — moved there at 0.7.9 so plain-python tests can
# drive it. This module keeps only geometry: topology allocation and the
# per-frame vertex updates, batched per canonical species pair.


def _compute_bond_verts_vectorized(p0s: np.ndarray, p1s: np.ndarray,
                                    radius: float, n_sides: int) -> np.ndarray:
    """Compute cylinder verts for many bonds at once.

    Parameters
    ----------
    p0s, p1s : (M, 3) arrays of start/end positions in Å.
    radius : cylinder radius in Å.
    n_sides : number of facets around each cylinder.

    Returns
    -------
    verts : (M * 2 * n_sides, 3) array of vertex positions, with the layout
        ``[bond0 ring_p0, bond0 ring_p1, bond1 ring_p0, bond1 ring_p1, ...]``.
        This is the layout `foreach_set('co', ...)` expects when paired with
        face indices built by ``_make_cylinder_face_indices``.
    """
    M = len(p0s)
    if M == 0:
        return np.zeros((0, 3), dtype=np.float64)

    directions = p1s - p0s
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    safe_len = np.maximum(lengths, 1e-12)
    axes = directions / safe_len     # (M, 3)

    # Helper vector for cross product — chosen per-axis to avoid degenerate cross
    helpers = np.where(np.abs(axes[:, 2:3]) < 0.9,
                       np.array([0.0, 0.0, 1.0]),
                       np.array([1.0, 0.0, 0.0]))   # (M, 3)
    perp1 = np.cross(axes, helpers)
    # Also clamp the perp normalisation — for degenerate zero-length bonds
    # (which we deliberately produce when collapsing a broken bond to a
    # single midpoint), axes ≈ 0 → perp1 ≈ 0 → naive normalize is NaN.
    # Clamped to 1e-12 we get perp1 ≈ 0 instead, which gives ring_dir = 0
    # and all bond verts land at the (degenerate) midpoint — correct.
    perp1_norm = np.linalg.norm(perp1, axis=1, keepdims=True)
    perp1 = perp1 / np.maximum(perp1_norm, 1e-12)
    perp2 = np.cross(axes, perp1)    # (M, 3)

    angles = np.linspace(0.0, 2 * np.pi, n_sides, endpoint=False)
    cos_a = np.cos(angles)                                # (n_sides,)
    sin_a = np.sin(angles)
    # rings: (M, n_sides, 3) = radius * (cos*perp1 + sin*perp2)
    ring_dir = (cos_a[None, :, None] * perp1[:, None, :] +
                sin_a[None, :, None] * perp2[:, None, :])
    rings_p0 = p0s[:, None, :] + radius * ring_dir       # (M, n_sides, 3)
    rings_p1 = p1s[:, None, :] + radius * ring_dir

    # Layout: for each bond, n_sides verts at p0 then n_sides at p1
    out = np.empty((M, 2 * n_sides, 3), dtype=np.float64)
    out[:, :n_sides, :] = rings_p0
    out[:, n_sides:, :] = rings_p1
    return out.reshape(M * 2 * n_sides, 3)


def _make_cylinder_face_indices(n_bonds: int, n_sides: int,
                                 first_vert: int = 0) -> list[tuple]:
    """Generate face indices for ``n_bonds`` cylinders, starting at
    ``first_vert`` in the combined mesh."""
    faces: list[tuple] = []
    verts_per_cyl = 2 * n_sides
    for b in range(n_bonds):
        base = first_vert + b * verts_per_cyl
        # Side quads
        for i in range(n_sides):
            j = (i + 1) % n_sides
            faces.append((base + i, base + j, base + n_sides + j, base + n_sides + i))
        # End caps
        faces.append(tuple(base + k for k in range(n_sides)))
        faces.append(tuple(base + k for k in range(2 * n_sides - 1, n_sides - 1, -1)))
    return faces


def _get_or_build_bond_color_material(pair_key: str,
                                      rgb: tuple[float, float, float]
                                      ) -> bpy.types.Material:
    """Solid curated-color bond material for one pair (navigator custom mode).

    ``rgb`` is LINEAR (``bonds.hex_to_linear`` of the side-car display-sRGB
    hex). Writes BOTH shading paths — the Principled ``Base Color`` (Material
    Preview / Rendered) and ``diffuse_color`` (Solid) — the same dual-path
    discipline as the species and grey bond materials. Reused by name across
    rebuilds, so re-importing a different hex rewrites the nodes in place."""
    name = f"PV.BondColor.{pair_key}"
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    n_out = nt.nodes.new("ShaderNodeOutputMaterial")
    n_bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    n_bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    n_bsdf.inputs["Roughness"].default_value = 0.5
    nt.links.new(n_bsdf.outputs["BSDF"], n_out.inputs["Surface"])
    mat.diffuse_color = (rgb[0], rgb[1], rgb[2], 1.0)
    return mat


def create_bond_object(scene) -> bpy.types.Object:
    """Create the Bonds object, including the static bond list, material slots,
    AND pre-allocated mesh topology.

    The mesh is built ONCE with all the vertex slots and face indices needed
    for the bond list. Per-frame updates only rewrite the vertex coordinates
    via ``foreach_set('co', ...)``, which is roughly 10× faster than
    rebuilding the topology each frame with ``mesh.from_pydata``.

    Material slot layout when ``bond_two_color`` is on:
      - One slot per chemical species in the material, using the same
        ``PV.Mat.<species>`` materials as atom spheres so colors stay in sync.
      - Then a slot for the fallback grey ``PV.BondMat``.
      - Then (0.7.12) one ``PV.BondColor.<pair>`` slot per curated pair that
        carries a side-car ``color`` — a solid cylinder in that hex that
        overrides the species split for the pair (navigator custom mode).
    """
    data = get_loaded_data(scene)
    if data is None:
        raise RuntimeError("No phonon data loaded")
    props = scene.phonon_vision

    existing = bpy.data.objects.get(BONDS_OBJ)
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)
    old_mesh = bpy.data.meshes.get("PV.BondsMesh")
    if old_mesh is not None:
        bpy.data.meshes.remove(old_mesh)

    species_objs = _find_species_objects()
    if not species_objs:
        mesh = bpy.data.meshes.new("PV.BondsMesh")
        obj = bpy.data.objects.new(BONDS_OBJ, mesh)
        get_or_create_collection(scene).objects.link(obj)
        return obj

    first = next(iter(species_objs.values()))
    n_slots = int(first["n_slots_total"])
    atom_idx = np.asarray(first["atom_indices"], dtype=np.int64)
    all_base_pos, _ = motion.unpack_geometry(
        first["all_base_pos"], first["uc_indices"], n_slots)
    symbols_per_slot = [data.material.symbols[int(ai)] for ai in atom_idx]

    nx, ny, nz = props.supercell_n[:]
    lat = np.asarray(data.material.lattice, dtype=np.float64)
    # NOTE: include_images=False because if bond_periodic is on, the image
    # atoms are already real slots in the supercell — internal bonds catch
    # everything we need. If bond_periodic is off, we just don't get PBC
    # bonds at all.
    two_color = bool(props.bond_two_color)
    _, _, n_sides = _quality_settings(scene)

    # Curated side-car spec (replace semantics) or None -> default criterion.
    # The criterion compares in Å: all_base_pos comes from expand_cell in Å
    # and 1 Blender unit ≡ 1 Å throughout, so absolute side-car cutoffs
    # apply verbatim.
    from phonon_core.radii import covalent_radius
    spec = bondcore.spec_from_json(props.bonds_spec_json)
    pairs = bondcore.build_bond_list(
        all_base_pos, symbols_per_slot, float(props.bond_tolerance),
        lattice=lat, supercell=(nx, ny, nz),
        include_images=False,
        spec=spec, radius_fn=covalent_radius,
    )

    # Sort bonds by canonical species pair (stable) so every downstream
    # structure — topology, material slots, per-frame vertex batches — is
    # grouped per pair: one contiguous batch per pair, each with its own
    # cylinder radius (spec radius_A / radius_scale), and ready for a later
    # version's per-pair materials.
    pair_key = [bondcore.canonicalize_pair(symbols_per_slot[p[0]],
                                           symbols_per_slot[p[1]])
                for p in pairs]
    order = sorted(range(len(pairs)), key=lambda k: pair_key[k])
    pairs = [pairs[k] for k in order]
    pair_key = [pair_key[k] for k in order]
    scales = bondcore.pair_scales(spec)
    radii_A = bondcore.pair_radii_A(spec)
    colors = bondcore.pair_colors(spec)

    mesh = bpy.data.meshes.new("PV.BondsMesh")
    obj = bpy.data.objects.new(BONDS_OBJ, mesh)
    coll = get_or_create_collection(scene)
    coll.objects.link(obj)
    obj["pairs_i"] = [int(p[0]) for p in pairs]
    obj["pairs_j"] = [int(p[1]) for p in pairs]
    # image offsets stored flat as 3N floats (x,y,z per pair)
    obj["pair_image_offsets"] = [float(c) for p in pairs for c in p[2]]
    obj["n_pairs"] = len(pairs)
    obj["symbols_per_slot_json"] = json.dumps(symbols_per_slot)
    # Per-bond canonical pair key + display radius (drives the per-pair
    # vertex batching in the frame update). radius_A (0.7.10) is the
    # AUTHORITATIVE absolute Å cylinder radius from the side-car; -1.0 marks
    # "absent" (parse clamps real values to >= 0.02, so the sentinel can
    # never collide). radius_scale is the legacy relative fallback.
    obj["pair_keys_json"] = json.dumps(pair_key)
    obj["pair_radius_scale"] = [float(scales.get(k, 1.0)) for k in pair_key]
    obj["pair_radius_A"] = [float(radii_A.get(k, -1.0)) for k in pair_key]

    # Equilibrium per-bond Cartesian distance, used per-frame to decide
    # whether the bond should be visible (stretch/compress thresholds).
    # Computed from base_pos here at build time so the per-frame path
    # only needs to compute current distances and compare.
    pair_eq_dist: list[float] = []
    for (i, j, off) in pairs:
        d = float(np.linalg.norm(
            all_base_pos[j] + np.asarray(off, dtype=np.float64)
            - all_base_pos[i]
        ))
        pair_eq_dist.append(d)
    obj["pair_eq_dist"] = pair_eq_dist

    # Remember the topology parameters so per-frame update can detect a
    # need to rebuild (toggle two_color, quality change)
    obj["topology_two_color"] = int(two_color)
    obj["topology_n_sides"] = int(n_sides)

    # Material slots: per-species first, then grey fallback last
    all_species = list(data.material.unique_species)
    species_slot: dict[str, int] = {}
    for idx, sp in enumerate(all_species):
        sp_mat = bpy.data.materials.get(f"PV.Mat.{sp}")
        if sp_mat is None:
            sp_mat = bpy.data.materials.new(f"PV.Mat.{sp}")
            sp_mat.use_nodes = True
            sp_mat.diffuse_color = (0.5, 0.5, 0.5, 1.0)
        obj.data.materials.append(sp_mat)
        species_slot[sp] = idx

    grey = bpy.data.materials.get("PV.BondMat")
    if grey is None:
        grey = bpy.data.materials.new("PV.BondMat")
        grey.use_nodes = True
        nt = grey.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        n_out = nt.nodes.new("ShaderNodeOutputMaterial")
        n_bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        n_bsdf.inputs["Base Color"].default_value = (0.55, 0.55, 0.55, 1.0)
        n_bsdf.inputs["Roughness"].default_value = 0.5
        nt.links.new(n_bsdf.outputs["BSDF"], n_out.inputs["Surface"])
    grey.diffuse_color = (0.55, 0.55, 0.55, 1.0)
    obj.data.materials.append(grey)
    obj["grey_slot"] = len(all_species)
    obj["species_to_slot_json"] = json.dumps(species_slot)

    # Curated per-pair color materials (navigator custom-color mode, 0.7.12):
    # each curated pair gets ONE solid material — the side-car hex, sRGB->linear
    # via bonds.hex_to_linear — appended as its own slot AFTER the grey
    # fallback (so grey stays at len(all_species)). A bond whose pair carries a
    # color is drawn as a single full cylinder pointed at this slot
    # (bond_material_plan -> "custom"), overriding the per-species split.
    color_slot: dict[str, int] = {}
    for key, hexstr in colors.items():
        cmat = _get_or_build_bond_color_material(key, bondcore.hex_to_linear(hexstr))
        obj.data.materials.append(cmat)
        color_slot[key] = len(obj.data.materials) - 1

    # --- Pre-allocate topology ---
    if not pairs:
        return obj

    n_pairs = len(pairs)
    grey_slot = len(all_species)

    # Per-bond cylinder count and material slots, decided by the bpy-free
    # bonds.bond_material_plan (pinned in tools/test_bonds.py):
    #   - bond_n_cyl[k]   : 1 or 2 cylinders for bond k
    #   - bond_slots[k]   : list of material slot indices (one per cylinder)
    # A curated per-pair COLOR wins (0.7.12): the whole cylinder is one solid
    # side-car color, overriding the split for that pair — the navigator's
    # custom-color branch. Otherwise: same-species bonds collapse to ONE
    # species cylinder (no midpoint seam at homospecific bonds like Br-Br),
    # different-species bonds keep the two-half VESTA-style topology, and with
    # two_color OFF globally every non-curated bond is a single grey cylinder.
    bond_n_cyl: list[int] = []
    bond_slots: list[list[int]] = []
    for k, (i, j, _off) in enumerate(pairs):
        sp_i = symbols_per_slot[i] if i < len(symbols_per_slot) else None
        sp_j = symbols_per_slot[j] if j < len(symbols_per_slot) else None
        slot_i = species_slot.get(sp_i, grey_slot) if sp_i else grey_slot
        slot_j = species_slot.get(sp_j, grey_slot) if sp_j else grey_slot
        n_cyl, style = bondcore.bond_material_plan(
            colors.get(pair_key[k]), two_color, sp_i == sp_j)
        if style == "custom":
            bond_slots.append([color_slot[pair_key[k]]])   # solid curated color
        elif style == "grey":
            bond_slots.append([grey_slot])
        elif style == "species":
            bond_slots.append([slot_i])
        else:   # "split" — two per-species half-cylinders
            bond_slots.append([slot_i, slot_j])
        bond_n_cyl.append(n_cyl)

    n_total_cyl = sum(bond_n_cyl)
    faces = _make_cylinder_face_indices(n_total_cyl, n_sides, first_vert=0)
    faces_per_cyl = n_sides + 2
    mat_slots: list[int] = []
    for slots in bond_slots:
        for s in slots:
            mat_slots.extend([s] * faces_per_cyl)

    # Stash the per-bond cylinder count so the per-frame update knows the
    # layout. We could re-derive it from symbols + two_color, but storing
    # it makes the per-frame path simpler and detached from settings drift.
    obj["bond_n_cyl"] = bond_n_cyl
    obj["n_total_cyl"] = int(n_total_cyl)

    n_verts = n_total_cyl * 2 * n_sides
    # Allocate dummy vertices — real coords come in the per-frame update
    dummy_verts = [(0.0, 0.0, 0.0)] * n_verts
    mesh.from_pydata(dummy_verts, [], faces)
    mesh.update()
    for poly_idx, slot in enumerate(mat_slots):
        if poly_idx < len(mesh.polygons):
            mesh.polygons[poly_idx].material_index = slot
    for poly in mesh.polygons:
        poly.use_smooth = True

    return obj


def _rebuild_bonds_for_positions(scene, current_pos: np.ndarray) -> None:
    """Update bond cylinder vertex positions for the current frame.

    Uses ``foreach_set('co', ...)`` for a fast O(1) per-frame mesh update
    (no topology rebuild). Topology is rebuilt only when the user toggles
    ``bond_two_color`` or changes mesh quality.

    Bond layout (v0.4.0+):
      - Same-species bonds → 1 full-length cylinder (no midpoint seam)
      - Different-species bonds in two_color mode → 2 half cylinders
      - All bonds in two_color=off mode → 1 grey cylinder

    Per-bond cylinder count is stored as ``bond_n_cyl`` on the object;
    we build a flat list of (p0, p1) cylinder endpoints in the same order
    the topology was built, then run one vectorized vertex computation.
    """
    obj = bpy.data.objects.get(BONDS_OBJ)
    if obj is None:
        return
    props = scene.phonon_vision
    if not props.show_bonds:
        obj.hide_viewport = True
        obj.hide_render = True
        return
    obj.hide_viewport = False
    obj.hide_render = False

    pairs_i = list(obj.get("pairs_i", []))
    pairs_j = list(obj.get("pairs_j", []))
    if not pairs_i:
        return  # No bonds; mesh already empty

    # Check if topology matches current settings; rebuild if not.
    _, _, n_sides = _quality_settings(scene)
    two_color = bool(props.bond_two_color)
    topology_two_color = bool(obj.get("topology_two_color", 1))
    topology_n_sides = int(obj.get("topology_n_sides", n_sides))
    if topology_two_color != two_color or topology_n_sides != n_sides:
        # Topology changed — rebuild fully (this also rewrites bond_n_cyl)
        create_bond_object(scene)
        obj = bpy.data.objects.get(BONDS_OBJ)
        if obj is None:
            return

    # Read image offsets (flat list 3N)
    raw_offsets = list(obj.get("pair_image_offsets", []))
    n_pairs = len(pairs_i)
    if len(raw_offsets) == 3 * n_pairs:
        offsets = np.asarray(raw_offsets, dtype=np.float64).reshape(-1, 3)
    else:
        offsets = np.zeros((n_pairs, 3))

    pi = np.asarray(pairs_i, dtype=np.int64)
    pj = np.asarray(pairs_j, dtype=np.int64)
    p0_pair = current_pos[pi]                # (n_pairs, 3) Cartesian of atom i
    p1_pair = current_pos[pj] + offsets      # (n_pairs, 3) Cartesian of atom j (incl. PBC offset)

    # bond_thickness governs ONLY bonds without a curated radius_A — the
    # radius_A branch in bondcore.resolve_radius bypasses it entirely.
    base_radius = float(props.bond_thickness) * 0.06

    # Per-bond canonical pair key + radius data (0.7.9 per-pair batching;
    # 0.7.10 adds radius_A). Bonds are stored sorted by pair, so equal keys
    # are contiguous. Older scene builds without the keys degrade to one
    # all-bonds batch at scale 1.0 / no radius_A (-1.0 sentinel = absent).
    try:
        pair_keys = json.loads(obj.get("pair_keys_json", "[]"))
    except Exception:
        pair_keys = []
    if len(pair_keys) != n_pairs:
        pair_keys = [""] * n_pairs
    raw_scales = list(obj.get("pair_radius_scale", []))
    if len(raw_scales) != n_pairs:
        raw_scales = [1.0] * n_pairs
    raw_rad_A = list(obj.get("pair_radius_A", []))
    if len(raw_rad_A) != n_pairs:
        raw_rad_A = [-1.0] * n_pairs

    # Per-bond cylinder count from topology
    bond_n_cyl = list(obj.get("bond_n_cyl", [2] * n_pairs))  # default to old 2-per-bond if missing

    # Stretch / compress break checks: hide a bond when its current length
    # is outside [min_factor, max_factor] × equilibrium length. Collapsing
    # the bond's cylinder endpoints to a single midpoint makes its mesh
    # vertices degenerate → renders invisible without changing topology.
    eq_dist_list = list(obj.get("pair_eq_dist", []))
    cur_dist = np.linalg.norm(p1_pair - p0_pair, axis=1)
    if len(eq_dist_list) == n_pairs:
        eq_dist = np.asarray(eq_dist_list, dtype=np.float64)
        max_f = float(props.bond_break_max_factor)
        min_f = float(props.bond_break_min_factor)
        too_long = cur_dist > eq_dist * max_f
        too_short = cur_dist < eq_dist * min_f
        broken = too_long | too_short
    else:
        # No equilibrium distances stashed (very old build) — never break
        broken = np.zeros(n_pairs, dtype=bool)

    # For broken bonds, collapse both endpoints to a single midpoint so
    # the cylinder vertices degenerate to a point (invisible).
    if broken.any():
        mid = 0.5 * (p0_pair[broken] + p1_pair[broken])
        p0_pair = p0_pair.copy()
        p1_pair = p1_pair.copy()
        p0_pair[broken] = mid
        p1_pair[broken] = mid

    # Build per-cylinder endpoint arrays. For 1-cyl bonds: (p0, p1). For 2-cyl
    # bonds: (p0, mid) then (mid, p1). Order matches the topology built in
    # create_bond_object — cylinders for bond k come right after bond k-1.
    cyl_p0_list: list[np.ndarray] = []
    cyl_p1_list: list[np.ndarray] = []
    cyl_key_list: list[str] = []      # per-cylinder pair key (batch grouping)
    cyl_scale_list: list[float] = []  # per-cylinder legacy radius scale
    cyl_rad_A_list: list[float] = []  # per-cylinder absolute Å radius (-1 = absent)
    for k in range(n_pairs):
        n_cyl_k = bond_n_cyl[k]
        if n_cyl_k == 1:
            cyl_p0_list.append(p0_pair[k])
            cyl_p1_list.append(p1_pair[k])
        else:  # 2
            mid = 0.5 * (p0_pair[k] + p1_pair[k])
            cyl_p0_list.append(p0_pair[k])
            cyl_p1_list.append(mid)
            cyl_p0_list.append(mid)
            cyl_p1_list.append(p1_pair[k])
        cyl_key_list.extend([pair_keys[k]] * n_cyl_k)
        cyl_scale_list.extend([float(raw_scales[k])] * n_cyl_k)
        cyl_rad_A_list.extend([float(raw_rad_A[k])] * n_cyl_k)

    cyl_p0 = np.asarray(cyl_p0_list, dtype=np.float64)
    cyl_p1 = np.asarray(cyl_p1_list, dtype=np.float64)

    # One vectorized vertex computation PER canonical-pair batch: contiguous
    # run of equal pair key -> one call, radius resolved per pair by
    # bondcore.resolve_radius (navigator 0.6.76 precedence: absolute
    # radius_A wins and bypasses bond_thickness; else base × radius_scale;
    # else base). Uniform within the batch, so the vectorized kernel is
    # unchanged — no per-bond loops. Without a spec every scale is 1.0 with
    # no radius_A (and a pre-0.7.9 object is a single run), reproducing the
    # old single call exactly.
    n_cyl_total = len(cyl_key_list)
    vert_chunks: list[np.ndarray] = []
    start = 0
    for end in range(1, n_cyl_total + 1):
        if end == n_cyl_total or cyl_key_list[end] != cyl_key_list[start]:
            r_A = cyl_rad_A_list[start]
            vert_chunks.append(_compute_bond_verts_vectorized(
                cyl_p0[start:end], cyl_p1[start:end],
                bondcore.resolve_radius(r_A if r_A > 0.0 else None,
                                        cyl_scale_list[start], base_radius),
                n_sides))
            start = end
    if vert_chunks:
        all_verts = np.concatenate(vert_chunks, axis=0)
    else:
        all_verts = np.zeros((0, 3), dtype=np.float64)

    mesh = obj.data
    if len(mesh.vertices) != all_verts.shape[0]:
        # Mesh allocation mismatch — fall back to full rebuild (shouldn't happen)
        create_bond_object(scene)
        obj = bpy.data.objects.get(BONDS_OBJ)
        if obj is None:
            return
        mesh = obj.data
        if len(mesh.vertices) != all_verts.shape[0]:
            return  # Give up — out of sync

    # Fast flat write
    flat = all_verts.ravel().astype(np.float32, copy=False)
    mesh.vertices.foreach_set("co", flat)
    mesh.update()


# ============================================================
# Per-frame update — atoms + arrows + bonds
# ============================================================
def update_atom_positions(scene) -> None:
    """Recompute atom positions AND arrows for the current frame.

    Arrows track the *current frame's* displacement vector — same magnitude
    and direction as the actual motion at this instant. This matches
    phononwebsite's convention and is physically truthful: for real-eigenvector
    modes the arrows oscillate +/-, for complex eigenvectors they trace an
    ellipse around the atom.
    """
    data = get_loaded_data(scene)
    if data is None:
        return

    props = scene.phonon_vision
    species_objs = _find_species_objects()
    if not species_objs:
        return

    # Shared metadata (0.7.11: the unpack, engine call chain, clamp and
    # scatter live in bpy-free motion.py — conformance-tested in
    # tools/test_motion.py and end-to-end by tools/conformance_headless.py.
    # This function's remaining bpy work: read custom props, call motion.*,
    # foreach_set.)
    first = next(iter(species_objs.values()))
    n_slots = int(first["n_slots_total"])
    all_base_pos, uc_idx = motion.unpack_geometry(
        first["all_base_pos"], first["uc_indices"], n_slots)
    atom_idx = np.asarray(first["atom_indices"], dtype=np.int64)

    try:
        mode = data.get_mode(int(props.q_index), int(props.band_index))
    except Exception:
        return

    # Engine call chain + optional clamp (NON-PHYSICAL display aid — see
    # motion.py). Exposed as the displacement because the arrow builder
    # consumes it; current_pos = base + disp exactly as 0.7.10.
    disp = motion.displacements_at_frame(
        mode, data.material, uc_idx,
        frame=scene.frame_current,
        frames_per_cycle=int(props.frames_per_cycle),
        frame_start=int(scene.frame_start),
        amplitude=float(props.amplitude),
        atom_indices=atom_idx,
        use_canonical_phase=bool(props.use_canonical_phase),
        clamp=bool(props.clamp_amplitude),
        cap=float(props.max_displacement),
    )
    current_pos = all_base_pos + disp

    # Update species vertex meshes
    for sp, obj in species_objs.items():
        slot_indices = list(obj["slot_indices"])
        obj.data.vertices.foreach_set(
            "co", motion.species_layout(current_pos, slot_indices))
        obj.data.update()

    # Arrows: current frame's displacement direction & magnitude
    if props.show_arrows:
        _rebuild_arrows(scene, data, atom_idx, disp, current_pos)
    else:
        _hide_arrows()

    # Bonds: cylinders between bonded pairs (bond list is static, endpoints dynamic)
    _rebuild_bonds_for_positions(scene, current_pos)


def _make_arrow_face_indices(n_atoms: int, n_sides: int) -> list[tuple]:
    """Generate face indices for ``n_atoms`` arrows in a merged mesh.

    Each arrow's layout (matching ``_compute_arrow_verts_vectorized``):
      verts[base + 0 : base + n_sides]        — base ring at start, radius=shaft_radius
      verts[base + n_sides : base + 2*n_sides] — top shaft ring, radius=shaft_radius
      verts[base + 2*n_sides : base + 3*n_sides] — cone base ring, radius=head_radius
      verts[base + 3*n_sides]                 — single tip vertex

    Total per arrow: 3*n_sides + 1 verts, 3*n_sides faces (shaft quads,
    annular shaft→cone-base quads, cone triangles).
    """
    faces: list[tuple] = []
    verts_per_arrow = 3 * n_sides + 1
    for a in range(n_atoms):
        base = a * verts_per_arrow
        # Shaft sides
        for i in range(n_sides):
            j = (i + 1) % n_sides
            faces.append((base + i, base + j,
                           base + n_sides + j, base + n_sides + i))
        # Annular ring at shaft→cone-base transition (different radii)
        for i in range(n_sides):
            j = (i + 1) % n_sides
            faces.append((base + n_sides + i, base + n_sides + j,
                           base + 2 * n_sides + j, base + 2 * n_sides + i))
        # Cone triangles to the tip
        tip_idx = base + 3 * n_sides
        for i in range(n_sides):
            j = (i + 1) % n_sides
            faces.append((base + 2 * n_sides + i,
                           base + 2 * n_sides + j, tip_idx))
    return faces


def _compute_arrow_verts_vectorized(
    starts: np.ndarray,      # (n, 3) atom positions
    vecs: np.ndarray,        # (n, 3) arrow vectors (disp * arrow_scale)
    radius: float,           # shaft radius (Å)
    head_length_frac: float, # fraction of arrow length used as cone
    head_radius_mult: float, # cone-base radius / shaft radius
    n_sides: int,
    visible_threshold: float = 0.02,
) -> np.ndarray:
    """Compute vertex positions for n arrows in one vectorized pass.

    Returns a (n * (3*n_sides + 1), 3) array suitable for foreach_set('co').
    Arrows with ``|vec| < visible_threshold`` have all their vertices
    collapsed to the start point — the faces become degenerate and render
    as a single invisible point.
    """
    n = len(starts)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    lens = np.linalg.norm(vecs, axis=1)               # (n,)
    safe = np.maximum(lens, 1e-12)
    direction = vecs / safe[:, None]                  # (n, 3)
    visible = lens > visible_threshold

    # Perpendicular basis per arrow: choose (0,0,1) helper unless direction
    # is near-vertical, then use (1,0,0). Same anti-degeneracy trick as bonds.
    helpers = np.where(np.abs(direction[:, 2:3]) < 0.9,
                       np.array([0.0, 0.0, 1.0]),
                       np.array([1.0, 0.0, 0.0]))     # (n, 3)
    perp1 = np.cross(direction, helpers)
    p1norm = np.linalg.norm(perp1, axis=1, keepdims=True)
    perp1 = perp1 / np.maximum(p1norm, 1e-12)
    perp2 = np.cross(direction, perp1)

    angles = np.linspace(0.0, 2 * np.pi, n_sides, endpoint=False)
    cos_a = np.cos(angles)[None, :, None]             # (1, n_sides, 1)
    sin_a = np.sin(angles)[None, :, None]
    ring_unit = (cos_a * perp1[:, None, :] +
                 sin_a * perp2[:, None, :])           # (n, n_sides, 3)

    head_r = radius * head_radius_mult
    shaft_len = lens * (1.0 - head_length_frac)       # (n,)

    starts3 = starts[:, None, :]                      # (n, 1, 3)
    direction3 = direction[:, None, :]                # (n, 1, 3)
    shaft3 = shaft_len[:, None, None]                 # (n, 1, 1)

    base_ring = starts3 + radius * ring_unit                                # (n, n_sides, 3)
    top_shaft = starts3 + direction3 * shaft3 + radius * ring_unit          # (n, n_sides, 3)
    base_head = starts3 + direction3 * shaft3 + head_r * ring_unit          # (n, n_sides, 3)
    tip = starts + direction * lens[:, None]                                # (n, 3)

    # Collapse invisible arrows to a point at their atom position
    if not visible.all():
        inv = ~visible
        starts_inv = starts[inv][:, None, :]          # (n_inv, 1, 3)
        base_ring[inv] = starts_inv
        top_shaft[inv] = starts_inv
        base_head[inv] = starts_inv
        tip[inv] = starts[inv]

    # Pack into (n, 3*n_sides + 1, 3) then flatten
    verts_per_arrow = 3 * n_sides + 1
    out = np.empty((n, verts_per_arrow, 3), dtype=np.float64)
    out[:, 0:n_sides, :] = base_ring
    out[:, n_sides:2 * n_sides, :] = top_shaft
    out[:, 2 * n_sides:3 * n_sides, :] = base_head
    out[:, 3 * n_sides, :] = tip
    return out.reshape(n * verts_per_arrow, 3)


def _rebuild_arrows(scene, data, atom_idx, disp, current_pos) -> None:
    """Per-frame arrow update — vectorized, no Python loop over atoms.

    Topology was pre-allocated in ``create_arrow_object`` for all
    ``n_slots`` atoms in the supercell. Each frame we just rewrite the
    flat vertex buffer via ``foreach_set('co', ...)``, same ~ms pattern
    we use for bonds.

    Invisible arrows (|vec| < 0.02 Å) get their verts collapsed to the
    atom's position, making them visually disappear without a topology
    change.
    """
    obj = bpy.data.objects.get(ARROWS_OBJ)
    if obj is None:
        return
    obj.hide_viewport = False
    obj.hide_render = False

    props = scene.phonon_vision

    # Arrow vector = current-frame displacement × user's arrow_scale.
    # The arrow physically represents the *instantaneous* atomic velocity
    # direction times the scale (matching phononwebsite's convention).
    arrow_vec = disp * float(props.arrow_scale)

    # Geometry — tuned for "smooth cone+cylinder" look matching ascii-phonons
    masses_by_sp: dict[str, float] = {}
    for sp, m in zip(data.material.symbols, data.material.masses):
        masses_by_sp.setdefault(sp, float(m))
    avg_mass = float(np.mean(list(masses_by_sp.values())))
    typical_radius = _atomic_radius_hint(avg_mass)
    radius = typical_radius * 0.20
    head_frac = 0.30
    head_mult = 2.6
    _, _, n_sides = _quality_settings(scene)

    # Sanity: topology mismatch → fall back to full rebuild.
    # (This can happen if quality changed after scene was built.)
    n_atoms = len(current_pos)
    verts_per_arrow = 3 * n_sides + 1
    expected_verts = n_atoms * verts_per_arrow
    if len(obj.data.vertices) != expected_verts:
        create_arrow_object(scene)
        obj = bpy.data.objects.get(ARROWS_OBJ)
        if obj is None or len(obj.data.vertices) != expected_verts:
            return

    flat = _compute_arrow_verts_vectorized(
        current_pos, arrow_vec, radius, head_frac, head_mult, n_sides,
    ).ravel().astype(np.float32, copy=False)
    obj.data.vertices.foreach_set("co", flat)
    obj.data.update()


def _hide_arrows() -> None:
    obj = bpy.data.objects.get(ARROWS_OBJ)
    if obj is not None:
        obj.hide_viewport = True
        obj.hide_render = True


def _find_species_objects() -> dict[str, bpy.types.Object]:
    out = {}
    for obj in bpy.data.objects:
        if obj.name.startswith(ATOMS_PREFIX):
            sp = obj.name[len(ATOMS_PREFIX):]
            out[sp] = obj
    return out


# ============================================================
# Top-level entry points
# ============================================================
def build_scene_from_data(scene) -> None:
    create_atom_objects(scene)
    create_arrow_object(scene)
    create_cell_object(scene)
    create_bond_object(scene)
    refresh_scene_for_current_mode(scene)
    refresh_visibility(scene)


def refresh_scene_for_current_mode(scene) -> None:
    update_atom_positions(scene)


def refresh_bonds(scene) -> None:
    """Rebuild the bond *list* (e.g. when bond_tolerance changes)."""
    if get_loaded_data(scene) is None:
        return
    try:
        create_bond_object(scene)
        update_atom_positions(scene)   # populate geometry from current pos
    except Exception as e:
        print(f"[phonon-vision] bond rebuild failed: {e}")


def refresh_species_visuals(scene) -> None:
    """Apply current ``species_overrides`` colors and radii to existing objects.

    Edits the template sphere mesh and material *in place*, preserving the
    Object pointer that DupliVerts references. This is the difference
    between the user seeing the change immediately (in-place) versus needing
    a depsgraph kick (object replacement). Per-species bond half-cylinders
    auto-refresh because they reference the same ``PV.Mat.<species>``
    materials.
    """
    if get_loaded_data(scene) is None:
        return
    n_rings, n_segs, _ = _quality_settings(scene)

    touched_mats = []
    for entry in scene.phonon_vision.species_overrides:
        sp = entry.name
        rgb = (float(entry.color[0]), float(entry.color[1]), float(entry.color[2]))
        radius = float(entry.radius)
        vib = float(scene.phonon_vision.species_vibrancy)

        template = bpy.data.objects.get(SPHERE_PREFIX + sp)
        atoms_obj = bpy.data.objects.get(ATOMS_PREFIX + sp)
        if template is None or atoms_obj is None:
            continue

        # 1) Update the material in place — the data block stays the same,
        #    its nodes and diffuse_color are rewritten. Bond half-cylinders
        #    share this material, so they update at no extra cost.
        _get_or_build_species_material(sp, rgb, emission_strength=vib)

        # 2) Rebuild the sphere mesh data with the new radius and swap it
        #    onto the existing template object (so the DupliVerts parent link
        #    stays). ORDER IS CRITICAL: capture the old mesh by NAME (a string,
        #    not a datablock ref), build the new mesh (Blender auto-suffixes
        #    if the name collides, so it is never the mesh currently bound to
        #    the template), swap, THEN remove the old mesh by name. We must
        #    NOT remove a mesh while it is still the template's active .data —
        #    bpy.data.meshes.remove(do_unlink=True) cascades to deleting the
        #    dependent object, which previously orphaned the template and made
        #    `template.data` raise "StructRNA of type Object has been removed".
        old_mesh_name = template.data.name if template.data is not None else None
        new_mesh = _build_sphere_mesh_only(sp, radius, n_rings=n_rings, n_segs=n_segs)
        template.data = new_mesh
        # Re-assign the (just-rebuilt) material to the new mesh
        mat = bpy.data.materials.get(f"PV.Mat.{sp}")
        if mat is not None:
            if not new_mesh.materials:
                new_mesh.materials.append(mat)
            else:
                new_mesh.materials[0] = mat
            touched_mats.append(mat)
        # Now it is safe to free the old mesh — the template points at
        # new_mesh, so the old one has 0 users and removing it cannot
        # cascade to any object. Re-fetch by name; never touch a ref held
        # across the swap.
        if old_mesh_name is not None and old_mesh_name != new_mesh.name:
            old_mesh = bpy.data.meshes.get(old_mesh_name)
            if old_mesh is not None and old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)

    # Invalidate the edited materials, re-evaluate the depsgraph, and redraw.
    # (When called from an operator this runs in a clean context; when reached
    # via the deferred property-callback path it does too — see
    # state._schedule_geometry, drained on the idle timer.)
    _invalidate_and_redraw(touched_mats)


def _build_sphere_mesh_only(species: str, radius: float,
                            n_rings: int = 12, n_segs: int = 16) -> bpy.types.Mesh:
    """Build a UV sphere as a *Mesh datablock only* (no Object).

    Used by refresh_species_visuals so we can swap an existing template's
    mesh data without touching the Object — DupliVerts parent link stays
    intact and the change is visible immediately.
    """
    # Build with the canonical refresh name. We deliberately do NOT remove an
    # existing mesh of this name here: if one exists it is almost certainly the
    # template's CURRENT .data, and removing it (do_unlink=True) would cascade
    # to deleting the template object. Instead we let Blender auto-suffix the
    # name (".refresh" -> ".refresh.001"); the caller removes the genuinely-old
    # mesh by name AFTER swapping, when it has 0 users and is safe to free.
    mesh_name = f"PV.SphereMesh.{species}.refresh"
    mesh = bpy.data.meshes.new(mesh_name)

    verts = []
    faces = []
    verts.append((0.0, 0.0, radius))
    for ring in range(1, n_rings):
        theta = np.pi * ring / n_rings
        z = radius * np.cos(theta)
        r = radius * np.sin(theta)
        for seg in range(n_segs):
            phi = 2 * np.pi * seg / n_segs
            verts.append((r * np.cos(phi), r * np.sin(phi), z))
    verts.append((0.0, 0.0, -radius))
    bottom_idx = len(verts) - 1

    for s in range(n_segs):
        nx_seg = (s + 1) % n_segs
        faces.append((0, 1 + s, 1 + nx_seg))
    for ring in range(n_rings - 2):
        ring_start = 1 + ring * n_segs
        next_ring_start = 1 + (ring + 1) * n_segs
        for s in range(n_segs):
            nx_seg = (s + 1) % n_segs
            faces.append((
                ring_start + s, next_ring_start + s,
                next_ring_start + nx_seg, ring_start + nx_seg,
            ))
    last_ring_start = 1 + (n_rings - 2) * n_segs
    for s in range(n_segs):
        nx_seg = (s + 1) % n_segs
        faces.append((bottom_idx, last_ring_start + nx_seg, last_ring_start + s))

    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for poly in mesh.polygons:
        poly.use_smooth = True
    return mesh


def refresh_visibility(scene) -> None:
    props = scene.phonon_vision
    arrows = bpy.data.objects.get(ARROWS_OBJ)
    if arrows is not None:
        arrows.hide_viewport = not props.show_arrows
        arrows.hide_render = not props.show_arrows
    cell = bpy.data.objects.get(CELL_OBJ)
    if cell is not None:
        cell.hide_viewport = not props.show_unit_cell
        cell.hide_render = not props.show_unit_cell
    bonds = bpy.data.objects.get(BONDS_OBJ)
    if bonds is not None:
        bonds.hide_viewport = not props.show_bonds
        bonds.hide_render = not props.show_bonds


def clear_scene(scene) -> None:
    """Remove all PhononVision objects AND their orphan data-blocks.

    Over many load/clear cycles the .blend file accumulates orphan
    ``PV.Mat.X``, ``PV.SphereMesh.X``, ``PV.AtomsMesh.X``, ``PV.CellMat``,
    ``PV.BondMat``, ``PV.ArrowMat``, light data, and the PV camera.
    Without explicit cleanup these stay as zero-user data blocks until
    Blender's "Purge Orphan Data" is invoked manually.

    We collect everything with a ``PV.`` prefix across the relevant
    bpy.data containers and remove them all in one pass. Anything still
    in use elsewhere (shouldn't happen normally) is left alone by Blender.
    """
    # 1. Objects first, so children/parents and material slots release their refs.
    to_remove = [obj for obj in bpy.data.objects if obj.name.startswith("PV.")]
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)

    # 2. Meshes — orphans now that the objects are gone
    for mesh in list(bpy.data.meshes):
        if mesh.name.startswith("PV.") and mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    # 3. Materials — same
    for mat in list(bpy.data.materials):
        if mat.name.startswith("PV.") and mat.users == 0:
            bpy.data.materials.remove(mat)

    # 4. Light data
    for light in list(bpy.data.lights):
        if light.name.startswith("PV.") and light.users == 0:
            bpy.data.lights.remove(light)

    # 5. Camera data
    for cam in list(bpy.data.cameras):
        if cam.name.startswith("PV.") and cam.users == 0:
            bpy.data.cameras.remove(cam)

    # 6. The collection itself, if empty
    coll = bpy.data.collections.get(COLLECTION_NAME)
    if coll is not None and len(coll.objects) == 0:
        bpy.data.collections.remove(coll)
