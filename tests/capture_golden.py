"""Capture golden supercell+image-atom output from the VERBATIM 0.5.3 logic.

These two functions are copied byte-for-byte from addon-0.5.3/scene_setup.py
(generate_supercell_slots, generate_image_atom_slots) with bpy removed. The
output is the acceptance gate for phonon_core.expand's "halo" mode: the new
unified expand_cell must reproduce these arrays exactly.
"""
import sys
import os
from pathlib import Path

import numpy as np

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir / "src"))

from phonon_core import load_band_yaml, load_band_hdf5
from phonon_core.radii import covalent_radius


# ---- VERBATIM from addon-0.5.3/scene_setup.py (bpy-free) ----
def generate_supercell_slots(material, nx, ny, nz):
    n_atoms = material.num_atoms
    cart_uc = material.cartesian_coords
    lat = material.lattice
    uc_idx, atom_idx, pos = [], [], []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                R = ix * lat[0] + iy * lat[1] + iz * lat[2]
                for s in range(n_atoms):
                    uc_idx.append([ix, iy, iz])
                    atom_idx.append(s)
                    pos.append(cart_uc[s] + R)
    return (np.asarray(uc_idx, dtype=np.int64),
            np.asarray(atom_idx, dtype=np.int64),
            np.asarray(pos, dtype=np.float64))


def generate_image_atom_slots(material, base_uc_idx, base_atom_idx, base_pos,
                               nx, ny, nz, bond_tolerance,
                               *, max_iterations=2, verbose=False):
    lat = material.lattice
    species_radii = {sp: float(covalent_radius(sp))
                     for sp in set(material.symbols)}
    sc_a = nx * lat[0]
    sc_b = ny * lat[1]
    sc_c = nz * lat[2]
    all_pos = np.asarray(base_pos, dtype=np.float64).copy()
    all_atom_idx = np.asarray(base_atom_idx, dtype=np.int64).copy()
    all_uc_idx = np.asarray(base_uc_idx, dtype=np.int64).copy()
    all_radii = np.array([species_radii[material.symbols[int(a)]]
                          for a in all_atom_idx], dtype=np.float64)
    src_pos = np.asarray(base_pos, dtype=np.float64)
    src_atom_idx = np.asarray(base_atom_idx, dtype=np.int64)
    src_uc_idx = np.asarray(base_uc_idx, dtype=np.int64)
    src_radii = np.array([species_radii[material.symbols[int(a)]]
                          for a in src_atom_idx], dtype=np.float64)
    image_uc_acc = []
    image_atom_acc = []
    image_pos_acc = []
    seen_keys = set()
    for i in range(len(all_pos)):
        seen_keys.add((int(all_uc_idx[i, 0]), int(all_uc_idx[i, 1]),
                       int(all_uc_idx[i, 2]), int(all_atom_idx[i])))
    triplets = [(ix, iy, iz)
                for ix in (-1, 0, 1)
                for iy in (-1, 0, 1)
                for iz in (-1, 0, 1)
                if (ix, iy, iz) != (0, 0, 0)]
    for iteration in range(max_iterations):
        max_cutoff = 2.0 * float(all_radii.max()) * bond_tolerance
        box_min = all_pos.min(axis=0) - max_cutoff
        box_max = all_pos.max(axis=0) + max_cutoff
        new_pos = []
        new_atom_idx = []
        new_uc_idx = []
        for ix, iy, iz in triplets:
            T = ix * sc_a + iy * sc_b + iz * sc_c
            translated = src_pos + T
            near_mask = ((translated > box_min).all(axis=1) &
                         (translated < box_max).all(axis=1))
            if not np.any(near_mask):
                continue
            candidate_slots = np.flatnonzero(near_mask)
            for c in candidate_slots:
                c = int(c)
                src_uc = src_uc_idx[c]
                cand_uc = (int(src_uc[0]) + ix * nx,
                           int(src_uc[1]) + iy * ny,
                           int(src_uc[2]) + iz * nz)
                if not (-1 <= cand_uc[0] <= nx and
                        -1 <= cand_uc[1] <= ny and
                        -1 <= cand_uc[2] <= nz):
                    continue
                cand_atom = int(src_atom_idx[c])
                cand_key = (cand_uc[0], cand_uc[1], cand_uc[2], cand_atom)
                if cand_key in seen_keys:
                    continue
                cand_radius = src_radii[c]
                d = all_pos - translated[c]
                dist = np.linalg.norm(d, axis=1)
                cutoff = (cand_radius + all_radii) * bond_tolerance
                within = (dist > 0.5) & (dist < cutoff)
                if not np.any(within):
                    continue
                seen_keys.add(cand_key)
                new_pos.append(translated[c])
                new_atom_idx.append(cand_atom)
                new_uc_idx.append(cand_uc)
                image_uc_acc.append(cand_uc)
                image_atom_acc.append(cand_atom)
                image_pos_acc.append(translated[c])
        if not new_pos:
            break
        new_pos_arr = np.asarray(new_pos, dtype=np.float64)
        new_atom_idx_arr = np.asarray(new_atom_idx, dtype=np.int64)
        new_uc_idx_arr = np.asarray(new_uc_idx, dtype=np.int64)
        new_radii = np.array([species_radii[material.symbols[int(a)]]
                              for a in new_atom_idx_arr], dtype=np.float64)
        all_pos = np.vstack([all_pos, new_pos_arr])
        all_atom_idx = np.concatenate([all_atom_idx, new_atom_idx_arr])
        all_uc_idx = np.vstack([all_uc_idx, new_uc_idx_arr])
        all_radii = np.concatenate([all_radii, new_radii])
    if not image_uc_acc:
        return (np.zeros((0, 3), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, 3), dtype=np.float64))
    return (np.asarray(image_uc_acc, dtype=np.int64),
            np.asarray(image_atom_acc, dtype=np.int64),
            np.asarray(image_pos_acc, dtype=np.float64))


def halo(material, nx, ny, nz, tol=1.2):
    uc, a, p = generate_supercell_slots(material, nx, ny, nz)
    iuc, ia, ip = generate_image_atom_slots(material, uc, a, p, nx, ny, nz, tol)
    if len(ia):
        uc = np.concatenate([uc, iuc], axis=0)
        a = np.concatenate([a, ia], axis=0)
        p = np.concatenate([p, ip], axis=0)
    return uc, a, p


UP = os.environ.get("PV_GOLDEN_SOURCE_DIR", ".")
cases = {
    "snse2_3x3x1": (load_band_yaml(f"{UP}/2d_SnSe2_band.yaml"), (3, 3, 1)),
    "wse2_3x3x1":  (load_band_yaml(f"{UP}/2d_WSe2_band.yaml"), (3, 3, 1)),
}

golden = {}
for name, (data, (nx, ny, nz)) in cases.items():
    mat = data.material
    # default bond_tolerance in the add-on is 1.2 (verify below)
    uc, a, p = halo(mat, nx, ny, nz, tol=1.2)
    golden[f"{name}__uc"] = uc
    golden[f"{name}__atom"] = a
    golden[f"{name}__pos"] = p
    print(f"{name}: n_uc_atoms={mat.num_atoms}  halo_total={len(a)}  "
          f"images={len(a) - mat.num_atoms*nx*ny*nz}")
    # also report fractional-coord wrap status
    fc = mat.fractional_coords
    wrapped = fc - np.floor(fc)
    moved = not np.allclose(wrapped, fc, atol=0, rtol=0)
    print(f"    fractional in [0,1)? {'NO (would wrap)' if moved else 'yes (no-op)'}")

np.savez(base_dir / "tests" / "golden_expand.npz", **golden)
print("\nSaved golden to phonon-core/tests/golden_expand.npz")
