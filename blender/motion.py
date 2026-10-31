"""phonon_vision.motion — bpy-free render-path kinematics plumbing.

The physics (Bloch phases, mass-weighted displacements, frame->phase) lives
in the vendored phonon_core engine and is parity-guarded + golden-covered
THERE. What was never testable before 0.7.11 is the add-on-local plumbing
between the engine and Blender's meshes:

1. the flatten -> custom-prop -> reshape(n_slots, 3) round-trip of
   ``all_base_pos`` / ``uc_indices`` (pack_geometry / unpack_geometry);
2. the per-frame engine call chain + the optional display clamp
   (displacements_at_frame / positions_at_frame);
3. the per-species ``foreach_set('co', ...)`` scatter layout
   (species_layout).

This module is that plumbing, factored out of scene_setup in the discipline
of bonds.py: pure numpy, no bpy, engine imports lazy (so plain-python tests
can load it and the sibling stub-loaded modules never need the engine).
scene_setup's remaining bpy work is ONLY: read custom props, call motion.*,
foreach_set. Behaviour is byte-identical to 0.7.10 — the float ops here are
the 0.7.10 in-scene lines moved verbatim, in the same order.

Unit convention (PINNED): 1 Blender unit ≡ 1 Å. Base positions come from
phonon_core.expand.expand_cell in Å and displacements from
compute_displacements_at_phase in Å; NOTHING here (or downstream) applies a
scale factor. tools/test_motion.py pins this.

Displacement convention (engine, for reference): the displacement of atom s
in unit cell ℓ at global phase φ is

    u_{ℓs}(φ) = A · Re[ e_s(q) · exp(i (q·R_ℓ + φ)) ] / sqrt(m_s)

— mass-weighted eigenvector convention as in Dove, *Introduction to Lattice
Dynamics* (1993), eq. 4.16, with the dynamical-matrix/eigenvector convention
of Togo & Tanaka, Scr. Mater. 108, 1 (2015) (phonopy), and Bloch phase
q·R_ℓ = 2π·q_frac·n exactly (fractional q, integer cell index n).

THE CLAMP IS NON-PHYSICAL: it rescales each over-long displacement vector
independently to |u| = cap, which distorts the eigenvector's amplitude
ratios e_k/sqrt(m_k) between atoms (a mode's shape is defined by exactly
those ratios). It exists to keep extreme-amplitude animations legible.
The physics-faithful path is clamp=False; tools/test_motion.py asserts
that path matches the engine and that clamping only ever shrinks vectors.
"""
from __future__ import annotations

import numpy as np


# ============================================================
# Custom-prop geometry round-trip
# ============================================================
def pack_geometry(base_pos: np.ndarray, uc_idx: np.ndarray):
    """(n_slots, 3) arrays -> flat python lists for Blender ID custom props.

    C-order (row-major) flatten, PINNED: slot-by-slot, x,y,z within a slot
    (numpy's default). unpack_geometry must invert this exactly; the pair is
    round-trip-tested in tools/test_motion.py.
    """
    flat_pos = np.asarray(base_pos, dtype=np.float64).flatten().tolist()
    flat_uc = np.asarray(uc_idx, dtype=np.int64).flatten().tolist()
    return flat_pos, flat_uc


def unpack_geometry(flat_pos, flat_uc, n_slots: int):
    """Inverse of pack_geometry: custom-prop lists -> (base_pos, uc_idx).

    C-order reshape (numpy default), matching pack_geometry's C-order
    flatten — the 0.7.10 in-scene ops verbatim.
    """
    base_pos = np.asarray(flat_pos, dtype=np.float64).reshape(int(n_slots), 3)
    uc_idx = np.asarray(flat_uc, dtype=np.int64).reshape(int(n_slots), 3)
    return base_pos, uc_idx


# ============================================================
# Per-frame displacements / positions
# ============================================================
def displacements_at_frame(mode, material, uc_idx, *,
                           frame: int, frames_per_cycle: int, frame_start: int,
                           amplitude: float, atom_indices,
                           use_canonical_phase: bool = False,
                           clamp: bool = False, cap=None) -> np.ndarray:
    """Per-slot Cartesian displacement (Å) at one animation frame.

    The engine call chain of scene_setup.update_atom_positions (0.7.10),
    verbatim: canonicalize_mode (iff use_canonical_phase) -> frame_to_phase
    -> compute_bloch_phases -> compute_displacements_at_phase, then the
    optional soft clamp. Exposed separately from positions_at_frame because
    the arrow builder consumes the displacement itself.

    clamp/cap: iff ``clamp`` is True, every vector longer than ``cap`` (Å)
    is rescaled to exactly ``cap``; shorter vectors are untouched. See the
    module docstring: this is a NON-PHYSICAL display aid — it breaks the
    inter-atom amplitude ratios that define the mode shape.
    """
    from phonon_core.animation import frame_to_phase
    from phonon_core.kinematics import (
        canonicalize_mode,
        compute_bloch_phases,
        compute_displacements_at_phase,
    )

    if use_canonical_phase:
        mode = canonicalize_mode(mode)

    phase = frame_to_phase(
        int(frame),
        int(frames_per_cycle),
        frame_start=int(frame_start),
    )

    bloch = compute_bloch_phases(mode.q_point, uc_idx)
    disp = compute_displacements_at_phase(
        mode, material,
        amplitude=float(amplitude),
        phase=float(phase),
        atom_indices=atom_indices,
        bloch_phases=bloch,
    )

    # Optional clamp: if any per-atom |disp| exceeds cap, soft-cap by scaling
    # its vector down to that length (0.7.10 lines verbatim). NON-PHYSICAL.
    if clamp:
        cap = float(cap)
        lens = np.linalg.norm(disp, axis=1)
        too_long = lens > cap
        if np.any(too_long):
            # Scale each over-long vector to exactly `cap` magnitude
            disp = disp.copy()
            scale = np.where(too_long, cap / np.maximum(lens, 1e-12), 1.0)
            disp = disp * scale[:, None]

    return disp


def positions_at_frame(mode, material, base_pos, uc_idx, *,
                       frame: int, frames_per_cycle: int, frame_start: int,
                       amplitude: float, atom_indices,
                       use_canonical_phase: bool = False,
                       clamp: bool = False, cap=None) -> np.ndarray:
    """(n_slots, 3) atom positions in Blender units (≡ Å) at one frame.

    base_pos + displacements_at_frame(...): NO unit scaling anywhere —
    1 BU ≡ 1 Å is pinned by tools/test_motion.py and enforced end-to-end by
    tools/conformance_headless.py (this function is its bpy-free reference).
    """
    disp = displacements_at_frame(
        mode, material, uc_idx,
        frame=frame, frames_per_cycle=frames_per_cycle,
        frame_start=frame_start, amplitude=amplitude,
        atom_indices=atom_indices,
        use_canonical_phase=use_canonical_phase,
        clamp=clamp, cap=cap,
    )
    return np.asarray(base_pos, dtype=np.float64) + disp


# ============================================================
# Per-species mesh scatter
# ============================================================
def species_layout(current_pos: np.ndarray, slot_indices) -> np.ndarray:
    """Flat float64 ``co`` array for one species' vertex mesh.

    Exactly what ``mesh.vertices.foreach_set('co', ...)`` consumes: vertex k
    of the species mesh is global slot ``slot_indices[k]``, laid out
    x,y,z per vertex in slot_indices order (C-order flatten) — the 0.7.10
    in-scene ops verbatim.
    """
    sel = np.asarray(current_pos, dtype=np.float64)[
        np.asarray(slot_indices, dtype=np.int64)]
    return sel.astype(np.float64).flatten()


# ============================================================
# ENGINE SOURCE (0.8.0) — the A2 numpy engine as a phonon data source
# ============================================================
# Bpy-free by the same discipline as everything above: the ENGINE CALL CHAIN
# is this module's charter (docstring point 2), and the job loader + the
# plain-python tests must reconstruct engine sources without Blender. The
# vendored-wheel imports stay LAZY (inside functions) so stub-loaded sibling
# modules and the bpy-free suites never pay for them at import time.

def sniff_phonopy_yaml(path: str, max_bytes: int = 65536) -> bool:
    """Content sniff: is this a phonopy.yaml (structure file), not band/mesh?

    A phonopy.yaml / phonopy_disp.yaml / phonopy_params.yaml carries a
    top-level ``supercell_matrix:`` key (band.yaml / mesh.yaml never do).
    Line-anchored on the first ``max_bytes`` so multi-MB band.yaml files are
    never fully read just to be routed.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(max_bytes)
    except OSError:
        return False
    for line in head.splitlines():
        if line.startswith(b"supercell_matrix:"):
            return True
    return False


class EngineSource:
    """A loaded A2 engine masquerading as a phonon data source.

    Duck-types the exact seam file sources satisfy — ``.material``,
    ``.get_mode(q_index, band_index)``, ``.num_qpoints`` — so
    scene_setup.update_atom_positions, the frame handler, the Mode Navigator
    readout and the imaginary badge all consume engine modes UNCHANGED.
    The difference: q is CONTINUOUS. ``set_q`` evaluates D(q) via the
    vendored A2 engine (~0.07 ms per evaluation — live-scrub budget) and
    caches (freqs, eigvecs); ``get_mode`` reads the cache, IGNORING q_index
    (an engine source has no q grid — the current q is engine state, set by
    the q_frac_* properties / the job loader).
    """

    kind = "ENGINE"
    num_qpoints = 1          # the ONE live q (Mode Navigator hides the index)

    def __init__(self, engine, yaml_path: str, fc_path: str):
        self.engine = engine
        self.yaml_path = yaml_path
        self.fc_path = fc_path
        self.material = engine.material
        self._q = None
        self._freqs = None
        self._eigvecs = None
        self.set_q((0.0, 0.0, 0.0))

    @property
    def n_bands(self) -> int:
        return int(self._freqs.shape[0])

    @property
    def current_q(self):
        return self._q.copy()

    @property
    def frequencies(self):
        """(N_b,) signed THz at the current q (a copy)."""
        return self._freqs.copy()

    def set_q(self, q_frac) -> np.ndarray:
        """Evaluate D(q) at ``q_frac`` and cache the modes; returns freqs."""
        from phonon_core.engine import evaluate_dq_np   # lazy (vendored wheel)
        q = np.asarray(q_frac, dtype=np.float64).reshape(3)
        freqs, eigvecs = evaluate_dq_np(self.engine, q)
        self._q, self._freqs, self._eigvecs = q, freqs, eigvecs
        return freqs

    def get_mode(self, q_index: int, band_index: int):
        """PhononMode at the CURRENT q (q_index ignored — engine state)."""
        from phonon_core.types import PhononMode        # lazy (vendored wheel)
        b = int(band_index)
        if not 0 <= b < self._freqs.shape[0]:
            raise IndexError(f"band_index {b} out of range "
                             f"(0..{self._freqs.shape[0] - 1})")
        return PhononMode(q_point=self._q.copy(), band_index=b,
                          frequency=float(self._freqs[b]),
                          eigenvector=self._eigvecs[b])


def load_engine_source(yaml_path: str, force_constants: str = None) -> EngineSource:
    """Build an :class:`EngineSource` from phonopy.yaml (+ FORCE_CONSTANTS).

    FC discovery mirrors the A2 engine's own rule (explicit arg -> sibling
    text FORCE_CONSTANTS -> sibling force_constants.hdf5) so the RESOLVED
    path can be displayed in the Data panel and serialized into jobs. Errors
    (missing FC, the BORN refusal, parse failures) propagate VERBATIM — the
    engine's own messages are the honest operator error text.
    """
    import os
    from phonon_core.engine import load_engine_np       # lazy (vendored wheel)
    yaml_path = os.path.abspath(os.fspath(yaml_path))
    here = os.path.dirname(yaml_path)
    fc = force_constants
    if fc is None:
        for cand in (os.path.join(here, "FORCE_CONSTANTS"),
                     os.path.join(here, "force_constants.hdf5")):
            if os.path.exists(cand):
                fc = cand
                break
    eng = load_engine_np(yaml_path, force_constants=fc)
    return EngineSource(eng, yaml_path,
                        os.path.abspath(fc) if fc else "(embedded)")
