#!/usr/bin/env python3
"""Conformance + regression tests for motion.py (0.7.11) — plain Python, no bpy.

motion.py is the bpy-free render-path plumbing between the vendored
phonon_core engine and Blender's meshes. The engine physics is golden-covered
in the core line; THESE tests close the add-on-local links:

- the flatten -> custom-prop -> reshape round-trip (pack/unpack_geometry);
- ENGINE CONFORMANCE: positions_at_frame == base + A·Re[engine displacement]
  with the engine's own compute_bloch_phases, recomputed INDEPENDENTLY here,
  over a full period, for a Gamma AND a non-Gamma mode on a 2x2x1 supercell;
- the Å->BU pin (1 Blender unit ≡ 1 Å, no hidden scale);
- the clamp (NON-PHYSICAL display aid: it rescales over-long vectors
  independently, distorting the eigenvector amplitude ratios e_k/sqrt(m_k)
  that define the mode shape — the physics-faithful path is clamp=False);
- the foreach_set('co', ...) scatter layout;
- a 0.7.10 REGRESSION GOLDEN (current_pos captured from the PRE-refactor
  scene_setup.update_atom_positions lines, float64-hex-exact).

The engine comes from the tree's own vendored wheel (zip-imported), so the
conformance target is EXACTLY what ships:

    python tools/test_motion.py
"""
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
from _subset import WHEEL_VERSION  # single source: tools/_subset.py
_WHEEL = os.path.join(_HERE, os.pardir, "wheels",
                      f"phonon_core-{WHEEL_VERSION}-py3-none-any.whl")
if _WHEEL not in sys.path:
    sys.path.insert(0, _WHEEL)   # pure-python wheel: zip-importable

from phonon_core.animation import frame_to_phase                    # noqa: E402
from phonon_core.expand import expand_cell                          # noqa: E402
from phonon_core.kinematics import (                                # noqa: E402
    canonicalize_mode,
    compute_bloch_phases,
    compute_displacements_at_phase,
)
from phonon_core.types import Material, PhononMode                  # noqa: E402


def _load_motion_module():
    spec = importlib.util.spec_from_file_location(
        "pv_motion", os.path.join(_HERE, os.pardir, "motion.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_motion_module()

# ---- FIXED FIXTURE (the golden below was captured from the 0.7.10 code
# path on exactly this; do not change any value without recapturing) ----
LAT = np.array([[4.0, 0.0, 0.0],
                [-2.0, 3.4641016151377544, 0.0],
                [0.0, 0.0, 6.5]])
MAT = Material(
    lattice=LAT,
    reciprocal_lattice=np.linalg.inv(LAT).T,
    symbols=["Sn", "Se"],
    masses=[118.71, 78.971],
    fractional_coords=[[0.0, 0.0, 0.0],
                       [1.0 / 3.0, 2.0 / 3.0, 0.25]],
)
EIG = np.array([[0.10 + 0.05j, -0.02 + 0.00j, 0.00 + 0.02j],
                [0.03 + 0.00j, 0.00 - 0.04j, 0.01 + 0.01j]])
Q_NONGAMMA = np.array([0.5, 0.25, 0.0])
Q_GAMMA = np.array([0.0, 0.0, 0.0])
FRAME, FPC, FSTART = 7, 24, 1
AMPLITUDE = 8.0
CAP = 0.05

_EXP = expand_cell(MAT, 2, 2, 1, mode="strict")
UC_IDX = _EXP.n_int
ATOM_IDX = _EXP.j
BASE_POS = _EXP.r_cart
N_SLOTS = len(ATOM_IDX)


def _mode(q):
    return PhononMode(q_point=np.asarray(q, dtype=np.float64), band_index=3,
                      frequency=2.5, eigenvector=EIG)


def _from_hex(rows):
    return np.array([[float.fromhex(h) for h in row] for row in rows],
                    dtype=np.float64)


# Captured 2026-07-06 from the PRE-refactor 0.7.10 update_atom_positions
# lines (verbatim copy driven by the vendored wheel) on the fixture above:
# non-Gamma mode, frame 7/24 (start 1), amplitude 8, canonical phase ON.
# CLAMPED = clamp on at cap 0.05 Å (engages on exactly 2 of the 8 slots).
GOLDEN_UNCLAMPED = _from_hex([
    ["0x1.f9676e365465bp-58", "-0x1.ae6625307c6bap-8", "-0x1.ae6625307c6b9p-7"],
    ["0x1.8bc4fff484101p-7", "0x1.2bb9d7038f3c9p+1", "0x1.9ef827555cfd5p+0"],
    ["0x1.0000000000000p+2", "0x1.ae6625307c6bcp-8", "0x1.ae6625307c6b9p-7"],
    ["0x1.fe743b000b7bfp+1", "0x1.237b11ae7726fp+1", "0x1.a107d8aaa302bp+0"],
    ["-0x1.0a81fe686f09ap+1", "0x1.bd1614aab5471p+1", "-0x1.ae6625307c6bap-8"],
    ["-0x1.031789ffe9082p+1", "0x1.7288ea19e700ep+2", "0x1.9ce8760016f7ep+0"],
    ["0x1.0a81fe686f09ap+1", "0x1.b9b94860544e3p+1", "0x1.ae6625307c6bdp-8"],
    ["0x1.031789ffe9082p+1", "0x1.707938c4a0fb8p+2", "0x1.a31789ffe9082p+0"],
])
GOLDEN_CLAMPED = _from_hex([
    ["0x1.f9676e365465bp-58", "-0x1.ae6625307c6bap-8", "-0x1.ae6625307c6b9p-7"],
    ["0x1.8bc4fff484101p-7", "0x1.2bb9d7038f3c9p+1", "0x1.9ef827555cfd5p+0"],
    ["0x1.0000000000000p+2", "0x1.ae6625307c6bcp-8", "0x1.ae6625307c6b9p-7"],
    ["0x1.fe743b000b7bfp+1", "0x1.237b11ae7726fp+1", "0x1.a107d8aaa302bp+0"],
    ["-0x1.064ccc6594b93p+1", "0x1.bc69babebb4ebp+1", "-0x1.020c393684090p-8"],
    ["-0x1.031789ffe9082p+1", "0x1.7288ea19e700ep+2", "0x1.9ce8760016f7ep+0"],
    ["0x1.064ccc6594b93p+1", "0x1.ba65a24c4e469p+1", "0x1.020c393684092p-8"],
    ["0x1.031789ffe9082p+1", "0x1.707938c4a0fb8p+2", "0x1.a31789ffe9082p+0"],
])


def _positions(q, frame, *, clamp=False, canonical=True):
    return M.positions_at_frame(
        _mode(q), MAT, BASE_POS, UC_IDX,
        frame=frame, frames_per_cycle=FPC, frame_start=FSTART,
        amplitude=AMPLITUDE, atom_indices=ATOM_IDX,
        use_canonical_phase=canonical, clamp=clamp, cap=CAP)


def _engine_reference(q, frame, *, canonical=True):
    """INDEPENDENT recompute: base + A·Re[engine displacement at unit
    amplitude], engine's own frame_to_phase / compute_bloch_phases. The
    engine applies amplitude as a final linear factor, so A × disp(A=1)
    is bit-identical to disp(A) — this pins the amplitude semantics too."""
    mode = _mode(q)
    if canonical:
        mode = canonicalize_mode(mode)
    phase = frame_to_phase(frame, FPC, frame_start=FSTART)
    bloch = compute_bloch_phases(mode.q_point, UC_IDX)
    unit_disp = compute_displacements_at_phase(
        mode, MAT, amplitude=1.0, phase=phase,
        atom_indices=ATOM_IDX, bloch_phases=bloch)
    return BASE_POS + AMPLITUDE * unit_disp


# --- ROUND-TRIP --------------------------------------------------------------
def test_pack_unpack_roundtrip_exact():
    flat_pos, flat_uc = M.pack_geometry(BASE_POS, UC_IDX)
    assert isinstance(flat_pos, list) and isinstance(flat_uc, list)
    assert len(flat_pos) == 3 * N_SLOTS and len(flat_uc) == 3 * N_SLOTS
    base2, uc2 = M.unpack_geometry(flat_pos, flat_uc, N_SLOTS)
    assert np.array_equal(base2, BASE_POS) and base2.dtype == np.float64
    assert np.array_equal(uc2, UC_IDX) and uc2.dtype == np.int64


# --- ENGINE CONFORMANCE (full period, Gamma and non-Gamma) -------------------
def test_engine_conformance_gamma():
    for f in range(FPC + 1):    # 0..frames_per_cycle inclusive (wrap frame too)
        got = _positions(Q_GAMMA, f)
        ref = _engine_reference(Q_GAMMA, f)
        assert np.allclose(got, ref, rtol=0.0, atol=1e-13), f"frame {f}"


def test_engine_conformance_non_gamma():
    # 2x2x1 supercell at q = (0.5, 0.25, 0): the Bloch envelope is exercised
    # for real — cells at n1=1 flip sign, n2=1 quarter-turn. Guard that the
    # phases actually differ between cells (else this test proves nothing).
    bloch = compute_bloch_phases(Q_NONGAMMA, UC_IDX)
    assert len(np.unique(np.round(bloch, 12))) > 1
    for f in range(FPC + 1):
        got = _positions(Q_NONGAMMA, f)
        ref = _engine_reference(Q_NONGAMMA, f)
        assert np.allclose(got, ref, rtol=0.0, atol=1e-13), f"frame {f}"


# --- Å -> BU PIN --------------------------------------------------------------
def test_angstrom_to_bu_pin():
    # Single atom, mass 4, real unit eigenvector, amplitude 2, phase 0:
    # disp = 2 · Re[1·e^0] / sqrt(4) = 1.0 EXACTLY (all powers of two).
    # The atom must move by exactly 1.0 Blender unit — any hidden Å->BU
    # scale, however close to 1, breaks bitwise equality here.
    mat1 = Material(lattice=np.eye(3), reciprocal_lattice=np.eye(3),
                    symbols=["Sn"], masses=[4.0],
                    fractional_coords=[[0.0, 0.0, 0.0]])
    mode1 = PhononMode(q_point=np.zeros(3), band_index=0, frequency=1.0,
                       eigenvector=np.array([[1.0 + 0.0j, 0.0, 0.0]]))
    base1 = np.array([[0.25, 0.5, 0.75]])
    got = M.positions_at_frame(
        mode1, mat1, base1, np.zeros((1, 3), dtype=np.int64),
        frame=0, frames_per_cycle=FPC, frame_start=0,
        amplitude=2.0, atom_indices=np.array([0], dtype=np.int64),
        use_canonical_phase=False)
    moved = got - base1
    assert moved[0, 0] == 1.0 and moved[0, 1] == 0.0 and moved[0, 2] == 0.0


# --- CLAMP (non-physical display aid) ----------------------------------------
def test_clamp_bounded_distortion():
    # clamp=False is the PHYSICS-FAITHFUL path: it must equal the engine.
    faithful = _positions(Q_NONGAMMA, FRAME, clamp=False)
    assert np.allclose(faithful, _engine_reference(Q_NONGAMMA, FRAME),
                       rtol=0.0, atol=1e-13)
    # clamp=True: over-long vectors shrink to EXACTLY cap, same direction;
    # in-cap vectors are untouched bitwise. The per-atom rescale is what
    # makes the clamp NON-PHYSICAL — it distorts the inter-atom amplitude
    # ratios e_k/sqrt(m_k) that define the mode shape. Bounded, though:
    # nothing ever grows.
    clamped = _positions(Q_NONGAMMA, FRAME, clamp=True)
    d_raw = faithful - BASE_POS
    d_cl = clamped - BASE_POS
    lens_raw = np.linalg.norm(d_raw, axis=1)
    lens_cl = np.linalg.norm(d_cl, axis=1)
    over = lens_raw > CAP
    assert over.any() and not over.all()   # fixture straddles the cap
    assert np.allclose(lens_cl[over], CAP, rtol=0.0, atol=1e-12)
    assert np.array_equal(d_cl[~over], d_raw[~over])          # untouched
    assert np.all(lens_cl <= lens_raw + 1e-15)                # only shrinks
    # direction preserved: clamped vector parallel to raw (positive scale)
    dots = np.einsum("ij,ij->i", d_cl[over], d_raw[over])
    assert np.all(dots > 0)
    cross = np.cross(d_cl[over], d_raw[over])
    assert np.allclose(cross, 0.0, atol=1e-12)


# --- SCATTER LAYOUT -----------------------------------------------------------
def test_species_scatter_layout():
    # Vertex k of a species mesh is global slot slot_indices[k]; foreach_set
    # consumes x,y,z per vertex in that order. Distinct values per slot and
    # axis pin the exact interleaving.
    current = np.array([[i, 10.0 + i, 20.0 + i] for i in range(6)],
                       dtype=np.float64)
    flat = M.species_layout(current, [3, 0, 2])
    assert flat.dtype == np.float64
    assert flat.tolist() == [3.0, 13.0, 23.0,
                             0.0, 10.0, 20.0,
                             2.0, 12.0, 22.0]


# --- 0.7.10 REGRESSION GOLDEN --------------------------------------------------
def test_regression_0_7_10_golden():
    # The refactor must be byte-identical to the 0.7.10 in-scene code path:
    # float64-hex goldens, exact equality (no tolerance).
    got_unclamped = _positions(Q_NONGAMMA, FRAME, clamp=False)
    got_clamped = _positions(Q_NONGAMMA, FRAME, clamp=True)
    assert np.array_equal(got_unclamped, GOLDEN_UNCLAMPED)
    assert np.array_equal(got_clamped, GOLDEN_CLAMPED)
    # and the two goldens differ exactly where the clamp engaged (2 slots)
    differing = np.any(GOLDEN_UNCLAMPED != GOLDEN_CLAMPED, axis=1)
    assert int(differing.sum()) == 2



# ---------------------------------------------------------------------------
# 0.8.0: ENGINE-fed displacement path — the Bloch phase at ARBITRARY q
# ---------------------------------------------------------------------------
def test_engine_mode_bloch_phase_exact():
    """Feed evaluate_dq_np output (vendored wheel) at an INCOMMENSURATE q
    through the displacement path and assert the Bloch relation EXACTLY:
    cells related by integer n differ by e^{2*pi*i q.n} — reconstructed from
    two quarter-cycle frames as Z = d(phi=0) - i*d(phi=pi/2), so
    Z[cell n2] == Z[cell n1] * e^{2*pi*i q.(n2-n1)} per component. Amplitude,
    1/sqrt(m) and the canonical mode phase are per-atom/mode constants that
    cancel in the cross-cell ratio; clamp is OFF (the faithful path)."""
    import tempfile
    sys.path.insert(0, _HERE)                       # tools/ for the fixture
    from build_vendored_wheel import _write_smoke_fixture
    from phonon_core.engine import load_engine_np, evaluate_dq_np

    q = np.array([0.137, 0.291, 0.0])               # incommensurate, generic
    with tempfile.TemporaryDirectory() as d:
        _write_smoke_fixture(d)
        eng = load_engine_np(os.path.join(d, "phonopy.yaml"))
    freqs, eigs = evaluate_dq_np(eng, q)
    band = 4
    mode = PhononMode(q_point=q, band_index=band,
                      frequency=float(freqs[band]), eigenvector=eigs[band])

    mat = eng.material
    exp = expand_cell(mat, 2, 2, 1, mode="strict")
    base, uc_idx, atom_idx = exp.r_cart, exp.n_int, exp.j
    fpc, fstart = 24, 1
    assert abs(frame_to_phase(fstart, fpc, frame_start=fstart)) < 1e-15
    assert abs(frame_to_phase(fstart + 6, fpc, frame_start=fstart)
               - np.pi / 2) < 1e-12

    def disp(frame):
        pos = M.positions_at_frame(
            mode, mat, base, uc_idx, frame=frame, frames_per_cycle=fpc,
            frame_start=fstart, amplitude=8.0, atom_indices=atom_idx,
            use_canonical_phase=True, clamp=False, cap=0.05)
        return pos - base

    Z = disp(fstart) - 1j * disp(fstart + 6)        # (n_slots, 3) complex
    n_checked = 0
    scale = float(np.max(np.abs(Z)))
    for s1 in range(len(atom_idx)):
        for s2 in range(s1 + 1, len(atom_idx)):
            if atom_idx[s1] != atom_idx[s2]:
                continue                             # Bloch relates SAME atom
            dn = np.asarray(uc_idx[s2], float) - np.asarray(uc_idx[s1], float)
            expected = Z[s1] * np.exp(2j * np.pi * float(q @ dn))
            assert np.allclose(Z[s2], expected, atol=1e-12 * scale), (s1, s2)
            n_checked += 1
    assert n_checked >= 6                            # 2 atoms x C(4,2) pairs
    print(f"  Bloch q.R = 2*pi*q_frac.n verified on {n_checked} cell pairs "
          f"at q={q.tolist()} (atol 1e-12 relative)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"\nALL {len(tests)} MOTION / RENDER-PATH CONFORMANCE TESTS PASS")
