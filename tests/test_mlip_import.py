"""Tests for mlip_import.convert_hessian.

Oracle design: phonopy's full-format FORCE_CONSTANTS *is* the supercell
Hessian, so a phonopy-built toy provides a self-contained round trip — treat
the toy's (symmetrized) FC as "the MLIP Hessian", convert, and the A2 engine
on the converted triple must reproduce the A2 engine on the original triple
to the metric doctrine. The negative controls carry the real teeth: a
swapped atom pair must show the classic q-dependent signature (invisible at
Γ, large at generic q), and asymmetric input must be refused, not absorbed.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from phonon_core.mlip_import import convert_hessian            # noqa: E402
from phonon_core.engine.dq_numpy import (                      # noqa: E402
    evaluate_dq_np_batch, load_engine_np,
)
from engine_metric import LAM_TOL, doctrine_agreement          # noqa: E402

phonopy = pytest.importorskip("phonopy", reason="toy builder needs phonopy")


# ------------------------------------------------------------- construction

def build_toy(dirpath, seed=42):
    """Rocksalt-like 2x2x2 toy with a SYMMETRIC random Hessian, saved as a
    reference phonopy triple. Returns (yaml_path, hessian (N,N,3,3))."""
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.file_IO import write_FORCE_CONSTANTS

    cell = PhonopyAtoms(
        symbols=["Na", "Cl"],
        cell=np.array([[0.0, 2.85, 2.85],
                       [2.85, 0.0, 2.85],
                       [2.85, 2.85, 0.0]]),
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        masses=[22.99, 35.45])
    ph = Phonopy(cell, supercell_matrix=np.diag([2, 2, 2]),
                 primitive_matrix=np.eye(3), log_level=0)
    n_s = len(ph.supercell)
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-1.0, 1.0, (n_s, n_s, 3, 3))
    hess = 0.5 * (raw + raw.transpose(1, 0, 3, 2))      # a true Hessian is symmetric

    # TRANSLATION-AVERAGE the random couplings. A Hessian from a periodic
    # supercell calculation satisfies Φ[T i, T j] = Φ[i, j] for supercell
    # lattice translations T EXACTLY (up to solver noise) — the energy model
    # is translation invariant. A raw random matrix does not, and that is
    # not a harmless idealization: the engine reads only the PRIMITIVE rows
    # and hermitizes, so for a non-TI input the hermitization term needs
    # partial group-column sums that no sum-rule correction can supply, and
    # Γ acoustics never close (measured: 0.21 eV/Å² null-vector leakage on
    # a projected-but-non-TI toy). Averaging over the 8 translations makes
    # the toy the thing a real MLIP Hessian actually is.
    spos = ph.supercell.scaled_positions                # (16, 3) supercell coords
    tau = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    kind = np.repeat([0, 1], 8)
    cells = np.rint(spos * 2 - tau[kind]).astype(int)   # unit-cell index per atom
    assert np.allclose(spos * 2 - tau[kind], cells, atol=1e-10)
    def _at(k, c):
        hit = np.where((kind == k) & (cells == np.mod(c, 2)).all(axis=1))[0]
        assert hit.size == 1
        return hit[0]
    acc = np.zeros_like(hess)
    for L in np.ndindex(2, 2, 2):
        perm = np.array([_at(kind[a2], cells[a2] + np.array(L))
                         for a2 in range(n_s)])
        acc += hess[np.ix_(perm, perm)]
    hess = acc / 8.0
    ph.force_constants = hess
    ph.save(os.path.join(dirpath, "phonopy.yaml"),
            settings={"force_constants": False})
    write_FORCE_CONSTANTS(hess, filename=os.path.join(dirpath, "FORCE_CONSTANTS"))
    return os.path.join(dirpath, "phonopy.yaml"), hess


def seeded_qs(n=24, seed=7):
    rng = np.random.default_rng(seed)
    qs = rng.uniform(-0.5, 0.5, size=(n, 3))
    qs[0] = 0.0                                          # Γ always included
    return qs


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    ref = tmp_path_factory.mktemp("mlip_ref")
    yaml_path, hess = build_toy(str(ref))
    eng = load_engine_np(yaml_path,
                         force_constants=os.path.join(str(ref), "FORCE_CONSTANTS"))
    qs = seeded_qs()
    f_ref, _ = evaluate_dq_np_batch(eng, qs)
    return {"dir": str(ref), "yaml": yaml_path, "hess": hess,
            "qs": qs, "f_ref": f_ref}


def convert_and_eval(toy_, out, hess, **kw):
    os.makedirs(out, exist_ok=True)
    info = convert_hessian(hess, toy_["yaml"], out, **kw)
    eng = load_engine_np(os.path.join(out, "phonopy.yaml"),
                         force_constants=os.path.join(out, "FORCE_CONSTANTS"))
    f, _ = evaluate_dq_np_batch(eng, toy_["qs"])
    return f, info


# ------------------------------------------------------------------ round trip

def test_round_trip_reproduces_the_reference(toy, tmp_path):
    f, info = convert_and_eval(toy, str(tmp_path / "rt"), toy["hess"])
    d_nu, d_lam = doctrine_agreement(toy["f_ref"], f)
    assert d_nu < 1e-10, f"|dnu| {d_nu}"
    assert d_lam < LAM_TOL
    assert info["n_supercell_atoms"] == 16
    assert info["max_asymmetry"] == 0.0
    assert not info["asr_enforced"]


def test_flat_3n_form_is_identical(toy, tmp_path):
    n = toy["hess"].shape[0]
    flat = toy["hess"].transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    f_flat, _ = convert_and_eval(toy, str(tmp_path / "a"), flat)
    f_4d, _ = convert_and_eval(toy, str(tmp_path / "b"), toy["hess"])
    assert np.array_equal(f_flat, f_4d), "the two input layouts must be one code path"


# ------------------------------------------------------------ negative controls

def test_swapped_pair_shows_the_q_dependent_signature(toy, tmp_path):
    """THE atom-order trap: one swapped supercell pair must be (a) nearly
    invisible at Γ and (b) large at generic q — and supplying the same
    permutation via atom_order must restore exact agreement."""
    n = toy["hess"].shape[0]
    # The pair must be two images of the SAME primitive atom (sp2p equal):
    # at Γ each primitive column is a plain group SUM, so a within-group swap
    # commutes and is invisible there — exactly the trap's signature. A
    # cross-primitive swap (a first version used Na image 3 ↔ Cl image 11)
    # changes the group sums and is visible at Γ too, which tests nothing.
    perm = np.arange(n); perm[[3, 5]] = perm[[5, 3]]   # two Na images
    scrambled = toy["hess"][np.ix_(perm, perm)]

    f_bad, _ = convert_and_eval(toy, str(tmp_path / "bad"), scrambled)
    gamma_err = np.abs(f_bad[0] - toy["f_ref"][0]).max()
    generic_err = np.abs(f_bad[1:] - toy["f_ref"][1:]).max()
    assert gamma_err < 1e-6, "a pure relabelling is invisible at Gamma"
    assert generic_err > 0.05, (
        f"swapped pair must bite at generic q (got {generic_err})")

    # atom_order[k] = where atom k sits in the scrambled Hessian = perm
    f_fixed, info = convert_and_eval(toy, str(tmp_path / "fix"), scrambled,
                                     atom_order=perm)
    d_nu, d_lam = doctrine_agreement(toy["f_ref"], f_fixed)
    assert d_nu < 1e-10 and d_lam < LAM_TOL
    assert info["atom_order_applied"]


def test_asymmetric_input_is_refused_not_absorbed(toy, tmp_path):
    noisy = toy["hess"].copy()
    noisy[0, 1, 0, 1] += 5e-3                      # breaks H == H^T pairing
    with pytest.raises(ValueError, match="asymmetry"):
        convert_hessian(noisy, toy["yaml"], str(tmp_path))
    # below tolerance: symmetrized and REPORTED, not hidden
    mild = toy["hess"].copy(); mild[0, 1, 0, 1] += 5e-7
    os.makedirs(tmp_path / "ok", exist_ok=True)
    info = convert_hessian(mild, toy["yaml"], str(tmp_path / "ok"))
    assert 0 < info["max_asymmetry"] < 1e-5


def test_bad_inputs_are_loud(toy, tmp_path):
    with pytest.raises(ValueError, match="units"):
        convert_hessian(toy["hess"], toy["yaml"], str(tmp_path), units="Ry/bohr^2")
    with pytest.raises(ValueError, match="outdir"):
        convert_hessian(toy["hess"], toy["yaml"], str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="shape"):
        convert_hessian(toy["hess"][:-1, :-1], toy["yaml"], str(tmp_path))
    bad_order = np.zeros(16, dtype=int)
    with pytest.raises(ValueError, match="permutation"):
        convert_hessian(toy["hess"], toy["yaml"], str(tmp_path),
                        atom_order=bad_order)


# ------------------------------------------------------------------------- ASR

def test_asr_enforcement_zeroes_gamma_acoustics(toy, tmp_path):
    """A random Hessian violates the sum rule badly; the diagonal correction
    must produce EXACT translation null modes at Γ (the old −0.129 THz
    bundled-payload lesson: ASR drift reads as fake instability)."""
    f_raw, info_raw = convert_and_eval(toy, str(tmp_path / "raw"), toy["hess"])
    assert info_raw["asr_drift_before"] > 0.1        # the toy genuinely drifts
    acoustic_raw = np.sort(np.abs(f_raw[0]))[:3]

    f_asr, info = convert_and_eval(toy, str(tmp_path / "asr"), toy["hess"],
                                   enforce_asr=True)
    assert info["asr_enforced"]
    assert info["asr_drift_after"] < 1e-12
    acoustic = np.sort(np.abs(f_asr[0]))[:3]
    assert acoustic.max() < 1e-5, f"Γ acoustics after ASR: {acoustic}"
    # and the correction genuinely changed something
    assert acoustic_raw.max() > 0.05


# ------------------------------------------------------------------ real data

def test_real_force_constants_round_trip():
    """Tier-3: treat a REAL dataset's supercell force constants as an MLIP
    Hessian and require the converted triple to reproduce the original
    engine exactly. 192 atoms, real geometry — the size and structure a
    genuine MLIP Hessian has, which the 16-atom toy cannot represent.

    The dataset ships COMPACT force constants (3, 192, 3, 3) while a Hessian
    is full-format by nature, so phonopy expands it here rather than a
    2.6 MB binary being stored: the data stays single-sourced in
    Phonopy-examples, and the expansion itself is checked (pair symmetry)
    before it is used as input.
    """
    base = os.path.join(os.path.dirname(__file__), "..", "..",
                        "Phonopy-examples", "fatbands_SnSe2", "data_task")
    yaml_path = os.path.join(base, "phonopy.yaml")
    fc_path = os.path.join(base, "FORCE_CONSTANTS")
    if not (os.path.exists(yaml_path) and os.path.exists(fc_path)):
        pytest.skip("real SnSe2 data_task tree not present")

    from phonopy import load as ph_load
    from phonopy.harmonic.force_constants import compact_fc_to_full_fc
    ph = ph_load(yaml_path, force_constants_filename=fc_path, log_level=0)
    fc = ph.force_constants
    if fc.shape[0] != fc.shape[1]:
        fc = compact_fc_to_full_fc(ph.primitive, fc)
    assert fc.shape[0] == fc.shape[1] == 192
    assert np.abs(fc - fc.transpose(1, 0, 3, 2)).max() < 1e-10, (
        "expanded FC is not pair-symmetric — cannot stand in for a Hessian")

    import tempfile
    eng_ref = load_engine_np(yaml_path, force_constants=fc_path)
    qs = seeded_qs(n=12, seed=13)
    f_ref, _ = evaluate_dq_np_batch(eng_ref, qs)
    with tempfile.TemporaryDirectory() as out:
        info = convert_hessian(fc, yaml_path, out)
        eng = load_engine_np(os.path.join(out, "phonopy.yaml"),
                             force_constants=os.path.join(out, "FORCE_CONSTANTS"))
        f, _ = evaluate_dq_np_batch(eng, qs)
    assert info["n_supercell_atoms"] == 192
    d_nu, d_lam = doctrine_agreement(f_ref, f)
    assert d_nu < 1e-9, f"|dnu| on real data = {d_nu}"
    assert d_lam < LAM_TOL
