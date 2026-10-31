"""Tests for the ph.x .dyn reader.

The oracle is unusually strong here and needs no fixture of our own: every
.dyn file carries BOTH the dynamical matrix and the frequencies ph.x derived
from it. Re-deriving the frequencies from the matrix therefore checks the
reader — units, mass convention, block indexing, hermiticity, the signed
square root — against the file's own answer, with nothing external assumed.

Real ph.x output is used where available (skips cleanly otherwise); the
synthetic case pins the format itself so the suite still has teeth on a
machine with no QE data at all.
"""

import glob
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phonon_core.qe_import import (  # noqa: E402
    RY_THZ, dyn_frequencies, read_qe_dyn, read_qe_dyn_set,
)
from phonon_core.qe_import.fc_file import AMU_RY, BOHR_A, RY_EV  # noqa: E402

#: Real ph.x output produced by the group. Not copied into the repo — it is
#: 19 files and the suite only reads it.
REAL_DIR = os.environ.get("PV_QE_DYN_FIXTURES", os.path.join(
    os.path.dirname(__file__), "fixtures", "qe", "real", "CsSnI3_epirelax"))


def real_files():
    if not os.path.isdir(REAL_DIR):
        return []
    return sorted(glob.glob(os.path.join(REAL_DIR, "CsSnI3.dyn*")))


# ------------------------------------------------------------------ constants

def test_ry_thz_is_derived_not_typed():
    """RY_THZ must follow from CODATA primitives, so a constants change
    propagates instead of silently disagreeing with the .fc side."""
    expect = RY_EV * 1.602176634e-19 / 6.62607015e-34 / 1e12
    assert RY_THZ == pytest.approx(expect, rel=1e-15)
    assert RY_THZ == pytest.approx(3289.842, rel=1e-6)   # the known value


# ------------------------------------------------------------------ synthetic

def write_dyn(path, *, omega_thz=2.5, mass_amu=50.0, alat_bohr=6.0, nat=1):
    """A one-atom .dyn whose D(q) is a known isotropic diagonal, so the
    frequency it must yield is fixed in closed form: nu = sqrt(k/m)*RY_THZ."""
    m_ry = mass_amu * AMU_RY
    k = (omega_thz / RY_THZ) ** 2 * m_ry            # Ry/bohr^2, isotropic
    lines = [
        "Dynamical matrix file", "synthetic",
        f"  1    {nat}   1 {alat_bohr}  0.0  0.0  0.0  0.0  0.0",
        f"           1  'C     '    {m_ry:.12f}",
        "    1    1      0.0000000000      0.0000000000      0.0000000000",
        "",
        "     Dynamical  Matrix in cartesian axes",
        "",
        "     q = (    0.000000000   0.000000000   0.000000000 ) ",
        "",
        "    1    1",
    ]
    for r in range(3):
        row = [0.0] * 6
        row[2 * r] = k
        lines.append("  " + "  ".join(f"{x:.8f}" for x in row))
    lines += ["", "     Diagonalizing the dynamical matrix", "",
              "     q = (    0.000000000   0.000000000   0.000000000 ) ", ""]
    for i in range(3):
        lines.append(f"     freq (    {i + 1}) =      {omega_thz:.6f} [THz] "
                     f"=      {omega_thz * 33.35641:.6f} [cm-1]")
        lines.append(" (  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000 ) ")
    open(path, "w").write("\n".join(lines) + "\n")
    return omega_thz


def test_synthetic_closed_form(tmp_path):
    """Reader + diagonalization reproduce an analytically constructed
    frequency: pins the Ry/bohr^2 -> THz chain and the mass convention."""
    p = str(tmp_path / "s.dyn")
    omega = write_dyn(p, omega_thz=2.5, mass_amu=50.0)
    d = read_qe_dyn(p)
    assert d.symbols == ["C"]
    assert d.masses[0] == pytest.approx(50.0, rel=1e-12)
    assert np.allclose(d.lattice, np.eye(3) * 6.0 * BOHR_A)
    f = dyn_frequencies(d)
    assert f.shape == (1, 3)
    assert np.allclose(f[0], omega, atol=1e-9)
    # and against the file's own printed values
    assert np.abs(np.sort(f[0]) - np.sort(d.file_frequencies[0])).max() < 1e-6


def test_malformed_dyn_is_loud(tmp_path):
    p = str(tmp_path / "bad.dyn")
    open(p, "w").write("Dynamical matrix file\nnothing else\n")
    with pytest.raises(ValueError):
        read_qe_dyn(p)
    p2 = str(tmp_path / "hdr_only.dyn")
    open(p2, "w").write("Dynamical matrix file\ntitle\n"
                        "  1    1   1 6.0  0.0  0.0  0.0  0.0  0.0\n"
                        "           1  'C     '    45572.2\n"
                        "    1    1      0.0  0.0  0.0\n")
    with pytest.raises(ValueError, match="no dynamical-matrix blocks"):
        read_qe_dyn(p2)


# ---------------------------------------------------------------- real ph.x

@pytest.mark.parametrize("path", real_files() or [None])
def test_real_dyn_self_consistency(path):
    """THE oracle: our diagonalization of D(q) must reproduce the frequencies
    ph.x printed in the same file, for every real file."""
    if path is None:
        pytest.skip("real ph.x .dyn output not present")
    try:
        d = read_qe_dyn(path)
    except ValueError as exc:
        if "no dynamical-matrix blocks" in str(exc) or "no 'ntyp nat" in str(exc):
            pytest.skip(f"{os.path.basename(path)}: grid-definition file (.dyn0)")
        raise
    assert d.file_frequencies is not None, "file printed no frequencies"
    f = dyn_frequencies(d)
    err = np.abs(np.sort(f[d.frequency_index], axis=1)
                 - np.sort(d.file_frequencies, axis=1)).max()
    assert err < 1e-5, f"{os.path.basename(path)}: |ours - file| = {err} THz"


def test_real_dyn_set():
    """Reading the whole set: q-blocks concatenate and the printed-frequency
    INDEX is remapped. ph.x diagonalizes only each star's representative, so
    a naive stack pairs the wrong rows — that error measured 1.19 THz where
    the per-file check was 1.7e-06, which is why frequency_index exists."""
    files = real_files()
    if not files:
        pytest.skip("real ph.x .dyn output not present")
    s = read_qe_dyn_set(files)
    assert len(s.qpoints_frac) > len(s.file_frequencies), (
        "this set should contain q-stars (more blocks than printed rows)")
    f = dyn_frequencies(s)
    err = np.abs(np.sort(f[s.frequency_index], axis=1)
                 - np.sort(s.file_frequencies, axis=1)).max()
    assert err < 1e-5, f"set self-check {err} THz"
    # every q-block is distinct: the star members are genuinely different q
    uniq = np.unique(np.round(s.qpoints_frac, 6), axis=0)
    assert len(uniq) == len(s.qpoints_frac)


def test_real_dyn_structure_and_polar_data():
    files = real_files()
    if not files:
        pytest.skip("real ph.x .dyn output not present")
    s = read_qe_dyn_set(files)
    # species resolved by mass (the .dyn labels here are correct, unlike the
    # .fc ones — so this also confirms the resolver leaves good labels alone)
    assert sorted(set(s.symbols)) == ["Cs", "I", "Sn"]
    assert s.symbols.count("I") == 3
    from phonopy.structure.atoms import atom_data
    std = {r[1]: r[3] for r in atom_data if r[1] and r[3]}
    for sym, m in zip(s.symbols, s.masses):
        assert abs(std[sym] - m) < 0.5

    assert s.dielectric is not None and s.born is not None
    assert np.abs(s.dielectric - s.dielectric.T).max() < 1e-6
    assert np.linalg.eigvalsh(s.dielectric).min() > 0
    # RAW DFPT Born charges: charge neutrality is NOT imposed here (q2r does
    # that when it writes the .fc, where the same sum is ~3e-08). Asserting
    # neutrality on .dyn would be asserting the wrong file's convention.
    assert s.born.shape == (len(s.symbols), 3, 3)
    assert np.isfinite(s.born).all()


def test_gamma_is_present_and_hermitian():
    files = real_files()
    if not files:
        pytest.skip("real ph.x .dyn output not present")
    s = read_qe_dyn_set(files)
    gi = int(np.argmin(np.linalg.norm(s.qpoints_frac, axis=1)))
    assert np.linalg.norm(s.qpoints_frac[gi]) < 1e-9, "no Gamma in the set"
    n = len(s.symbols)
    D = s.dynmat[gi].transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    assert np.abs(D - D.conj().T).max() < 1e-6, "D(Gamma) is not Hermitian"
    assert np.abs(D.imag).max() < 1e-8, "D(Gamma) should be real"
