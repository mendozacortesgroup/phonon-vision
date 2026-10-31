"""Tests for qe_import: the q2r .fc reader/converter.

Oracle design (see the module docstring for what is and is not pinned):
the fixtures are ``.fc`` files BUILT FROM THE FORMAT SPEC by the generators
below, for spring models whose dispersion is known in closed form. The chain
under test is  file → read_qe_fc → convert_qe_fc → A2 engine → frequencies,
compared against analytic ω(q) the reader never saw. Real q2r output slots
into the acceptance test at the bottom (fixture-guarded; instructions in
tests/fixtures/qe/README.md).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from phonon_core.qe_import import convert_qe_fc, read_qe_fc      # noqa: E402
from phonon_core.qe_import.fc_file import (                       # noqa: E402
    AMU_RY, BOHR_A, RY_BOHR2_TO_EV_A2,
)
from phonon_core.engine.dq_numpy import (                         # noqa: E402
    FREQ_FACTOR_THZ, evaluate_dq_np_batch, load_engine_np,
)
from engine_metric import LAM_TOL, doctrine_agreement             # noqa: E402

pytest.importorskip("phonopy", reason="convert_qe_fc writes the yaml via phonopy")


# --------------------------------------------------------- fixture generators

def _fc_lines(nr, blocks):
    """Emit the FC section: blocks = {(j1,j2,na1,na2): {(m1,m2,m3): eV/A^2}}.
    Values are converted to Ry/bohr^2 at write time (the file speaks QE
    units; the reader must undo them)."""
    out = []
    for (j1, j2, na1, na2), vals in blocks.items():
        out.append(f"{j1} {j2} {na1} {na2}")
        for m3 in range(1, nr[2] + 1):          # q2r order: m1 fastest
            for m2 in range(1, nr[1] + 1):
                for m1 in range(1, nr[0] + 1):
                    v = vals.get((m1, m2, m3), 0.0) / RY_BOHR2_TO_EV_A2
                    out.append(f"{m1} {m2} {m3} {v:.15e}")
    return out


def write_sc_toy(path, kx=1.0, ky=2.0, kz=3.5, mass=24.0, alat_bohr=5.0):
    """ibrav=1 simple-cubic, one atom, axial NN springs with DISTINCT spring
    constants per axis. Analytic: ω_α(q) = √(2k_α(1−cos 2πq_α)/m). The axis
    anisotropy is what makes a scrambled m-nesting or block order VISIBLE."""
    nr = (4, 4, 4)
    ks = {1: kx, 2: ky, 3: kz}
    blocks = {}
    for a in (1, 2, 3):
        vals = {(1, 1, 1): 2.0 * ks[a]}          # self term: exact ASR
        plus = [1, 1, 1]; plus[a - 1] = 2        # +1 cell along axis a
        minus = [1, 1, 1]; minus[a - 1] = nr[a - 1]   # −1 cell (periodic)
        vals[tuple(plus)] = -ks[a]
        vals[tuple(minus)] = -ks[a]
        blocks[(a, a, 1, 1)] = vals
        for b in (1, 2, 3):
            if b != a:
                blocks[(a, b, 1, 1)] = {}
    lines = [f"1 1 1 {alat_bohr} 0 0 0 0 0",
             f"1 'C' {mass * AMU_RY:.12f}",
             "1 1 0.0 0.0 0.0",
             "F",
             f"{nr[0]} {nr[1]} {nr[2]}"]
    lines += _fc_lines(nr, blocks)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return {"nr": nr, "ks": ks, "mass": mass}


def sc_analytic(ks, mass, qs):
    """Closed-form axial dispersion, sorted ascending per q, in THz."""
    out = []
    for q in qs:
        lam = [2.0 * ks[a + 1] * (1.0 - np.cos(2.0 * np.pi * q[a])) / mass
               for a in range(3)]
        out.append(sorted(FREQ_FACTOR_THZ * np.sqrt(np.maximum(lam, 0.0))))
    return np.array(out)


def write_two_sublattice_toy(path, kA=1.2, kB=2.8, mA=10.0, mB=40.0,
                             alat_bohr=6.0):
    """ibrav=0, two DECOUPLED sublattices with distinct masses and springs on
    a 2×2×2 grid — pins the explicit-lattice branch, tau parsing, the
    na-block order and the Ry-a.u. mass conversion (a wrong mass constant
    scales ω by √m and the doctrine catches it)."""
    nr = (2, 2, 2)
    at = np.array([[1.0, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 0.9]])
    tauY = 0.5 * at.sum(axis=0)
    blocks = {}
    for na, k in ((1, kA), (2, kB)):
        for a in (1, 2, 3):
            vals = {(1, 1, 1): 2.0 * k}
            fold = [1, 1, 1]; fold[a - 1] = 2    # ±1 coincide on a 2-grid
            vals[tuple(fold)] = -2.0 * k
            blocks[(a, a, na, na)] = vals
            for b in (1, 2, 3):
                if b != a:
                    blocks[(a, b, na, na)] = {}
        # zero inter-sublattice blocks, both orders
        other = 2 if na == 1 else 1
        for a in (1, 2, 3):
            for b in (1, 2, 3):
                blocks[(a, b, na, other)] = {}
    lines = [f"2 2 0 {alat_bohr} 0 0 0 0 0"]
    for row in at:
        lines.append(" ".join(f"{x:.12f}" for x in row))
    lines += [f"1 'Li' {mA * AMU_RY:.12f}",
              f"2 'Nb' {mB * AMU_RY:.12f}",
              "1 1 0.0 0.0 0.0",
              "2 2 " + " ".join(f"{x:.12f}" for x in tauY),
              "F",
              f"{nr[0]} {nr[1]} {nr[2]}"]
    lines += _fc_lines(nr, blocks)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return {"nr": nr, "kA": kA, "kB": kB, "mA": mA, "mB": mB}


def convert_and_freqs(fc_path, out, qs):
    os.makedirs(out, exist_ok=True)
    info = convert_qe_fc(fc_path, out)
    eng = load_engine_np(os.path.join(out, "phonopy.yaml"),
                         force_constants=os.path.join(out, "FORCE_CONSTANTS"))
    f, _ = evaluate_dq_np_batch(eng, qs)
    return f, info


def commensurate_qs(nr):
    qs = [(i / nr[0], j / nr[1], k / nr[2])
          for i in range(nr[0]) for j in range(nr[1]) for k in range(nr[2])]
    return np.array(qs)


# ------------------------------------------------------------ analytic oracle

def test_simple_cubic_matches_the_analytic_dispersion(tmp_path):
    """file → reader → converter → engine == closed form, at every
    supercell-commensurate q. Pins units, mass conversion, m-nesting, grid
    origin and periodic wrap — against physics the reader never saw."""
    fc = str(tmp_path / "sc.fc")
    spec = write_sc_toy(fc)
    qs = commensurate_qs(spec["nr"])
    f, info = convert_and_freqs(fc, str(tmp_path / "out"), qs)
    ana = sc_analytic(spec["ks"], spec["mass"], qs)
    d_nu, d_lam = doctrine_agreement(ana, np.sort(f, axis=1))
    assert d_nu < 1e-9, f"|dnu| vs analytic = {d_nu}"
    assert d_lam < LAM_TOL
    assert info["grid"] == [4, 4, 4]
    assert info["asr_drift"] < 1e-12               # the toy's exact self terms


def test_two_sublattices_pin_masses_blocks_and_ibrav0(tmp_path):
    fc = str(tmp_path / "two.fc")
    spec = write_two_sublattice_toy(fc)
    qs = commensurate_qs(spec["nr"])
    f, info = convert_and_freqs(fc, str(tmp_path / "out"), qs)
    ana = []
    for q in qs:
        lam = ([4.0 * spec["kA"] * (q[a] == 0.5) / spec["mA"] for a in range(3)]
               + [4.0 * spec["kB"] * (q[a] == 0.5) / spec["mB"] for a in range(3)])
        ana.append(sorted(FREQ_FACTOR_THZ * np.sqrt(lam)))
    d_nu, d_lam = doctrine_agreement(np.array(ana), np.sort(f, axis=1))
    assert d_nu < 1e-9 and d_lam < LAM_TOL
    assert info["n_unit_atoms"] == 2 and info["n_supercell_atoms"] == 16


def test_a1_agrees_on_the_converted_triple(tmp_path):
    """The written triple is legal phonopy input: A1 (phonopy-backed) and A2
    agree on it — guards the yaml/FC pairing, not just A2's reading."""
    from phonon_core.engine.dq_phonopy import evaluate_dq, load_engine
    fc = str(tmp_path / "sc.fc")
    spec = write_sc_toy(fc)
    out = str(tmp_path / "out")
    qs = commensurate_qs(spec["nr"])[:8]
    f2, _ = convert_and_freqs(fc, out, qs)
    ph = load_engine(os.path.join(out, "phonopy.yaml"),
                     force_constants=os.path.join(out, "FORCE_CONSTANTS"))
    f1 = np.array([evaluate_dq(ph, q)[0] for q in qs])
    d_nu, d_lam = doctrine_agreement(f1, f2)
    assert d_nu < 1e-8 and d_lam < LAM_TOL


# --------------------------------------------------------- reading discipline

def test_reader_honors_explicit_indices_not_line_order(tmp_path):
    """Swap two complete m-lines (indices AND values): a reader driven by the
    printed indices is unchanged bitwise; one driven by line order breaks."""
    fc = str(tmp_path / "sc.fc")
    spec = write_sc_toy(fc)
    qs = commensurate_qs(spec["nr"])[:6]
    f_ref, _ = convert_and_freqs(fc, str(tmp_path / "a"), qs)

    lines = open(fc).read().splitlines()
    body = [i for i, ln in enumerate(lines) if len(ln.split()) == 4
            and ln.split()[3] not in ("1", "2", "3", "4")][:0]  # noqa: F841
    # find two value lines inside the first block and swap them wholesale
    idx = [i for i, ln in enumerate(lines)
           if ln.split()[:3] == ["2", "1", "1"] or ln.split()[:3] == ["1", "2", "1"]]
    i1, i2 = idx[0], idx[1]
    lines[i1], lines[i2] = lines[i2], lines[i1]
    fc2 = str(tmp_path / "swapped.fc")
    open(fc2, "w").write("\n".join(lines) + "\n")
    f_swapped, _ = convert_and_freqs(fc2, str(tmp_path / "b"), qs)
    assert np.array_equal(f_ref, f_swapped)

    # teeth: swapping VALUES ONLY (indices kept) moves a spring to the wrong
    # R and the dispersion must change
    t1, t2 = lines[i1].split(), lines[i2].split()
    t1[3], t2[3] = t2[3], t1[3]
    lines[i1], lines[i2] = " ".join(t1), " ".join(t2)
    fc3 = str(tmp_path / "corrupt.fc")
    open(fc3, "w").write("\n".join(lines) + "\n")
    f_bad, _ = convert_and_freqs(fc3, str(tmp_path / "c"), qs)
    assert np.abs(f_bad - f_ref).max() > 0.1


def test_malformed_files_are_loud(tmp_path):
    fc = str(tmp_path / "sc.fc")
    write_sc_toy(fc)
    text = open(fc).read()
    trunc = str(tmp_path / "trunc.fc")
    open(trunc, "w").write("\n".join(text.splitlines()[:-10]))
    with pytest.raises(ValueError, match="truncated|missing"):
        read_qe_fc(trunc)
    with pytest.raises(NotImplementedError, match="ibrav=7"):
        bad = text.replace("1 1 1 5.0", "1 1 7 5.0", 1)
        p = str(tmp_path / "ibrav7.fc"); open(p, "w").write(bad)
        read_qe_fc(p)


def test_units_round_trip_through_the_header(tmp_path):
    """The reader must undo exactly the units the generator applied: masses
    Ry-a.u. → amu, tau alat → Å, FC Ry/bohr² → eV/Å²."""
    fc = str(tmp_path / "sc.fc")
    write_sc_toy(fc, mass=24.0, alat_bohr=5.0)
    qe = read_qe_fc(fc)
    assert qe.masses[0] == pytest.approx(24.0, rel=1e-12)
    assert np.allclose(qe.lattice, np.eye(3) * 5.0 * BOHR_A)
    # the self-term block is +2k_x = 2.0 eV/Å² in the xx component
    assert qe.fc[0, 0, 0, 0, 0, 0, 0] == pytest.approx(2.0, rel=1e-12)


# ------------------------------------------------------------------ polar gate

def test_polar_files_are_refused_with_the_reason(tmp_path):
    fc = str(tmp_path / "polar.fc")
    write_sc_toy(fc)
    lines = open(fc).read().splitlines()
    i = lines.index("F")
    # The REAL q2r layout, as proven by the CsSnI3 fixtures: the flag line may
    # carry a trailing number, the dielectric is 3 rows, and each atom's Born
    # tensor is preceded by its own INDEX line. An earlier version of this
    # generator omitted those index lines — a format assumption real data
    # corrected.
    eps = ["3.0 0.0 0.0", "0.0 3.0 0.0", "0.0 0.0 3.0"]
    born = ["    1", "0.0 0.0 0.0", "0.0 0.0 0.0", "0.0 0.0 0.0"]
    lines[i:i + 1] = ["  T   1.234567890"] + eps + born
    open(fc, "w").write("\n".join(lines) + "\n")
    qe = read_qe_fc(fc)
    assert qe.polar and qe.dielectric is not None
    with pytest.raises(NotImplementedError, match="REFUSED"):
        convert_qe_fc(fc, str(tmp_path))


# -------------------------------------------------- real-data acceptance hook

def _real_cases():
    real = os.path.join(os.path.dirname(__file__), "fixtures", "qe", "real")
    if not os.path.isdir(real):
        return []
    return [(d, os.path.join(real, d)) for d in sorted(os.listdir(real))
            if os.path.isdir(os.path.join(real, d))]


def _case_fc(cdir):
    fcs = [f for f in os.listdir(cdir) if f.endswith(".fc")]
    assert len(fcs) == 1, f"{cdir}: expected exactly one .fc, found {fcs}"
    return os.path.join(cdir, fcs[0])


@pytest.mark.parametrize("name,cdir", _real_cases() or [("none", None)])
def test_real_q2r_reader(name, cdir):
    """READER validation on real q2r output — the half that does not depend
    on NAC. Every assertion here is physics or format structure the reader
    could not have invented."""
    if cdir is None:
        pytest.skip("no real q2r fixtures yet (see fixtures/qe/README.md)")
    qe = read_qe_fc(_case_fc(cdir))

    # structure: a real file must produce a sane cell and a full FC block set
    assert len(qe.symbols) == len(qe.masses) == len(qe.positions)
    assert np.linalg.det(qe.lattice) > 0, "left-handed or degenerate lattice"
    assert all(n >= 1 for n in qe.nr)
    assert qe.fc.shape == (len(qe.symbols), len(qe.symbols),
                           qe.nr[0], qe.nr[1], qe.nr[2], 3, 3)
    assert np.isfinite(qe.fc).all()

    # species resolution: every symbol must be a real element whose standard
    # mass matches the file's mass (this is what catches the mislabelled
    # types real files carry)
    from phonopy.structure.atoms import atom_data
    std = {row[1]: row[3] for row in atom_data if row[1] and row[3]}
    for sym, m in zip(qe.symbols, qe.masses):
        assert sym in std, f"unresolved species {sym!r}"
        assert abs(std[sym] - m) < 0.5, f"{sym}: file mass {m} vs {std[sym]}"

    if qe.polar:
        # CHARGE NEUTRALITY: sum_a Z*_a = 0 is required by physics and is an
        # independent check that the polar block (with its per-atom index
        # lines) was parsed correctly. A mis-parse scrambles the tensors and
        # this sum stops vanishing.
        assert np.abs(qe.born.sum(axis=0)).max() < 1e-6, (
            f"sum of Born charges {np.abs(qe.born.sum(axis=0)).max():.2e} "
            "!= 0 — polar block mis-parsed")
        # dielectric tensor: symmetric and positive definite
        assert np.abs(qe.dielectric - qe.dielectric.T).max() < 1e-6
        assert np.linalg.eigvalsh(qe.dielectric).min() > 0


@pytest.mark.parametrize("name,cdir", _real_cases() or [("none", None)])
def test_real_q2r_frequency_oracle(name, cdir):
    """FREQUENCY oracle against matdyn — the full chain.

    Polar cases are skipped BY NAME, not silently: matdyn applies the
    non-analytic correction from the Born/dielectric data, and convert_qe_fc
    refuses polar input until the QE-to-phonopy NAC unit constant is pinned.
    Comparing anyway would measure the missing LO-TO splitting rather than
    the reader. These fixtures become the oracle that pins NAC the moment it
    lands — that is what they are staged for.
    """
    if cdir is None:
        pytest.skip("no real q2r fixtures yet (see fixtures/qe/README.md)")
    fc_path = _case_fc(cdir)
    qe = read_qe_fc(fc_path)
    if qe.polar:
        pytest.skip(f"{name}: polar (Born/dielectric present) — frequency "
                    "comparison pending NAC; the reader half is covered by "
                    "test_real_q2r_reader")

    ref = os.path.join(cdir, "reference_freqs.csv")
    assert os.path.exists(ref), f"{name}: reference_freqs.csv missing"
    rows = np.loadtxt(ref, delimiter=",", comments="#")
    qs, f_ref = rows[:, :3], rows[:, 3:]

    import tempfile
    with tempfile.TemporaryDirectory() as out:
        convert_qe_fc(fc_path, out)
        eng = load_engine_np(os.path.join(out, "phonopy.yaml"),
                             force_constants=os.path.join(out, "FORCE_CONSTANTS"))
        f, _ = evaluate_dq_np_batch(eng, qs)
    d_nu, d_lam = doctrine_agreement(f_ref, np.sort(f, axis=1))
    assert d_nu < 1e-3, f"{name}: |dnu| vs matdyn = {d_nu}"
