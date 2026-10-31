"""Reader/converter for Quantum ESPRESSO q2r ``.fc`` force-constant files.

The format (from q2r.f90 / matdyn.f90; parsed token-wise, never by column)
---------------------------------------------------------------------------
::

    ntyp nat ibrav celldm(1..6)
    [ibrav==0 only: 3 lines of lattice vectors, alat units]
    per type:  nt  'symbol'  mass          (mass in RYDBERG ATOMIC UNITS)
    per atom:  na  ityp  tau_x tau_y tau_z (Cartesian, alat units)
    T | F                                  (dielectric/Born block present?)
    [T only: 3x3 dielectric, then per atom a 3x3 Born tensor]
    nr1 nr2 nr3
    for j1, j2, na1, na2:                  (j slowest, na2 fastest)
        j1 j2 na1 na2
        m1 m2 m3  fc(m1,m2,m3)             (nr1*nr2*nr3 lines, m1 FASTEST)

FC values are Ry/bohr². The supercell is nr1 x nr2 x nr3 unit cells, and
``fc(j1,j2,na1,na2,m)`` couples atom na1 in the home cell with atom na2 in
the cell at lattice vector ``(m1-1, m2-1, m3-1)`` (grid indices are 1-based,
periodic).

Unit conversion (derived from CODATA 2018 primitives IN CODE — the
never-hand-type-a-derived-constant rule):

* ``Ry/bohr² → eV/Å²``: RY_EV / BOHR_A².
* header masses → amu: divide by AMU_RY = (m_u/m_e)/2 — Rydberg atomic
  units set mₑ = 1/2.

Convention pins: what the analytic oracle DOES and DOES NOT settle
------------------------------------------------------------------
The in-suite fixtures are ``.fc`` files BUILT FROM THIS FORMAT SPEC for
spring models whose dispersion is known in closed form, and the test chain
runs file → this reader → phonopy triple → A2 engine → frequencies vs the
analytic ω(q). That pins, against physics the reader never saw:

* the Ry/bohr² and mass unit conversions (wrong constant = wrong THz);
* the m-index NESTING (m1 fastest) and the (j1,j2,na1,na2) block order —
  pinned by anisotropic axis springs and by two decoupled sublattices with
  distinct masses, either of which detects a scrambled layout;
* the grid-index origin and periodic wrap.

It CANNOT settle, and this module therefore treats as **PENDING REAL
q2r OUTPUT** (fixtures + acceptance protocol in
``tests/fixtures/qe/README.md``):

* header quirks of specific QE versions (spacing variants are handled by
  token-wise parsing, but only real files prove it);
* the Fourier-sign convention. Time reversal makes ω(q) = ω(−q) in ANY
  crystal, so NO frequency oracle can pin the sign — it shows up only as
  e ↔ e*, which is invisible to frequencies and |e|² displays and matters
  only to phase-sensitive observables. Recorded, not guessed.
* **polar materials.** A ``T`` dielectric/Born block is parsed but
  conversion is REFUSED: activating NAC requires the QE↔phonopy NAC unit
  constant to be pinned against real matdyn output first — the exact
  failure mode of the pheasy-NAC lesson (two constants both named
  "factor"; the wrong one costs 10⁻² – 10⁻¹ THz and only an oracle
  catches it).

Requires phonopy (the ``engine`` extra) to write a guaranteed-valid
``phonopy.yaml`` — the same sanctioned pattern as ``mp_import.pheasy``.
"""

from __future__ import annotations

import os

import numpy as np

__all__ = ["read_qe_fc", "convert_qe_fc",
           "RY_EV", "BOHR_A", "AMU_RY", "RY_BOHR2_TO_EV_A2"]

# CODATA 2018 primitives; everything else is DERIVED.
RY_EV = 13.605693122994          # 1 Ry in eV
BOHR_A = 0.529177210903          # 1 bohr in Angstrom
_MU_OVER_ME = 1822.888486209     # m_u / m_e
AMU_RY = _MU_OVER_ME / 2.0       # 1 amu in Rydberg-a.u. mass (m_e = 1/2)
RY_BOHR2_TO_EV_A2 = RY_EV / BOHR_A ** 2


class QEForceConstants:
    """Parsed .fc content, converted to phonon-vision units (Å, amu, eV/Å²)."""

    def __init__(self, *, lattice, symbols, masses, positions, nr, fc,
                 polar, dielectric=None, born=None, symbol_sources=None):
        self.lattice = lattice            # (3,3) Å, rows a1,a2,a3
        self.symbols = symbols            # (nat,) str
        self.masses = masses              # (nat,) amu
        self.positions = positions        # (nat,3) Å Cartesian
        self.nr = nr                      # (3,) grid
        self.fc = fc                      # (nat,nat,nr1,nr2,nr3,3,3) eV/Å²
        self.polar = polar
        self.dielectric = dielectric
        self.born = born
        #: {type index: "label" | "blank-label" | "label(X)-vs-mass"} — how
        #: each species symbol was determined (real files mislabel; see
        #: _resolve_symbol)
        self.symbol_sources = symbol_sources or {}


def _latgen(ibrav, celldm):
    """Direct lattice for the ibrav values supported, in alat units.

    Only the unambiguous, tested cases are implemented; everything else is a
    loud NotImplementedError naming the workaround (ibrav=0 carries the
    lattice explicitly, and q2r preserves the pw.x ibrav, so re-running q2r
    is not needed — matdyn.f90's latgen covers ~15 variants and typing them
    untested would violate the derived-data rule in spirit)."""
    if ibrav == 1:                       # simple cubic
        return np.eye(3)
    if ibrav == 2:                       # fcc, QE convention
        return 0.5 * np.array([[-1.0, 0.0, 1.0],
                               [0.0, 1.0, 1.0],
                               [-1.0, 1.0, 0.0]])
    if ibrav == 3:                       # bcc, QE convention
        return 0.5 * np.array([[1.0, 1.0, 1.0],
                               [-1.0, 1.0, 1.0],
                               [-1.0, -1.0, 1.0]])
    if ibrav == 4:                       # hexagonal; celldm(3) = c/a
        c_a = celldm[2]
        if not c_a > 0:
            raise ValueError("ibrav=4 needs celldm(3) = c/a > 0")
        return np.array([[1.0, 0.0, 0.0],
                         [-0.5, np.sqrt(3.0) / 2.0, 0.0],
                         [0.0, 0.0, c_a]])
    if ibrav == 6:                       # simple tetragonal; celldm(3) = c/a
        c_a = celldm[2]
        if not c_a > 0:
            raise ValueError("ibrav=6 needs celldm(3) = c/a > 0")
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, c_a]])
    raise NotImplementedError(
        f"ibrav={ibrav} is not implemented (supported: 0,1,2,3,4,6). "
        "Real q2r output for other lattices is welcome as an acceptance "
        "fixture; see tests/fixtures/qe/README.md")


def _resolve_symbol(label, mass_amu):
    """Species symbol, resolving label-vs-mass disagreement in favour of MASS.

    Real q2r output cannot be trusted on labels: a CsSnI3 file on record
    carries types ``'Cs '`` (132.905 amu, correct), ``'   '`` (blank label,
    118.710 amu = Sn) and ``'Sn '`` (126.904 amu = iodine). The mass is
    physically load-bearing — it enters D(q) directly — while the label only
    names the species downstream, so a disagreement is resolved by mass and
    the substitution is reported in ``symbol_source``.
    """
    from phonon_core.radii import COVALENT_RADIUS_ANGSTROM  # element name set

    try:
        from phonopy.structure.atoms import atom_data
        table = {row[1]: row[3] for row in atom_data
                 if row[1] and row[3] is not None}
    except Exception:                                       # pragma: no cover
        table = {}

    clean = (label or "").strip().strip("'\"").strip()
    known = clean in table
    if known and abs(table[clean] - mass_amu) <= 0.5:
        return clean, "label"
    # label unusable or contradicted: pick the element whose standard mass
    # is closest, and only accept a convincing match
    if table:
        best = min(table, key=lambda el: abs(table[el] - mass_amu))
        if abs(table[best] - mass_amu) <= 0.5:
            reason = "blank-label" if not clean else f"label({clean})-vs-mass"
            return best, reason
    if known:
        return clean, "label-unverified"
    raise ValueError(
        f"cannot resolve species: label {label!r}, mass {mass_amu:.4f} amu "
        "matches no element within 0.5 amu")


def read_qe_fc(path):
    """Parse a q2r ``.fc`` file into :class:`QEForceConstants` (Å/amu/eV/Å²).

    The header is parsed LINE-oriented, not from a flat token stream: real
    q2r files carry quoted species labels containing spaces (``'Cs '``, and
    even an all-blank ``'   '``), which a whitespace split shreds. The FC
    section stays token-oriented — it is regular and free-format.
    """
    with open(path) as fh:
        raw_lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    li = 0

    def line():
        nonlocal li
        if li >= len(raw_lines):
            raise ValueError(f"{path}: truncated file at line {li}")
        out = raw_lines[li]
        li += 1
        return out

    def nums(n, src=None):
        vals = (src if src is not None else line()).split()
        if len(vals) < n:
            raise ValueError(f"{path}: expected {n} numbers, got {vals}")
        return [float(x) for x in vals[:n]]

    head = line().split()
    ntyp, nat, ibrav = (int(x) for x in head[:3])
    celldm = np.array([float(x) for x in head[3:9]])
    alat_A = celldm[0] * BOHR_A
    if not alat_A > 0:
        raise ValueError(f"{path}: celldm(1) = {celldm[0]} must be > 0")

    if ibrav == 0:
        at = np.array([nums(3) for _ in range(3)])
    else:
        at = _latgen(ibrav, celldm)
    lattice = at * alat_A                                  # rows, Å

    # species: label may be quoted WITH spaces (even all-blank), so take the
    # quoted field by regex and the mass as the last token on the line
    import re
    type_symbols, type_masses, symbol_sources = {}, {}, {}
    for _ in range(ntyp):
        ln = line()
        m = re.search(r"'([^']*)'|\"([^\"]*)\"", ln)
        label = (m.group(1) if m and m.group(1) is not None
                 else (m.group(2) if m else ""))
        rest = (ln[:m.start()] + " " + ln[m.end():]) if m else ln
        toks = rest.split()
        nt = int(toks[0])
        mass = float(toks[-1]) / AMU_RY                     # Ry-a.u. → amu
        sym, src = _resolve_symbol(label, mass)
        type_symbols[nt], type_masses[nt], symbol_sources[nt] = sym, mass, src

    symbols, masses, positions = [], [], []
    for _ in range(nat):
        toks = line().split()
        ityp = int(toks[1])
        tau = np.array([float(x) for x in toks[2:5]])
        symbols.append(type_symbols[ityp])
        masses.append(type_masses[ityp])
        positions.append(tau * alat_A)                     # alat → Å

    # polar flag: FIRST token of its line. Real files carry a trailing number
    # on the same line (observed: " T   3.4246150776654605"), so anything
    # beyond the flag is deliberately ignored rather than consumed as data.
    ptoks = line().split()
    polar = ptoks[0] in ("T", "t", ".true.", ".TRUE.")
    dielectric = born = None
    if polar:
        dielectric = np.array([nums(3) for _ in range(3)])
        born = np.empty((nat, 3, 3))
        for _ in range(nat):
            na = int(line().split()[0])                    # per-atom INDEX line
            if not 1 <= na <= nat:
                raise ValueError(f"{path}: bad Born atom index {na}")
            born[na - 1] = np.array([nums(3) for _ in range(3)])

    nr = tuple(int(x) for x in line().split()[:3])
    n_cells = nr[0] * nr[1] * nr[2]

    fc = np.empty((nat, nat, nr[0], nr[1], nr[2], 3, 3))
    seen = np.zeros((3, 3, nat, nat), dtype=bool)
    for _ in range(9 * nat * nat):
        j1, j2, na1, na2 = (int(x) for x in line().split()[:4])
        if seen[j1 - 1, j2 - 1, na1 - 1, na2 - 1]:
            raise ValueError(f"{path}: duplicate block {j1} {j2} {na1} {na2}")
        seen[j1 - 1, j2 - 1, na1 - 1, na2 - 1] = True
        for _ in range(n_cells):
            t = line().split()
            m1, m2, m3 = (int(x) for x in t[:3])
            fc[na1 - 1, na2 - 1, m1 - 1, m2 - 1, m3 - 1, j1 - 1, j2 - 1] = \
                float(t[3]) * RY_BOHR2_TO_EV_A2
    if li != len(raw_lines):
        raise ValueError(f"{path}: {len(raw_lines) - li} trailing lines — "
                         "block count mismatch")
    if not seen.all():
        raise ValueError(f"{path}: missing FC blocks")

    return QEForceConstants(lattice=lattice, symbols=symbols,
                            masses=np.array(masses), positions=np.array(positions),
                            nr=nr, fc=fc, polar=polar,
                            dielectric=dielectric, born=born,
                            symbol_sources=symbol_sources)


def convert_qe_fc(fc_path, outdir):
    """Write the phonopy triple for a q2r ``.fc`` file.

    Refuses polar files (see module docstring — the NAC unit constant is
    pending real-data pinning). Returns an info dict.
    """
    qe = read_qe_fc(fc_path)
    if qe.polar:
        raise NotImplementedError(
            f"{fc_path} carries a dielectric/Born block: polar conversion "
            "is REFUSED until the QE-to-phonopy NAC unit constant is pinned "
            "against real matdyn output (the pheasy-NAC lesson: a plausible "
            "wrong constant costs 1e-2 to 1e-1 THz and only an oracle "
            "catches it). Non-polar conversion of the same file would be "
            "silently wrong physics, so there is no override flag.")
    if not os.path.isdir(outdir):
        raise ValueError(f"outdir does not exist: {outdir}")

    try:
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
        from phonopy.file_IO import write_FORCE_CONSTANTS
    except ImportError as exc:                             # pragma: no cover
        raise ImportError(
            "qe_import needs phonopy to write a guaranteed-valid "
            "phonopy.yaml (install the 'engine' extra)") from exc

    nat, nr = len(qe.symbols), qe.nr
    inv_lat = np.linalg.inv(qe.lattice)
    frac = qe.positions @ inv_lat                          # unit-cell fractional

    cell = PhonopyAtoms(symbols=qe.symbols, cell=qe.lattice,
                        scaled_positions=np.mod(frac, 1.0),
                        masses=qe.masses)
    ph = Phonopy(cell, supercell_matrix=np.diag(nr),
                 primitive_matrix=np.eye(3), log_level=0)

    # map phonopy's supercell atoms to (unit atom k, cell index c): derive
    # from positions with an integer-residue assertion, never assumed from
    # ordering (the atom-position-gauge lesson).
    sc = ph.supercell
    spos = sc.scaled_positions                             # supercell frac
    nrv = np.array(nr)
    k_of = np.asarray(sc.numbers)                          # element per sc atom
    # unit-cell fractional of each supercell atom = spos * nr
    ucoord = spos * nrv
    kind = np.empty(len(spos), dtype=int)
    cellv = np.empty((len(spos), 3), dtype=int)
    for s in range(len(spos)):
        best, bestk = None, -1
        for k in range(nat):
            d = ucoord[s] - frac[k]
            c = np.rint(d)
            if np.abs(d - c).max() < 1e-6:
                best, bestk = c.astype(int), k
                break
        if best is None:
            raise ValueError("supercell atom does not reduce to a unit atom "
                             "+ integer cell — position gauge mismatch")
        kind[s] = bestk
        cellv[s] = np.mod(best, nrv)
    # every (k, c) pair must appear exactly once
    tags = kind * nr[0] * nr[1] * nr[2] + (
        (cellv[:, 0] * nr[1] + cellv[:, 1]) * nr[2] + cellv[:, 2])
    if len(set(tags.tolist())) != len(spos):
        raise ValueError("supercell atom mapping is not a bijection")

    # full supercell FC: Φ[(k1,c1),(k2,c2)] = fc_qe[k1,k2, (c2-c1) mod nr]
    n_s = len(spos)
    full = np.empty((n_s, n_s, 3, 3))
    for s1 in range(n_s):
        dm = np.mod(cellv - cellv[s1], nrv)                # (n_s, 3)
        full[s1] = qe.fc[kind[s1], kind, dm[:, 0], dm[:, 1], dm[:, 2]]

    ph.force_constants = full
    ph.save(os.path.join(outdir, "phonopy.yaml"),
            settings={"force_constants": False})
    write_FORCE_CONSTANTS(full, filename=os.path.join(outdir, "FORCE_CONSTANTS"))

    return {
        "n_unit_atoms": nat,
        "n_supercell_atoms": int(n_s),
        "grid": list(nr),
        "polar": False,
        "asr_drift": float(np.abs(full.sum(axis=1)).max()),
    }
