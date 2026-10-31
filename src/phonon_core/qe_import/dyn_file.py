"""Reader for Quantum ESPRESSO ``.dyn`` dynamical-matrix files (ph.x output).

Why this exists alongside the ``.fc`` reader
--------------------------------------------
``q2r.x`` Fourier-transforms a set of ``.dyn`` files into real-space force
constants, and for a POLAR material it first SUBTRACTS the long-range
dipole-dipole term so the result is short-ranged. The ``.fc`` file therefore
does not by itself reproduce the physics: ``matdyn.x`` adds the long-range
part back. That is why ``convert_qe_fc`` refuses polar input.

The ``.dyn`` files have no such caveat. Each carries the dynamical matrix
D(q) that ph.x actually computed, at the q it computed it — no
interpolation, no sum-rule choice, no reconstruction. For a polar material
this is the route that works today, and for any material it is the exact
reference the interpolated route is approximating.

Self-validation
---------------
Every ``.dyn`` carries BOTH D(q) and the frequencies ph.x obtained from it.
:func:`dyn_frequencies` re-derives the frequencies from the matrix, so the
reader is checked against the file's own answer with nothing external
assumed. Measured on real ph.x output: agreement to 1.3e-06 THz, which is
the file's printed precision.

Conventions (pinned by that self-check, not guessed)
----------------------------------------------------
* D(q) is written in Ry/bohr², NOT mass-weighted. Masses in the header are
  in Rydberg atomic units (mₑ = 1/2), the same as in ``.fc``.
* Frequencies follow from the eigenvalues λ of D(q)/√(mᵢmⱼ) as
  ν = sign(λ)·√|λ| · RY_THZ, with RY_THZ DERIVED from CODATA primitives.
* The ``q = ( ... )`` lines are CARTESIAN in 2π/alat units. Fractional
  (crystal) q — what the rest of phonon-vision speaks — is ``at @ q_cart``
  with ``at`` the direct lattice in alat units.
* One file may carry the whole STAR of q (observed: 8 blocks in one file),
  so blocks are collected as a list, never assumed to be one.
* The Γ file additionally carries the dielectric tensor and Born effective
  charges; both are read when present. These are the RAW DFPT charges: their
  sum need NOT vanish (measured 0.389 on real output). ``q2r.x`` imposes
  charge neutrality when it writes the ``.fc``, where the same sum comes out
  at 3e-08 — so the two files legitimately disagree, and only the ``.fc``
  side may be asserted neutral.
* ph.x diagonalizes ONCE per file, for the star's representative q, so
  ``file_frequencies`` covers the first rows of ``dynmat``, not all of them.
"""

from __future__ import annotations

import os
import re

import numpy as np

from phonon_core.qe_import.fc_file import AMU_RY, BOHR_A, RY_EV, _latgen

__all__ = ["QEDynamical", "read_qe_dyn", "dyn_frequencies",
           "dyn_modes", "RY_THZ"]

# CODATA 2018 primitives -> Ry expressed as a frequency, in THz. Derived, so a
# constants change propagates instead of silently disagreeing with the .fc side.
_EV_J = 1.602176634e-19          # exact, SI definition
_PLANCK_J_S = 6.62607015e-34     # exact, SI definition
RY_THZ = RY_EV * _EV_J / _PLANCK_J_S / 1.0e12      # ~3289.842 THz per Ry

_Q_RE = re.compile(r"q\s*=\s*\(\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*\)")
_FREQ_RE = re.compile(r"freq\s*\(\s*\d+\s*\)\s*=\s*([-\d.Ee+]+)\s*\[THz\]")
_TYPE_RE = re.compile(r"^\s*(\d+)\s*'([^']*)'\s*([-\d.Ee+]+)\s*$")


class QEDynamical:
    """Parsed ``.dyn`` content in phonon-vision units (Å, amu, Ry/bohr²)."""

    def __init__(self, *, lattice, symbols, masses, positions, qpoints_cart,
                 qpoints_frac, dynmat, file_frequencies, dielectric, born,
                 symbol_sources, frequency_index=None):
        self.lattice = lattice                  # (3,3) Å, rows a1,a2,a3
        self.symbols = symbols                  # (nat,)
        self.masses = masses                    # (nat,) amu
        self.positions = positions              # (nat,3) Å Cartesian
        self.qpoints_cart = qpoints_cart        # (Nq,3) 2*pi/alat units
        self.qpoints_frac = qpoints_frac        # (Nq,3) crystal
        self.dynmat = dynmat                    # (Nq,nat,nat,3,3) complex Ry/bohr^2
        self.file_frequencies = file_frequencies  # (Nf,3nat) THz, or None
        #: which rows of ``dynmat`` the printed frequencies correspond to.
        #: ph.x diagonalizes only the star REPRESENTATIVE, so for a single
        #: file this is [0, 1, ...] over the printed count, and for a
        #: concatenated set it points at each file's first block.
        self.frequency_index = (np.arange(len(file_frequencies))
                                if frequency_index is None
                                and file_frequencies is not None
                                else frequency_index)
        self.dielectric = dielectric            # (3,3) or None
        self.born = born                        # (nat,3,3) or None
        self.symbol_sources = symbol_sources


def read_qe_dyn(path):
    """Parse one ``.dyn`` file (which may hold the whole star of q)."""
    from phonon_core.qe_import.fc_file import _resolve_symbol

    lines = open(path).read().splitlines()
    if len(lines) < 4:
        raise ValueError(f"{path}: too short to be a .dyn file")

    # header: line 0 is a banner, line 1 a title, line 2 the numeric header
    hdr = None
    for probe in lines[:6]:
        t = probe.split()
        if len(t) >= 9:
            try:
                ints = [int(x) for x in t[:3]]
                floats = [float(x) for x in t[3:9]]
            except ValueError:
                continue
            hdr = (ints, np.array(floats), lines.index(probe))
            break
    if hdr is None:
        raise ValueError(f"{path}: no 'ntyp nat ibrav celldm(1..6)' header")
    (ntyp, nat, ibrav), celldm, hi = hdr

    alat_A = celldm[0] * BOHR_A
    if not alat_A > 0:
        raise ValueError(f"{path}: celldm(1) = {celldm[0]} must be > 0")

    li = hi + 1
    if ibrav == 0:
        at = np.array([[float(x) for x in lines[li + k].split()[:3]]
                       for k in range(3)])
        li += 3
    else:
        at = _latgen(ibrav, celldm)
    lattice = at * alat_A

    type_symbols, type_masses, sources = {}, {}, {}
    read = 0
    while read < ntyp and li < len(lines):
        m = _TYPE_RE.match(lines[li])
        li += 1
        if not m:
            continue
        nt, label, mass_ry = int(m.group(1)), m.group(2), float(m.group(3))
        mass = mass_ry / AMU_RY
        sym, src = _resolve_symbol(label, mass)
        type_symbols[nt], type_masses[nt], sources[nt] = sym, mass, src
        read += 1
    if read != ntyp:
        raise ValueError(f"{path}: found {read} of {ntyp} species lines")

    symbols, masses, positions = [], [], []
    while len(symbols) < nat and li < len(lines):
        t = lines[li].split()
        li += 1
        if len(t) != 5:
            continue
        try:
            ityp = int(t[1])
            tau = np.array([float(x) for x in t[2:5]])
        except ValueError:
            continue
        if ityp not in type_symbols:
            continue
        symbols.append(type_symbols[ityp])
        masses.append(type_masses[ityp])
        positions.append(tau * alat_A)
    if len(symbols) != nat:
        raise ValueError(f"{path}: found {len(symbols)} of {nat} atom lines")

    # ---- (q, D(q)) blocks: a file may carry the whole star -----------------
    qs_cart, mats = [], []
    k = li
    while k < len(lines):
        if "Dynamical" in lines[k] and "cartesian" in lines[k]:
            qm = None
            for j in range(k + 1, min(k + 6, len(lines))):
                qm = _Q_RE.search(lines[j])
                if qm:
                    k = j
                    break
            if qm is None:
                k += 1
                continue
            qs_cart.append([float(qm.group(i)) for i in (1, 2, 3)])
            D = np.zeros((nat, nat, 3, 3), dtype=np.complex128)
            seen = 0
            j = k + 1
            while j < len(lines) and seen < nat * nat:
                t = lines[j].split()
                if len(t) == 2 and t[0].isdigit() and t[1].isdigit():
                    a, b = int(t[0]), int(t[1])
                    if 1 <= a <= nat and 1 <= b <= nat and j + 3 < len(lines):
                        rows = []
                        ok = True
                        for r in range(3):
                            v = lines[j + 1 + r].split()
                            if len(v) != 6:
                                ok = False
                                break
                            try:
                                rows.append([float(x) for x in v])
                            except ValueError:
                                ok = False
                                break
                        if ok:
                            for r in range(3):
                                v = rows[r]
                                D[a - 1, b - 1, r] = [v[0] + 1j * v[1],
                                                      v[2] + 1j * v[3],
                                                      v[4] + 1j * v[5]]
                            seen += 1
                            j += 4
                            continue
                if "Dynamical" in lines[j] or "Diagonalizing" in lines[j]:
                    break
                j += 1
            if seen != nat * nat:
                raise ValueError(f"{path}: q-block at line {k} has {seen} of "
                                 f"{nat * nat} matrix blocks")
            mats.append(D)
            k = j
            continue
        k += 1
    if not mats:
        raise ValueError(f"{path}: no dynamical-matrix blocks found")

    qs_cart = np.array(qs_cart)
    # crystal q = at . q_cart  (at in alat units, q_cart in 2*pi/alat)
    qs_frac = qs_cart @ at.T

    # Frequencies ph.x printed. NOTE it diagonalizes ONCE per file — for the
    # star's REPRESENTATIVE q — not once per star member: a file with 8
    # q-blocks still carries a single 3N-frequency list. So this array covers
    # the FIRST len(file_frequencies) q-blocks, and the self-check compares
    # only those. (An earlier version required one block per q and silently
    # produced None for every starred file.)
    printed = _FREQ_RE.findall("\n".join(lines))
    file_freqs = None
    if printed and len(printed) % (3 * nat) == 0:
        arr = np.array([float(x) for x in printed]).reshape(-1, 3 * nat)
        if arr.shape[0] <= len(mats):
            file_freqs = arr

    # dielectric / Born (Gamma file only)
    dielectric = born = None
    for k, ln in enumerate(lines):
        if "Dielectric Tensor" in ln:
            rows = []
            for j in range(k + 1, min(k + 8, len(lines))):
                t = lines[j].split()
                if len(t) == 3:
                    try:
                        rows.append([float(x) for x in t])
                    except ValueError:
                        pass
                if len(rows) == 3:
                    break
            if len(rows) == 3:
                dielectric = np.array(rows)
        if "Effective Charges E-U" in ln and born is None:
            b = np.zeros((nat, 3, 3))
            got = 0
            j = k + 1
            while j < len(lines) and got < nat:
                t = lines[j].split()
                if len(t) >= 2 and t[0].lower().startswith("atom"):
                    # real ph.x writes "atom #    1" -> the index is the LAST
                    # token, not the second (which is the literal '#')
                    try:
                        idx = int(t[-1])
                    except ValueError:
                        j += 1
                        continue
                    rows = []
                    for r in range(j + 1, min(j + 5, len(lines))):
                        tt = lines[r].split()
                        if len(tt) == 3:
                            try:
                                rows.append([float(x) for x in tt])
                            except ValueError:
                                pass
                        if len(rows) == 3:
                            break
                    if len(rows) == 3 and 1 <= idx <= nat:
                        b[idx - 1] = np.array(rows)
                        got += 1
                j += 1
            if got == nat:
                born = b

    return QEDynamical(lattice=lattice, symbols=symbols,
                       masses=np.array(masses), positions=np.array(positions),
                       qpoints_cart=qs_cart, qpoints_frac=qs_frac,
                       dynmat=np.array(mats), file_frequencies=file_freqs,
                       dielectric=dielectric, born=born, symbol_sources=sources,
                       frequency_index=(np.arange(len(file_freqs))
                                        if file_freqs is not None else None))


def dyn_modes(dyn):
    """Frequencies AND eigenvectors from D(q), in the engine conventions.

    Returns ``(freq (Nq, 3N) signed THz, eigvec (Nq, 3N, N, 3) complex)``.

    * Mass-weights D by 1/√(mᵢmⱼ), hermitizes, diagonalizes; frequencies are
      the signed square root × RY_THZ (an unstable mode stays negative).
    * Eigenvectors are ``eigh`` columns of the MASS-WEIGHTED matrix — i.e.
      already in the project's convention (Dove eq. 4.16, Σ_s|e_s|² = 1,
      displacements u ∝ e/√m downstream), reshaped to the parser layout
      (band, atom, xyz) and phase-canonicalized per mode so φ = 0 is
      reproducible for the animator (the band-egress rule).

    ``dyn.file_frequencies`` is the oracle for the frequency half; the
    eigenvector half inherits its validation because both come from the same
    diagonalization of the same matrix.
    """
    from phonon_core.kinematics import canonicalize_eigenvector_phase

    m = np.repeat(np.asarray(dyn.masses) * AMU_RY, 3)      # back to Ry a.u.
    root = np.sqrt(np.outer(m, m))
    freqs, vecs = [], []
    for D in dyn.dynmat:
        n = D.shape[0]
        flat = D.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n) / root
        flat = 0.5 * (flat + flat.conj().T)
        lam, v = np.linalg.eigh(flat)
        freqs.append(np.sign(lam) * np.sqrt(np.abs(lam)) * RY_THZ)
        modes = v.T.reshape(3 * n, n, 3)                   # (band, atom, xyz)
        vecs.append(np.stack([canonicalize_eigenvector_phase(modes[b])
                              for b in range(3 * n)]))
    return np.array(freqs), np.array(vecs)


def dyn_frequencies(dyn):
    """Frequencies (Nq, 3N) in signed THz — the frequency half of
    :func:`dyn_modes` (one spelling; this is a thin wrapper).

    Compare against ``dyn.file_frequencies`` to validate a parse: the two are
    independent paths to the same number.
    """
    return dyn_modes(dyn)[0]


def read_qe_dyn_set(paths):
    """Read several ``.dyn`` files and concatenate their q-blocks.

    ``paths`` may include the ``.dyn0`` grid-definition file, which carries no
    matrices and is skipped. The structure of the first file with matrices is
    taken as the material and every later file is required to agree on the
    atom count — a mixed set is a user error worth catching early.
    """
    dyns = []
    for p in paths:
        if not os.path.exists(p):
            raise ValueError(f"missing .dyn file: {p}")
        try:
            dyns.append(read_qe_dyn(p))
        except ValueError as exc:
            msg = str(exc)
            # .dyn0 carries only the q-grid definition: no header, no
            # matrices. Skip those two shapes, propagate anything else.
            if ("no dynamical-matrix blocks" in msg
                    or "no 'ntyp nat ibrav" in msg):
                continue
            raise
    if not dyns:
        raise ValueError("no .dyn file in the set carried matrices")
    base = dyns[0]
    for d in dyns[1:]:
        if len(d.symbols) != len(base.symbols):
            raise ValueError("mixed materials in the .dyn set: "
                             f"{len(base.symbols)} vs {len(d.symbols)} atoms")
    # Concatenate, and REMAP the printed-frequency index. Each file's printed
    # frequencies belong to that file's own first q-blocks; stacking them
    # naively while stacking the (larger) star blocks destroys the pairing —
    # measured as a 1.19 THz self-check failure where per-file it was 1.7e-06.
    offsets, run = [], 0
    for d in dyns:
        offsets.append(run)
        run += len(d.qpoints_frac)
    base.qpoints_cart = np.vstack([d.qpoints_cart for d in dyns])
    base.qpoints_frac = np.vstack([d.qpoints_frac for d in dyns])
    base.dynmat = np.concatenate([d.dynmat for d in dyns], axis=0)
    ff, idx = [], []
    for off, d in zip(offsets, dyns):
        if d.file_frequencies is None:
            continue
        ff.append(d.file_frequencies)
        idx.extend(off + np.arange(len(d.file_frequencies)))
    base.file_frequencies = np.vstack(ff) if ff else None
    base.frequency_index = np.array(idx, dtype=int) if idx else None
    for d in dyns:                              # Gamma file carries these
        if d.dielectric is not None and base.dielectric is None:
            base.dielectric = d.dielectric
        if d.born is not None and base.born is None:
            base.born = d.born
    return base
