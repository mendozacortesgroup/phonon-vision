"""phonon_core.engine.dq_numpy — pure-numpy D(q) engine (A2, Level-2 Stage 4).

The numpy twin of :mod:`phonon_core.engine.dq_phonopy` (A1): same contract —
``load_engine_np`` + ``evaluate_dq_np`` returning SIGNED THz frequencies and
parser-layout eigenvectors — plus a batched API, with ZERO phonopy dependency.
This is the module the Blender add-on will vendor (numpy ships inside
Blender's Python; phonopy+spglib wheels are not an acceptable dependency).
A1 is the ORACLE: A2 ships only because it reproduces A1 everywhere tested
(tests/test_engine_dq_numpy.py, three tiers + negative controls).

PINNED CONVENTIONS (architect spike vs phonopy 4.3.1 — do not change without
re-running the oracle battery):

- Dynamical matrix::

      D(q)_{i alpha, j beta} = (m_i m_j)^(-1/2)
          * sum_{j_s in supercell images of primitive j}
              Phi_{alpha beta}(i, j_s)
              * [ (1/m) sum_{v in shortest(j_s, i)} e^{2 pi i q . v} ]

  with v in PRIMITIVE FRACTIONAL coordinates in the ATOM-POSITION gauge:
  v = R_lattice + (tau_j - tau_i)   (spike probe: integer residue of
  v - (tau_j - tau_i) is 0 to 1e-8; asserted at load time below).
- TIES: equal-weight average over ALL images at the shortest distance
  (multiplicity m). This defines D(q) for truncated-supercell force
  constants and must match phonopy's tie semantics — A1 IS THE ORACLE. The
  tie test is ``d <= d_min + symprec`` with symprec an ABSOLUTE length in
  ANGSTROM (phonopy's ``ShortestPairs`` compares ``distances < symprec``,
  default 1e-5, with NO distance scaling). It is NOT relative: a relative
  tolerance grows with the pair distance (1e-5 relative is 7e-5..1.1e-4 A at
  7-11 A) and then ties NON-degenerate images in pseudo-symmetric cells.
  Getting this wrong is INVISIBLE AT GAMMA — every phase there is 1, so
  averaging 1's still gives 1 — and only shows up at generic q; it cost the
  0.6.86 line a real bug (see CHANGELOG 0.6.87).
- Frequencies: nu = sign(lambda) * sqrt(|lambda|) * 15.633302300230191 THz
  (force constants in eV/Angstrom^2, masses in amu, lengths in Angstrom;
  the factor is phonopy's VaspToTHz = sqrt(e/amu)/(2 pi)/1e12/1e-10).
  Signed-sqrt convention: Togo & Tanaka, Scr. Mater. 108, 1 (2015).
- Eigenvectors: columns of eigh(D) (ascending eigenvalues, matching A1),
  each column reshaped per mode to (N_a, 3) — identical to the A1 / parser
  layout (``BandPath.eigenvectors`` per q). Mass-weighted, Maradudin &
  Vosko (1968) normalization — exactly what eigh of THIS D produces.
- Spike ground truth reproduced by the battery: |D_A2 - D_phonopy| ~ 7e-18,
  |d nu| ~ 6e-15 THz, overlap deficit 0.0 on the 3D toy.

Shortest-vector table (the real algorithmic content): for each pair
(supercell atom j_s, primitive atom i), the candidate vectors are
(r_{j_s} - r_i + L) over supercell lattice translations L in a [-1, 0, 1]^3
neighbor shell; the shortest SET is every candidate within ``symprec``
ANGSTROM (absolute) of the minimum distance — phonopy's rule, verbatim. The
tolerance is exposed as ``load_engine_np(..., symprec=1e-5)``: same name,
same default, same units as phonopy, so a caller who changes phonopy's
symprec can match it here. The shell is SELF-VALIDATING: the table is
rebuilt on a [-2..2]^3 shell and must agree exactly (else the wider shell
is adopted and re-validated against [-3..3]^3 — a mismatch there is an
error). Storage is phonopy-4.x-style: one flat (S, 3) svecs array +
(count, offset) per pair, for cache locality and batched einsum phases.

NAC / BORN: OUT OF SCOPE. If a BORN file sits next to phonopy.yaml this
loader REFUSES loudly rather than silently computing the wrong polar
physics — use the A1 engine for NAC materials.

YAML parsing: the file is read ONCE through the package's own machinery
(:func:`phonon_core.parsers._yaml_loader.load_yaml` — the C-loader shim —
and :func:`phonon_core.parsers._phonopy_yaml.parse_phonopy_yaml_material`
for the primitive cell). Only the engine-specific SUPERCELL-block
extraction (lattice, scaled positions, reduced_to maps) lives here: no
existing parser exposes it, and the vendored-parser set is deliberately
untouched (wheel parity).

Dependencies: numpy (always); PyYAML via the parsers (load path only);
h5py imported lazily ONLY when force_constants.hdf5 is the FC source.
No phonopy import anywhere in this module — guarded by the test battery.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from phonon_core.types import Material
from phonon_core.parsers._yaml_loader import load_yaml
from phonon_core.parsers._phonopy_yaml import parse_phonopy_yaml_material

#: sqrt(eV / (amu * Angstrom^2)) expressed in THz (ordinary frequency):
#: phonopy's VaspToTHz. PINNED — the oracle battery compares against A1
#: at 1e-8 THz, three orders below any representable drift in this factor.
FREQ_FACTOR_THZ = 15.633302300230191

#: Shortest-set tie tolerance: phonopy's ``symprec``, an ABSOLUTE length in
#: ANGSTROM (phonopy ShortestPairs: ``distances < symprec``, default 1e-5, no
#: distance scaling). NOT relative — see the TIES bullet above. Overridable
#: per call via ``load_engine_np(..., symprec=...)``.
DEFAULT_SYMPREC = 1e-5

#: Float epsilon on top of symprec, so an EXACT tie (two images at bitwise
#: equal distance, e.g. a symmetry-enforced degeneracy) can never be split by
#: rounding. Immaterial next to symprec; it exists only for exactness.
_TIE_EPS = 1e-12

#: Atom-position-gauge integer-residue assertion bound (spike: <= 1e-8).
_GAUGE_TOL = 1e-6


@dataclass
class NumpyEngine:
    """Everything :func:`evaluate_dq_np` needs, as plain numpy arrays.

    Constructed by :func:`load_engine_np`; tests may assemble one directly
    from arrays (Tier-1 battery does, to stay phonopy- and file-free).
    """
    plat: np.ndarray        # (3, 3) primitive lattice, rows a_i (Angstrom)
    masses: np.ndarray      # (N_p,) primitive atom masses (amu)
    p2s: np.ndarray         # (N_p,) supercell index of each primitive atom
    sp2p: np.ndarray        # (N_s,) primitive index of each supercell atom
    fc: np.ndarray          # (N_p, N_s, 3, 3) compact force constants (eV/A^2)
    svecs: np.ndarray       # (S, 3) flat shortest vectors, PRIMITIVE FRACTIONAL
    counts: np.ndarray      # (N_s, N_p) shortest-set multiplicities
    offsets: np.ndarray     # (N_s, N_p) offsets into svecs (C-order pairs)
    material: Material      # the primitive cell (symbols/masses/lattices)

    @property
    def n_p(self) -> int:
        return int(self.masses.shape[0])

    @property
    def n_s(self) -> int:
        return int(self.sp2p.shape[0])


# ---------------------------------------------------------------------------
# FORCE_CONSTANTS readers
# ---------------------------------------------------------------------------
def parse_force_constants_text(path: str) -> np.ndarray:
    """Parse a phonopy FORCE_CONSTANTS text file -> (n1, n2, 3, 3) float64.

    Header line "n1 n2": compact (n1 = N_primitive) or full (n1 == n2 = N_s).
    Each block: an index line "i j" followed by three rows of three floats.
    Blocks are read SEQUENTIALLY (phonopy's own reader does the same); the
    index line is sanity-checked, not used for placement.
    """
    with open(path) as fh:
        tokens = fh.read().split()
    if len(tokens) < 2:
        raise ValueError(f"{path}: not a FORCE_CONSTANTS file (empty header)")
    n1, n2 = int(tokens[0]), int(tokens[1])
    need = 2 + n1 * n2 * 11          # per block: 2 index ints + 9 floats
    if len(tokens) != need:
        raise ValueError(
            f"{path}: token count {len(tokens)} != expected {need} "
            f"for header {n1} {n2}")
    body = np.asarray(tokens[2:], dtype=np.float64).reshape(n1 * n2, 11)
    idx_j = body[:, 1].astype(np.int64)
    # the j index must cycle 1..n2 for each i row — cheap corruption guard
    if not np.array_equal(idx_j, np.tile(np.arange(1, n2 + 1), n1)):
        raise ValueError(f"{path}: block index sequence is not 1..{n2} per row")
    return body[:, 2:].reshape(n1, n2, 3, 3)


def _read_force_constants_hdf5(path: str, p2s: np.ndarray) -> np.ndarray:
    """Read force_constants.hdf5 -> (n1, N_s, 3, 3); verify unit + p2s_map."""
    import h5py   # lazy: only the hdf5 branch needs it
    with h5py.File(path, "r") as fh:
        fc = np.asarray(fh["force_constants"], dtype=np.float64)
        if "physical_unit" in fh:
            unit = np.asarray(fh["physical_unit"]).ravel()[0]
            if isinstance(unit, bytes):
                unit = unit.decode()
            if unit != "eV/angstrom^2":
                raise ValueError(
                    f"{path}: physical_unit {unit!r} != 'eV/angstrom^2'")
        if "p2s_map" in fh:
            file_p2s = np.asarray(fh["p2s_map"], dtype=np.int64)
            if fc.shape[0] == p2s.shape[0] and not np.array_equal(file_p2s, p2s):
                raise ValueError(
                    f"{path}: p2s_map {file_p2s.tolist()} != the yaml-derived "
                    f"map {p2s.tolist()}")
    return fc


def _to_compact(fc: np.ndarray, p2s: np.ndarray, n_s: int) -> np.ndarray:
    """(N_s, N_s) full or (N_p, N_s) compact FC -> compact rows."""
    if fc.ndim != 4 or fc.shape[2:] != (3, 3) or fc.shape[1] != n_s:
        raise ValueError(f"force constants shape {fc.shape} does not match "
                         f"the supercell ({n_s} atoms)")
    if fc.shape[0] == n_s:              # full -> compact rows via p2s
        return np.ascontiguousarray(fc[p2s])
    if fc.shape[0] == p2s.shape[0]:     # already compact
        return np.ascontiguousarray(fc)
    raise ValueError(f"force constants shape {fc.shape}: first axis is "
                     f"neither N_s={n_s} nor N_p={p2s.shape[0]}")


# ---------------------------------------------------------------------------
# Shortest-vector table
# ---------------------------------------------------------------------------
def _shortest_table_shell(r_s: np.ndarray, r_p: np.ndarray, slat: np.ndarray,
                          shell: int, symprec: float = DEFAULT_SYMPREC):
    """Distances + tie masks for one neighbor shell.

    The tie test is phonopy's: ``d <= d_min + symprec`` with symprec an
    ABSOLUTE length in Angstrom (no distance scaling).

    Returns (dmin (N_s,N_p), mask (N_s,N_p,K), cand (N_s,N_p,K,3) Cartesian).
    """
    rng = np.arange(-shell, shell + 1)
    L = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"),
                 axis=-1).reshape(-1, 3)          # (K, 3) supercell units
    t = L @ slat                                   # (K, 3) Cartesian
    # cand[j_s, i, k] = r_s[j_s] - r_p[i] + t[k]
    cand = (r_s[:, None, None, :] - r_p[None, :, None, :]
            + t[None, None, :, :])                 # (N_s, N_p, K, 3)
    d = np.linalg.norm(cand, axis=-1)              # (N_s, N_p, K)
    dmin = d.min(axis=-1)                          # (N_s, N_p)
    mask = d <= dmin[..., None] + symprec + _TIE_EPS
    return dmin, mask, cand


def build_shortest_vectors(slat: np.ndarray, plat: np.ndarray,
                           r_s_cart: np.ndarray, p2s: np.ndarray,
                           symprec: float = DEFAULT_SYMPREC):
    """The (svecs, counts, offsets) table for :class:`NumpyEngine`.

    ``symprec`` is phonopy's ABSOLUTE Angstrom tie tolerance (see the module
    docstring): the shortest set of a pair is every candidate within symprec
    of that pair's minimum distance.

    Self-validating shell (module docstring): built on the [-1..1]^3 shell,
    verified against [-2..2]^3; widened (and re-verified against [-3..3]^3)
    only if they disagree.
    """
    r_p = r_s_cart[p2s]                            # (N_p, 3)
    shell = 1
    dmin, mask, cand = _shortest_table_shell(r_s_cart, r_p, slat, shell, symprec)
    d2, m2, c2 = _shortest_table_shell(r_s_cart, r_p, slat, shell + 1, symprec)
    if not (np.allclose(dmin, d2, atol=1e-10)
            and np.array_equal(mask.sum(-1), m2.sum(-1))):
        # a shortest vector lay on the [-1..1] shell edge: widen and re-verify
        shell = 2
        dmin, mask, cand = d2, m2, c2
        d3, m3, _ = _shortest_table_shell(r_s_cart, r_p, slat, shell + 1, symprec)
        if not (np.allclose(dmin, d3, atol=1e-10)
                and np.array_equal(mask.sum(-1), m3.sum(-1))):
            raise ValueError(
                "shortest-vector table does not converge by the [-2..2]^3 "
                "neighbor shell — pathological supercell geometry")
    counts = mask.sum(axis=-1).astype(np.int64)    # (N_s, N_p)
    if counts.min() < 1:
        raise ValueError("empty shortest set (internal error)")
    inv_plat = np.linalg.inv(plat)
    svecs = cand[mask] @ inv_plat                  # (S, 3) PRIMITIVE FRACTIONAL
    offsets = np.zeros_like(counts)
    offsets.reshape(-1)[1:] = np.cumsum(counts.reshape(-1))[:-1]
    return np.ascontiguousarray(svecs), counts, offsets


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_engine_np(phonopy_yaml: str,
                   force_constants: Optional[str] = None,
                   symprec: float = DEFAULT_SYMPREC) -> NumpyEngine:
    """Build a :class:`NumpyEngine` given phonopy.yaml (+ FORCE_CONSTANTS).

    FC discovery mirrors A1: an explicit ``force_constants`` path wins; else
    a sibling text ``FORCE_CONSTANTS``; else a sibling
    ``force_constants.hdf5``. A sibling ``BORN`` file is a hard REFUSAL
    (this engine has no non-analytic correction).

    Parameters
    ----------
    symprec : float, default 1e-5
        The shortest-vector tie tolerance, an ABSOLUTE length in ANGSTROM —
        phonopy's ``symprec``, same name/default/units, so a caller who
        changed phonopy's symprec can match it here. Two images of a pair are
        tied (and their phases equal-weight averaged) iff their distances
        differ by <= symprec. See the module docstring: this is NOT a relative
        tolerance, and the difference is invisible at Gamma.
    """
    phonopy_yaml = os.fspath(phonopy_yaml)
    here = os.path.dirname(os.path.abspath(phonopy_yaml))

    if os.path.exists(os.path.join(here, "BORN")):
        raise RuntimeError(
            "A2 has no non-analytic correction yet; use the A1 engine "
            f"(BORN file found next to {os.path.basename(phonopy_yaml)})")

    material = parse_phonopy_yaml_material(phonopy_yaml)   # primitive cell
    data = load_yaml(phonopy_yaml)

    # ---- supercell block (engine-specific; no parser exposes it) ----
    sc = data.get("supercell")
    if not (isinstance(sc, dict) and "lattice" in sc and "points" in sc):
        raise ValueError(f"{phonopy_yaml}: no 'supercell' block — this "
                         "engine needs the full phonopy.yaml, not band/mesh "
                         "output yaml")
    slat = np.asarray(sc["lattice"], dtype=np.float64)          # rows (3,3)
    pts = sc["points"]
    sfrac = np.asarray([p["coordinates"] for p in pts], dtype=np.float64)
    smass = np.asarray([p["mass"] for p in pts], dtype=np.float64)
    ssym = [p["symbol"] for p in pts]
    try:
        reduced = np.asarray([p["reduced_to"] for p in pts],
                             dtype=np.int64) - 1                # 0-based
    except KeyError:
        raise ValueError(f"{phonopy_yaml}: supercell points carry no "
                         "'reduced_to' fields (phonopy >= 2.x writes them)")

    n_s = sfrac.shape[0]
    n_p = material.masses.shape[0]
    plat = np.asarray(material.lattice, dtype=np.float64)
    r_s_cart = sfrac @ slat                                     # (N_s, 3)
    inv_plat = np.linalg.inv(plat)

    # ---- p2s: match each primitive atom to its supercell representative ----
    reps = sorted(set(int(r) for r in reduced))                 # candidates
    if len(reps) != n_p:
        raise ValueError(f"{phonopy_yaml}: {len(reps)} distinct reduced_to "
                         f"targets != {n_p} primitive atoms")
    prim_frac = np.asarray(material.fractional_coords, dtype=np.float64)
    p2s = np.empty(n_p, dtype=np.int64)
    used = set()
    for k in range(n_p):
        hit = -1
        for s in reps:
            if s in used or ssym[s] != material.symbols[k]:
                continue
            dfrac = r_s_cart[s] @ inv_plat - prim_frac[k]
            if np.max(np.abs(dfrac - np.round(dfrac))) < 1e-6:
                hit = s
                break
        if hit < 0:
            raise ValueError(f"{phonopy_yaml}: no supercell representative "
                             f"matches primitive atom {k} "
                             f"({material.symbols[k]} @ {prim_frac[k]})")
        p2s[k] = hit
        used.add(hit)
    if not np.allclose(smass[p2s], material.masses, atol=1e-8):
        raise ValueError(f"{phonopy_yaml}: representative masses disagree "
                         "with the primitive cell")

    # sp2p: primitive index of each supercell atom
    rep_to_prim = {int(p2s[k]): k for k in range(n_p)}
    try:
        sp2p = np.asarray([rep_to_prim[int(r)] for r in reduced],
                          dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"{phonopy_yaml}: reduced_to target {e} is not a "
                         "primitive representative")

    # ---- force constants ----
    if force_constants is None:
        cand_txt = os.path.join(here, "FORCE_CONSTANTS")
        cand_h5 = os.path.join(here, "force_constants.hdf5")
        if os.path.exists(cand_txt):
            force_constants = cand_txt
        elif os.path.exists(cand_h5):
            force_constants = cand_h5
        else:
            raise FileNotFoundError(
                f"no FORCE_CONSTANTS / force_constants.hdf5 next to "
                f"{phonopy_yaml} and none given")
    if force_constants.endswith((".hdf5", ".h5")):
        fc_raw = _read_force_constants_hdf5(force_constants, p2s)
    else:
        fc_raw = parse_force_constants_text(force_constants)
    fc = _to_compact(fc_raw, p2s, n_s)

    # ---- shortest-vector table + the pinned gauge assertion ----
    svecs, counts, offsets = build_shortest_vectors(slat, plat, r_s_cart, p2s,
                                                    symprec)
    # ATOM-POSITION gauge probe (module docstring): v - (tau_j - tau_i) must
    # be a primitive LATTICE vector (integer components) for every pair.
    tau = r_s_cart[p2s] @ inv_plat                              # (N_p,) frac
    pair_j = np.repeat(sp2p[:, None], n_p, axis=1)              # (N_s, N_p)
    pair_i = np.repeat(np.arange(n_p)[None, :], n_s, axis=0)
    dtau_flat = np.repeat((tau[pair_j] - tau[pair_i]).reshape(-1, 3),
                          counts.reshape(-1), axis=0)           # (S, 3)
    resid = svecs - dtau_flat
    gauge_err = float(np.max(np.abs(resid - np.round(resid))))
    if gauge_err > _GAUGE_TOL:
        raise ValueError(f"atom-position-gauge residue {gauge_err:.2e} "
                         f"exceeds {_GAUGE_TOL:.0e} (supercell/primitive "
                         "geometry inconsistent?)")

    return NumpyEngine(plat=plat, masses=np.asarray(material.masses,
                                                    dtype=np.float64),
                       p2s=p2s, sp2p=sp2p, fc=fc,
                       svecs=svecs, counts=counts, offsets=offsets,
                       material=material)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _dynamical_matrices(eng: NumpyEngine, q_arr: np.ndarray) -> np.ndarray:
    """Stacked Hermitian D(q) for (Nq, 3) fractional q -> (Nq, 3N, 3N)."""
    n_p, n_s = eng.n_p, eng.n_s
    nq = q_arr.shape[0]
    # tie-averaged phase per (q, pair): flat einsum then segment mean —
    # the batched-from-day-one design (one exp / one reduceat per call).
    # fixed-order contraction (NOT `q_arr @ svecs.T`): BLAS gemm picks
    # Nq-dependent blocking, breaking the pinned batch == singles BITWISE
    # identity; einsum(optimize=False) sums the 3 components in a fixed
    # order per element regardless of Nq.
    arg = np.einsum("qk,sk->qs", q_arr, eng.svecs, optimize=False)
    phase_flat = np.exp(2j * np.pi * arg)                         # (Nq, S)
    sums = np.add.reduceat(phase_flat, eng.offsets.reshape(-1), axis=1)
    ph = (sums / eng.counts.reshape(-1)).reshape(nq, n_s, n_p)    # (Nq,N_s,N_p)
    # D[q, i, a, j, b] = sum_{s: sp2p[s]==j} ph[q, s, i] * fc[i, s, a, b],
    # contracted per primitive-column group j with a FIXED einsum order
    # (optimize=False): the fp summation order over s is then independent of
    # Nq, which is what makes batch == loop-of-singles BITWISE (pinned).
    D5 = np.empty((nq, n_p, 3, n_p, 3), dtype=np.complex128)
    for j in range(n_p):
        idx = np.flatnonzero(eng.sp2p == j)
        D5[:, :, :, j, :] = np.einsum("qsi,isab->qiab", ph[:, idx, :],
                                      eng.fc[:, idx], optimize=False)
    D = D5.reshape(nq, 3 * n_p, 3 * n_p)
    msqrt = np.sqrt(np.repeat(eng.masses, 3))
    D /= np.outer(msqrt, msqrt)[None, :, :]
    return (D + np.conj(np.transpose(D, (0, 2, 1)))) / 2.0        # hermitize


def dynamical_matrix_at_np(eng: NumpyEngine, q_frac) -> np.ndarray:
    """The raw (3N, 3N) complex D(q) — A2's answer to A1's
    ``dynamical_matrix_at``; exposed for the oracle battery only."""
    q = np.asarray(q_frac, dtype=np.float64).reshape(1, 3)
    return _dynamical_matrices(eng, q)[0]


def evaluate_dq_np_batch(eng: NumpyEngine,
                         q_array) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate D(q) at a batch of fractional q-points.

    Parameters
    ----------
    q_array : (Nq, 3) array-like — primitive-basis fractional q.

    Returns
    -------
    freq_THz : (Nq, N_b) float64 — SIGNED THz, ascending per q (A1 contract).
    eigvec : (Nq, N_b, N_a, 3) complex128 — parser-layout eigenvectors
        (eigh columns transposed to modes, each reshaped (N_a, 3)).
    """
    q = np.atleast_2d(np.asarray(q_array, dtype=np.float64))
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"q_array must be (Nq, 3), got {q.shape}")
    D = _dynamical_matrices(eng, q)
    lam, cols = np.linalg.eigh(D)                    # ascending, batched
    freq = np.sign(lam) * np.sqrt(np.abs(lam)) * FREQ_FACTOR_THZ
    n_b = lam.shape[1]
    n_a = n_b // 3
    # columns are modes; rows encode atom*3+cartesian — the identical
    # transpose+reshape A1 / parsers/band_hdf5.py apply.
    eig = np.transpose(cols, (0, 2, 1)).reshape(q.shape[0], n_b, n_a, 3)
    return freq, eig


def evaluate_dq_np(eng: NumpyEngine, q_frac) -> Tuple[np.ndarray, np.ndarray]:
    """Single-q convenience wrapping :func:`evaluate_dq_np_batch` —
    bitwise-identical to batch row 0 by construction (A1's signature)."""
    q = np.asarray(q_frac, dtype=np.float64).reshape(1, 3)
    freq, eig = evaluate_dq_np_batch(eng, q)
    return freq[0], eig[0]
