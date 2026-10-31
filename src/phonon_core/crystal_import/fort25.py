"""Parsers for CRYSTAL23 fort.25 BAND blocks and the phonon .out.

fort.25 BAND block (observed on real CRYSTAL23 output; matches the group's
own ``PhononBands.py`` reading):

    -%-1BAND    9 1000 0.00000E+00 5.05505E-01 0.00000E+00
     0.00000E+00 2.50000E+03
        0    0    0    3    0    0
    <frequencies: n_points * n_bands values, 6 per line, cm^-1, E-format,
     often WITHOUT separating spaces (fixed 12-char fields) — token-splitting
     shreds "-2.37542E-01-2.37542E-01"; fields are cut by position/regex>

Header tokens: [1] = n_bands, [2] = n_points; third line = the segment's
integer endpoints (from_i from_j from_k to_i to_j to_k) whose DENOMINATOR
comes from the .out path lines. Frequencies are cm^-1; conversion uses the
project's ONE frozen constant (io.json_v1.THZ_TO_CM1) — the group's script
hard-codes 0.029979, which is 1/THZ_TO_CM1 to 4 digits.
"""

from __future__ import annotations

import json
import re

import numpy as np

from phonon_core.io.json_v1 import THZ_TO_CM1
from phonon_core.radii import covalent_radius, jmol_color

__all__ = ["read_fort25", "read_crystal_out_structure", "convert_crystal_bands"]

_NUM = re.compile(r"[-+]?\d+\.\d+E[-+]?\d+")
#: standard fractional coordinates -> conventional hexagonal/square labels
_LABELS = {
    (0.0, 0.0, 0.0): "Γ",
    (0.5, 0.0, 0.0): "M", (0.0, 0.5, 0.0): "M",
    (1 / 3, 1 / 3, 0.0): "K", (2 / 3, 2 / 3, 0.0): "K",
    (0.5, 0.5, 0.0): "X",
}


def _label_for(q):
    for frac, lab in _LABELS.items():
        if max(abs(q[0] - frac[0]), abs(q[1] - frac[1]), abs(q[2] - frac[2])) < 1e-6:
            return lab
    return "(" + ",".join(f"{x:.3g}" for x in q) + ")"


def read_fort25(path):
    """Parse every ``-%-...BAND`` block.

    Returns a list of segments:
    ``{n_bands, n_points, from_int, to_int, freq_thz (n_points, n_bands)}``.
    Frequencies are SIGNED THz (CRYSTAL's small negative acoustic values at
    Γ stay negative — sign is physics here as everywhere else).
    """
    segments = []
    cur = None

    def close(cur):
        vals = np.array(cur.pop("vals"), dtype=np.float64)
        expect = cur["n_points"] * cur["n_bands"]
        if vals.size != expect:
            raise ValueError(
                f"{path}: BAND block has {vals.size} values, expected "
                f"{cur['n_points']}x{cur['n_bands']} = {expect}")
        cur["freq_thz"] = (vals / THZ_TO_CM1).reshape(cur["n_points"],
                                                      cur["n_bands"])
        segments.append(cur)

    with open(path) as fh:
        lines = fh.read().splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("-%-") and "BAND" in ln:
            if cur is not None:
                close(cur)
            t = ln.split()
            ints = lines[i + 2].split()
            cur = {"n_bands": int(t[1]), "n_points": int(t[2]),
                   "from_int": tuple(int(x) for x in ints[:3]),
                   "to_int": tuple(int(x) for x in ints[3:6]),
                   "vals": []}
            i += 3
            continue
        if cur is not None:
            cur["vals"].extend(float(x) for x in _NUM.findall(ln))
        i += 1
    if cur is not None:
        close(cur)
    if not segments:
        raise ValueError(f"{path}: no BAND blocks found")
    return segments


def read_crystal_out_structure(path):
    """Geometry + path metadata from a CRYSTAL SCELPHONO phonon .out.

    The .out prints the SUPERCELL (lattice and atoms); the primitive cell is
    recovered by inverting the printed expansion matrix and folding the
    supercell atoms — the fold must yield exactly n_super / det(E) distinct
    atoms, asserted, not assumed. For a SLAB the table's third column is
    CARTESIAN Z in Angstrom (CRYSTAL's 2D convention) and E acts on the two
    periodic axes only.

    Returns dict: ``lattice`` (3,3) Å primitive rows, ``symbols``, ``masses``
    (the run's own isotopic masses), ``fractional_coords`` (z folded over the
    vacuum axis for slabs), ``dimensionality``, ``path`` = list of
    ``(from_frac, to_frac)`` in primitive fractional units.
    """
    text = open(path).read()
    lines = text.splitlines()

    # expansion matrix (slab: E3 row is zero -> treat as identity on axis 3)
    m = re.search(r"EXPANSION MATRIX OF PRIMITIVE CELL\s*\n"
                  r"\s*E1\s+(\S+)\s+(\S+)\s+(\S+)\s*\n"
                  r"\s*E2\s+(\S+)\s+(\S+)\s+(\S+)\s*\n"
                  r"\s*E3\s+(\S+)\s+(\S+)\s+(\S+)", text)
    if not m:
        raise ValueError(f"{path}: no EXPANSION MATRIX block — is this a "
                         "SCELPHONO phonon output?")
    E = np.array([float(x) for x in m.groups()]).reshape(3, 3)
    is_slab = np.allclose(E[2], 0.0)
    if is_slab:
        E = E.copy()
        E[2, 2] = 1.0

    # supercell lattice: LAST "DIRECT LATTICE VECTORS CARTESIAN COMPONENTS"
    lat_blocks = [k for k, ln in enumerate(lines)
                  if "DIRECT LATTICE VECTORS CARTESIAN COMPONENTS" in ln]
    if not lat_blocks:
        raise ValueError(f"{path}: no cartesian lattice block")
    k = lat_blocks[-1]
    L_super = np.array([[float(x) for x in lines[k + 2 + r].split()[:3]]
                        for r in range(3)])
    L_prim = np.linalg.inv(E) @ L_super

    # supercell atom table: last "ATOMS IN THE UNIT CELL: N" table
    at_blocks = [k for k, ln in enumerate(lines)
                 if "ATOMS IN THE UNIT CELL:" in ln]
    if not at_blocks:
        raise ValueError(f"{path}: no atom table")
    k = at_blocks[-1]
    n_super = int(lines[k].rsplit(":", 1)[1])
    rows = []
    j = k + 3
    while len(rows) < n_super and j < len(lines):
        t = lines[j].split()
        j += 1
        if len(t) >= 7 and t[0].isdigit():
            sym = t[3].capitalize()
            rows.append((sym, float(t[4]), float(t[5]), float(t[6])))
    if len(rows) != n_super:
        raise ValueError(f"{path}: found {len(rows)} of {n_super} atoms")

    # fold supercell atoms into the primitive cell. Slab: columns are
    # (frac_a, frac_b, z_angstrom); bulk: all-fractional.
    det = abs(round(np.linalg.det(E)))
    def qkey(x):
        # quantize a fractional coordinate WITH modular wrap: 0.99999 and
        # 0.00001 must collide (a first fold gave 6 "distinct" atoms of 3
        # because round-after-mod separates the two sides of the seam)
        return int(round((x % 1.0) * 1.0e4)) % 10000

    prim = {}
    for sym, a, b, c in rows:
        if is_slab:
            fa = (np.array([a, b]) @ E[:2, :2]) % 1.0     # supercell->primitive frac
            key = (sym, qkey(fa[0]), qkey(fa[1]), round(c, 4))
            frac = (fa[0], fa[1], c)
        else:
            fv = (np.array([a, b, c]) @ E) % 1.0
            key = (sym,) + tuple(qkey(x) for x in fv)
            frac = tuple(fv)
        prim.setdefault(key, frac)
    n_prim = n_super // int(det)
    if len(prim) != n_prim:
        raise ValueError(f"{path}: primitive fold gave {len(prim)} distinct "
                         f"atoms, expected {n_prim} (= {n_super}/{int(det)})")
    symbols, fracs = [], []
    for (sym, *_), frac in prim.items():
        symbols.append(sym)
        if is_slab:
            # z is cartesian Å -> fractional over the vacuum axis
            fracs.append([frac[0], frac[1],
                          (frac[2] / L_prim[2, 2]) % 1.0])
        else:
            fracs.append(list(frac))

    # masses actually used (isotopic table), per species symbol
    mass_by_sym = {}
    mm = re.search(r"ISOTOPIC MASS \(AMU\) FOR FREQUENCY CALCULATION"
                   r"([\s\S]{0,20000}?)\n\s*\n\s*\n", text)
    if mm is None:
        mm = re.search(r"ISOTOPIC MASS \(AMU\) FOR FREQUENCY CALCULATION"
                       r"([\s\S]{0,20000})", text)
    if mm:
        # entries are "idx SYM mass" triplets; the table may be terminated by
        # a ****** divider, so match triplets rather than walking tokens
        for g in re.finditer(r"(\d+)\s+([A-Z][A-Z]?)\s+(\d+\.\d+)",
                             mm.group(1)):
            mass_by_sym[g.group(2).capitalize()] = float(g.group(3))
    if not mass_by_sym:
        raise ValueError(f"{path}: no ISOTOPIC MASS table — masses are the "
                         "run's own convention and are not guessed")
    masses = [mass_by_sym[s] for s in symbols]

    # the path, in primitive fractional units: FROM K (i j k) TO K (...) / den
    path_segs = []
    for fm in re.finditer(r"FROM K\s*\(\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\)\s*"
                          r"TO K\s*\(\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\)\s*"
                          r"WITH DENOMINATOR\s+(\d+)", text):
        v = [int(x) for x in fm.groups()]
        den = float(v[6])
        path_segs.append((tuple(x / den for x in v[0:3]),
                          tuple(x / den for x in v[3:6])))

    return {"lattice": L_prim, "symbols": symbols, "masses": masses,
            "fractional_coords": np.array(fracs),
            "dimensionality": 2 if is_slab else 3, "path": path_segs,
            "n_supercell_atoms": n_super, "expansion_diag": np.diag(E).tolist()}


def convert_crystal_bands(f25_path, out_path, output=None):
    """fort.25 + .out → navigator payload (HONEST UNWEIGHTED bands).

    Eigenvector arrays are ZERO — fort.25 carries none (see the package
    docstring) — so the viewer's blend falls back to a uniform species mix
    and the animation shows zero displacement instead of invented motion.
    ``meta.conventions.eigenvectors`` states this in the payload itself.
    """
    segs = read_fort25(f25_path)
    st = read_crystal_out_structure(out_path)
    if len(st["path"]) != len(segs):
        raise ValueError(f"path mismatch: .out declares {len(st['path'])} "
                         f"segments, fort.25 carries {len(segs)}")
    n_b = segs[0]["n_bands"]
    n_at = len(st["symbols"])
    if n_b != 3 * n_at:
        raise ValueError(f"{n_b} bands but {n_at} primitive atoms "
                         f"(need 3N = {3 * n_at}) — wrong .out for this f25?")

    recip = np.linalg.inv(st["lattice"]).T          # crystallographic, no 2π
    dist_all, freq_all, qf_all, qc_all = [], [], [], []
    tick_pos, tick_lab, seg_ranges = [], [], []
    run = 0.0
    for si, (seg, (q_from, q_to)) in enumerate(zip(segs, st["path"])):
        n_p = seg["n_points"]
        # consistency: the f25 block's own integer endpoints match the .out's
        t = np.linspace(0.0, 1.0, n_p)[:, None]
        qf = np.array(q_from)[None, :] * (1 - t) + np.array(q_to)[None, :] * t
        qc = qf @ recip
        step = np.linalg.norm(np.diff(qc, axis=0), axis=1)
        d = run + np.concatenate([[0.0], np.cumsum(step)])
        if si == 0:
            tick_pos.append(0.0)
            tick_lab.append(_label_for(q_from))
        tick_pos.append(float(d[-1]))
        tick_lab.append(_label_for(q_to))
        seg_ranges.append([float(d[0]), float(d[-1])])
        run = float(d[-1])
        dist_all.append(d)
        freq_all.append(seg["freq_thz"])
        qf_all.append(qf)
        qc_all.append(qc)

    dist = np.concatenate(dist_all)
    freq = np.vstack(freq_all)
    qf = np.vstack(qf_all)
    qc = np.vstack(qc_all)
    n_q = len(dist)
    zeros_e = np.zeros((n_q, n_b, n_at, 3))

    payload = {
        "meta": {
            "material": {
                "symbols": st["symbols"],
                "masses": [float(m) for m in st["masses"]],
                "lattice": st["lattice"].tolist(),
                "reciprocal_lattice": recip.tolist(),
                "fractional_coords": st["fractional_coords"].tolist(),
                "dimensionality": st["dimensionality"],
                "species_style": {
                    s: {"color": list(jmol_color(s)),
                        "radius": float(covalent_radius(s))}
                    for s in dict.fromkeys(st["symbols"])
                },
            },
            "conventions": {
                "frequency": "signed THz (fort.25 cm^-1 / THZ_TO_CM1)",
                "eigenvectors": "ZERO — CRYSTAL fort.25 carries frequencies "
                                "only; atom weights and mode animation are "
                                "unavailable from this format. The blend "
                                "colour is the viewer's uniform fallback, "
                                "not physics.",
                "masses": "the run's own isotopic masses (.out table)",
            },
            "source": {
                "origin": "crystal23-fort25",
                "arbitrary_q": False,
                "arbitrary_q_reason": "band-plot file: path frequencies only; "
                                      "no dynamical matrix in this format",
                "supercell_atoms": st["n_supercell_atoms"],
                "expansion_diag": st["expansion_diag"],
            },
        },
        "band": {
            "distance": dist.tolist(),
            "segments": seg_ranges,
            "labels": [[tick_lab[i], tick_lab[i + 1]]
                       for i in range(len(seg_ranges))],
            "tick_positions": tick_pos,
            "breaks": [],
            "high_symmetry_points": [
                {"label": lab, "distance": float(pos)}
                for lab, pos in zip(tick_lab, tick_pos)],
            "frequency": freq.tolist(),
            "character": np.zeros((n_q, n_b)).tolist(),
            "q_cart": qc.tolist(),
            "q_frac": qf.tolist(),
            "eigenvectors_real": zeros_e.tolist(),
            "eigenvectors_imag": zeros_e.tolist(),
        },
    }
    if output:
        text = json.dumps(payload, separators=(",", ":"))
        with open(output, "w") as fh:
            fh.write(("window.NAV_DATA = " + text + ";\n")
                     if output.endswith(".js") else text)
    info = {"n_q": int(n_q), "n_bands": int(n_b),
            "segments": [s["n_points"] for s in segs],
            "path_labels": tick_lab, "output": output}
    return payload, info
