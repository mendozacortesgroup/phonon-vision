"""QE ``.dyn`` set → navigator payload with ``mesh_modes``.

What this makes possible
------------------------
A ph.x ``.dyn`` set is exact D(q) on the irreducible points of a uniform
grid, star-expanded to the full grid — polar materials included. Baked as a
payload it gives the viewer, for a 3D material, everything that is honest to
give: click-to-animate exact modes, and the Tier-1 THERMAL ELLIPSOIDS, whose
BZ average is licensed by the grid (see below). It deliberately does NOT give
a dispersion curve: 64 grid points are not a band path, and interpolating
between them needs force constants — the ``.fc`` route, which is refused for
polar materials. The 2D panel therefore shows an honest SPECTRUM: frequency
against |q|, drawn as markers only (the panel's native mode — no fake
connectivity is ever drawn).

The equal-weight licence
------------------------
The Tier-1 ADP average assumes every ``mesh_modes`` node carries equal
weight. For a star-expanded ``.dyn`` set that is true exactly when the
expansion covers the FULL uniform grid — each point once, none missing.
:func:`convert_qe_dyn` VERIFIES this (each q must sit on a grid multiple to
1e-6 and the set must be a bijection onto the grid) and refuses otherwise:
an incomplete or duplicated set would silently bias every thermal tensor.
Measured on the real CsSnI3 set: 64/64 cells, q on-grid to 1.8e-09.

Payload shape
-------------
``{meta, band, mesh_modes}`` — no ``surface`` (bands-over-BZ is a 2D-material
concept), no ``dos``, no ``provenance`` engine block (nothing here came from
the A1/A2 engines; ``meta.source`` carries the .dyn provenance instead,
with ``arbitrary_q: false`` and the reason). The viewer is presence-keyed on
every one of these absences; the 0.6.88 dim=3 payload proved the
band-without-surface boot path, and the Tier-1 gate reads ``mesh_modes``
directly.
"""

from __future__ import annotations

import json
import os

import numpy as np

from phonon_core.qe_import.dyn_file import QEDynamical, dyn_modes, read_qe_dyn_set
from phonon_core.radii import covalent_radius, jmol_color

__all__ = ["convert_qe_dyn"]

#: eigenvector components are rounded to this many SIGNIFICANT figures in the
#: mesh_modes block — the 0.6.81 precision policy (size is driven by
#: eigenvector text; frequencies and q always stay full precision).
_EIG_SIG_FIGS = 6


def _round_sig(x, sig=_EIG_SIG_FIGS):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    nz = x != 0
    mag = np.floor(np.log10(np.abs(x[nz])))
    factor = 10.0 ** (sig - 1 - mag)
    out[nz] = np.round(x[nz] * factor) / factor
    return out


def _grid_index(q_frac, grid, tol=1e-6):
    """Map fractional q onto integer grid cells, refusing off-grid points."""
    idx = np.asarray(q_frac) * np.asarray(grid)
    cells = np.rint(idx)
    dev = np.abs(idx - cells).max()
    if dev > tol:
        raise ValueError(
            f"q-points are not on the {tuple(grid)} grid (max deviation "
            f"{dev:.2e} grid units) — cannot claim an equal-weight mesh")
    return np.mod(cells.astype(int), np.asarray(grid))


def convert_qe_dyn(dyn_or_paths, output=None, *, group_a=None):
    """Build (and optionally write) a navigator payload from a ``.dyn`` set.

    Parameters
    ----------
    dyn_or_paths : :class:`QEDynamical` or sequence of paths — the set. Pass
        every file of the calculation (``.dyn0`` is skipped automatically);
        the grid-completeness check below is what makes a partial set safe to
        hand in by accident.
    output : optional path — ``.js`` writes ``window.NAV_DATA = {...};``
        (what the viewer loads), anything else raw JSON.
    group_a : species symbol for the 2D "character" colour channel (the
        group-A |e|^2 weight). Default: the first species in the file.

    Returns
    -------
    (payload dict, info dict) — info carries the facts a caller must be able
    to quote: grid, node count, the frequency-oracle agreement against the
    file's own printed frequencies, and the group-A choice.
    """
    dyn = (dyn_or_paths if isinstance(dyn_or_paths, QEDynamical)
           else read_qe_dyn_set(list(dyn_or_paths)))

    symbols = list(dyn.symbols)
    n_at = len(symbols)
    if group_a is None:
        group_a = symbols[0]
    if group_a not in symbols:
        raise ValueError(f"group_a {group_a!r} not among species {symbols}")

    # ---- the equal-weight licence: full uniform grid, each cell exactly once
    # The grid shape is inferred from the q-set itself (the .dyn0 file is not
    # required): for a Gamma-centred uniform grid the denominators of the
    # fractional q's ARE the grid. Infer per axis as the smallest n with
    # q*n integral, bounded by the node count.
    q = np.asarray(dyn.qpoints_frac, dtype=np.float64)
    n_q = len(q)
    grid = []
    for ax in range(3):
        col = q[:, ax]
        n_ax = None
        for n in range(1, n_q + 1):
            if np.abs(col * n - np.rint(col * n)).max() < 1e-6:
                n_ax = n
                break
        if n_ax is None:
            raise ValueError(f"axis {ax}: q values are not commensurate with "
                             "any uniform grid up to the node count")
        grid.append(n_ax)
    if int(np.prod(grid)) != n_q:
        raise ValueError(
            f"{n_q} q-points do not fill the inferred {tuple(grid)} grid "
            f"({int(np.prod(grid))} cells) — the star expansion is "
            "incomplete or duplicated, and an equal-weight thermal average "
            "over it would be biased. Pass the complete .dyn set.")
    cells = _grid_index(q, grid)
    tags = {tuple(c) for c in cells}
    if len(tags) != n_q:
        raise ValueError("duplicate q-points in the set (same grid cell "
                         "appears twice) — equal weights would double-count")

    # ---- modes, sorted by |q| for the spectrum panel ------------------------
    freq, eig = dyn_modes(dyn)                        # (Nq, Nb), (Nq, Nb, Na, 3)
    # reciprocal rows (crystallographic, NO 2*pi — the project convention;
    # the MP ledger's "do not consume a 2*pi reciprocal" lesson applies here
    # in the writing direction):
    recip = np.linalg.inv(dyn.lattice).T              # rows b_i, 1/Angstrom
    q_cart = q @ recip                                # (Nq, 3) 1/Angstrom
    order = np.argsort(np.linalg.norm(q_cart, axis=1), kind="stable")
    q, q_cart, cells = q[order], q_cart[order], cells[order]
    freq, eig = freq[order], eig[order]

    dist = np.linalg.norm(q_cart, axis=1)

    # group-A character: sum of |e|^2 over group-A atoms, per (q, band)
    ga_mask = np.array([s == group_a for s in symbols])
    w = (np.abs(eig) ** 2).sum(axis=3)                # (Nq, Nb, Na)
    char = w[:, :, ga_mask].sum(axis=2)               # in [0, 1] by Dove norm

    frac = np.mod(dyn.positions @ np.linalg.inv(dyn.lattice), 1.0)

    def split6(e):
        return (_round_sig(e.real).tolist(), _round_sig(e.imag).tolist())

    mm_re, mm_im = split6(eig)

    payload = {
        "meta": {
            "material": {
                "symbols": symbols,
                "masses": [float(m) for m in dyn.masses],
                "lattice": dyn.lattice.tolist(),
                "reciprocal_lattice": recip.tolist(),
                "fractional_coords": frac.tolist(),
                "dimensionality": 3,
                "species_style": {
                    sym: {"color": list(jmol_color(sym)),
                          "radius": float(covalent_radius(sym))}
                    for sym in dict.fromkeys(symbols)
                },
            },
            "conventions": {
                "frequency": "signed THz (nu < 0 = unstable); "
                             "sign(lam)*sqrt(|lam|)*RY_THZ from ph.x D(q)",
                "eigenvectors": "mass-weighted, sum_s |e_s|^2 = 1 (Dove 4.16), "
                                "phase-canonicalized per mode",
                "band_panel": "SPECTRUM, not a path: x = |q| (1/Angstrom), "
                              "markers only — a .dyn grid carries no "
                              "path ordering and none is invented",
                "mesh_modes": "the star-expanded uniform grid; verified "
                              "complete, so equal-weight BZ averages "
                              "(thermal ellipsoids) are licensed",
            },
            "source": {
                "origin": "qe-dyn",
                "arbitrary_q": False,
                "arbitrary_q_reason": "ph.x dynamical matrices exist only at "
                                      "the computed grid q; interpolation "
                                      "would need force constants",
                "polar": bool(dyn.dielectric is not None),
                "grid": [int(g) for g in grid],
                "symbol_sources": {str(k): v for k, v in
                                   (dyn.symbol_sources or {}).items()},
            },
        },
        "band": {
            "distance": dist.tolist(),
            "segments": [[float(dist[0]), float(dist[-1])]],
            "labels": [["Γ", ""]],
            "tick_positions": [0.0],
            "breaks": [],
            "high_symmetry_points": [],
            "frequency": freq.tolist(),
            "character": char.tolist(),
            "q_cart": q_cart.tolist(),
            "q_frac": q.tolist(),
            "eigenvectors_real": eig.real.tolist(),
            "eigenvectors_imag": eig.imag.tolist(),
        },
        "mesh_modes": {
            "grid": [int(g) for g in grid],
            "q_frac": q.tolist(),
            "q_cart": q_cart.tolist(),
            "frequency": freq.tolist(),
            "eigenvectors_real": mm_re,
            "eigenvectors_imag": mm_im,
        },
    }

    # the oracle carried through: payload frequencies vs the file's own
    oracle = None
    if dyn.file_frequencies is not None and dyn.frequency_index is not None:
        # frequency_index refers to pre-sort rows; map through the sort
        inv_order = np.argsort(order, kind="stable")
        rows = inv_order[dyn.frequency_index]
        ours = np.sort(freq[rows], axis=1)
        oracle = float(np.abs(ours - np.sort(dyn.file_frequencies, axis=1)).max())

    if output:
        text = json.dumps(payload, separators=(",", ":"))
        with open(output, "w") as fh:
            if output.endswith(".js"):
                fh.write("window.NAV_DATA = " + text + ";\n")
            else:
                fh.write(text)

    info = {
        "n_q": int(n_q),
        "n_bands": int(freq.shape[1]),
        "grid": [int(g) for g in grid],
        "group_a": group_a,
        "polar": bool(dyn.dielectric is not None),
        "freq_oracle_thz": oracle,
        "output": output,
    }
    return payload, info
