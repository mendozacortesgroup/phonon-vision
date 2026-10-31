"""Generate the radius tables consumed by the viewer.

    ~/anaconda3/bin/python tools/gen_radius_tables.py

Writes ``web/radii_data.js``. Run it again to refresh; never edit the output
by hand. The project rule is that derived data is GENERATED from a citable
source and sanity-checked, not typed from memory, and a 96-element radius
table is exactly the kind of data where a hand-typed transcription error is
both easy to make and invisible until it silently changes a bond count.

Source
------
pymatgen's ``periodic_table.json`` (Ong et al., Comput. Mater. Sci. 68, 314
(2013)), which itself transcribes:

  * ``Atomic radius``          -- Slater, J. Chem. Phys. 41, 3199 (1964),
                                  with later additions.
  * ``Van der waals radius``   -- Bondi, J. Phys. Chem. 68, 441 (1964),
                                  extended by Alvarez, Dalton Trans. 42,
                                  8617 (2013) for elements Bondi omitted.

Covalent radii are NOT taken from here: phonon-core already carries the
Cordero (2008) set in ``radii.py`` and that table stays the single source, so
bond detection is unaffected by anything this script does.

Why there is no ionic style
---------------------------
A single ionic radius per element does not exist. Shannon's radii are indexed
by oxidation state AND coordination number, and the spread is not a detail:
selenium is 0.64 A as Se(4+), 0.56 A as Se(6+) and 1.84 A as Se(2-). In SnSe2
the selenium is Se(2-), so a per-element average -- pymatgen reports 1.013 A
for Se -- describes no real chemical situation. Emitting it would put a
number on screen that looks authoritative and means nothing. Choosing the
state per structure (by electronegativity, say) is defensible but is a
chemistry inference, not a radius table, so it is left as an explicit
non-goal rather than guessed at.
"""

from __future__ import annotations

import json
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web",
                   "radii_data.js")
Z_MAX = 96          # matches the coverage of phonon-core's Cordero table


def collect():
    try:
        from pymatgen.core import Element
        from pymatgen.core.periodic_table import Element as _E  # noqa: F401
    except ImportError:
        sys.exit("pymatgen is required to GENERATE the table (the generated "
                 "web/radii_data.js has no runtime dependency).\n"
                 "Run with an interpreter that has it, e.g.\n"
                 "  ~/anaconda3/bin/python tools/gen_radius_tables.py")

    import pymatgen.core as pc
    from importlib.metadata import version

    raw = json.load(open(os.path.join(os.path.dirname(pc.__file__),
                                      "periodic_table.json")))

    atomic, vdw, missing = {}, {}, {"atomic": [], "vdw": []}
    for sym, rec in raw.items():
        try:
            z = int(rec["Atomic no"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (1 <= z <= Z_MAX):
            continue

        a = rec.get("Atomic radius")
        v = rec.get("Van der waals radius")
        # pymatgen stores some values as strings with units attached
        a = _as_float(a)
        v = _as_float(v)
        if a is not None:
            atomic[sym] = round(a, 3)
        else:
            missing["atomic"].append(sym)
        if v is not None:
            vdw[sym] = round(v, 3)
        else:
            missing["vdw"].append(sym)

    return atomic, vdw, missing, version("pymatgen")


def _as_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).split()[0])
    except (ValueError, IndexError):
        return None


def sanity_check(atomic, vdw):
    """Spot-check against values that are well known independently.

    If a future pymatgen reshuffles its schema and this script starts reading
    the wrong field, these catch it immediately rather than after the numbers
    reach a figure. Tolerances are loose because the point is to detect a
    wrong FIELD, not to police the third decimal.
    """
    expect_vdw = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
                  "Se": 1.90, "Sn": 2.17, "Ar": 1.88}
    bad = []
    for sym, want in expect_vdw.items():
        got = vdw.get(sym)
        if got is None or abs(got - want) > 0.06:
            bad.append(f"vdW {sym}: got {got}, expected ~{want}")
    # every vdW radius must exceed the atomic radius for the same element
    for sym in sorted(set(atomic) & set(vdw)):
        if vdw[sym] <= atomic[sym]:
            bad.append(f"{sym}: vdW {vdw[sym]} <= atomic {atomic[sym]}")
    if bad:
        sys.exit("SANITY CHECK FAILED:\n  " + "\n  ".join(bad))
    return len(expect_vdw)


def emit(atomic, vdw, missing, pmg_version, n_checked):
    def table(d):
        items = sorted(d.items(), key=lambda kv: kv[0])
        lines, row = [], []
        for sym, val in items:
            row.append(f'"{sym}":{val:g}')
            if len(row) == 8:
                lines.append("    " + ", ".join(row) + ",")
                row = []
        if row:
            lines.append("    " + ", ".join(row))
        return "\n".join(lines)

    body = f"""/* radii_data.js — GENERATED by tools/gen_radius_tables.py. DO NOT EDIT.
 *
 * Regenerate with:
 *     ~/anaconda3/bin/python tools/gen_radius_tables.py
 *
 * Source: pymatgen {pmg_version} periodic_table.json, which transcribes
 *   atomic radii  — Slater, J. Chem. Phys. 41, 3199 (1964)
 *   vdW radii     — Bondi, J. Phys. Chem. 68, 441 (1964), extended by
 *                   Alvarez, Dalton Trans. 42, 8617 (2013)
 * Covalent radii are NOT here: phonon-core's Cordero (2008) table in
 * radii.py remains the single source for those, so bond detection is
 * untouched by this file.
 *
 * There is deliberately no ionic table. Shannon radii are indexed by
 * oxidation state and coordination number — Se is 0.64 A as Se(4+) and
 * 1.84 A as Se(2-) — so a single per-element value describes no real
 * chemistry. See the generator's docstring.
 *
 * Values in Angstrom. Coverage: Z <= {Z_MAX}. Elements absent from a table
 * fall back to the covalent radius at the point of use, never to a
 * silent constant.
 */
(function (global) {{
  "use strict";

  var ATOMIC = {{
{table(atomic)}
  }};

  var VDW = {{
{table(vdw)}
  }};

  global.PVRadii = {{
    atomic: ATOMIC,
    vdw: VDW,
    /* null (not a guess) when the element is absent, so the caller decides
     * the fallback and can say which style it actually used. */
    get: function (style, symbol) {{
      var t = style === "vdw" ? VDW : (style === "atomic" ? ATOMIC : null);
      if (!t) return null;
      var v = t[symbol];
      return (typeof v === "number") ? v : null;
    }},
    styles: ["covalent", "atomic", "vdw"],
    coverage: {{ atomic: {len(atomic)}, vdw: {len(vdw)} }},
    source: "pymatgen {pmg_version} (Slater 1964; Bondi 1964 / Alvarez 2013)"
  }};
}})(typeof globalThis !== "undefined" ? globalThis : this);
"""
    with open(OUT, "w") as fh:
        fh.write(body)

    print(f"wrote {os.path.normpath(OUT)}")
    print(f"  atomic radii : {len(atomic)} elements")
    print(f"  vdW radii    : {len(vdw)} elements")
    print(f"  sanity checks: {n_checked} known values + vdW>atomic ordering, all passed")
    for k, v in missing.items():
        if v:
            print(f"  no {k} value : {' '.join(v)}")


if __name__ == "__main__":
    a, v, miss, ver = collect()
    n = sanity_check(a, v)
    emit(a, v, miss, ver, n)
