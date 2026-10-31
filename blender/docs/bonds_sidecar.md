# Bond-settings side-car (`phonon-vision-bonds`, version 1)

Portable import/export of the navigator's **curated bond types** — the editable
bond-type table (pair cutoffs, deletions) plus per-pair radius scales and custom
colors — as a small standalone JSON file. Introduced at 0.6.69.

This is deliberately **not** part of `json_v1` or the navigator payload: bond
curation is presentational (no physics reads it), the viewer cannot write files
under `file://` (so in-payload persistence could never round-trip from the page
anyway), and keeping it a side-car leaves the frozen wire format untouched. The
viewer holds curation in session state only — **never `localStorage`** (the
`file://` portability invariant).

## File format

```json
{
  "format": "phonon-vision-bonds",
  "version": 1,
  "generator": "phonon-vision navigator",
  "material": { "formula": "SnSeSe", "natoms": 3 },
  "units": "angstrom",
  "bonds": {
    "Se-Sn": { "max": 3.11, "radius_scale": 1.5, "color": "#aa33cc" },
    "Se-Se": { "max": 2.88 }
  }
}
```

| Field | Type | Meaning |
|---|---|---|
| `format` | string | Must be exactly `"phonon-vision-bonds"`. Anything else → the file is rejected, state untouched. |
| `version` | number | Format version (1). Not enforced on import; unknown fields elsewhere are ignored (forward compatibility). |
| `generator` | string | Informational. The navigator writes `"phonon-vision navigator"` (the page has no version string available client-side — the payload carries none). |
| `material.formula` | string | The loaded payload's `meta.material.symbols` joined (e.g. `"SnSeSe"`). Loose binding only — see below. |
| `material.natoms` | int | Number of basis atoms. Loose binding only. |
| `units` | string | Always `"angstrom"`. |
| `bonds` | object | The **complete** curated bond-type table, keyed by species pair. |

Per-pair fields (all optional except `max` in practice; unknown fields ignored):

| Field | Type | Meaning |
|---|---|---|
| `max` | number (Å) | **Absolute** distance cutoff for this pair — drives both bond detection and the live-break cutoff (`bondTypes[].max`). Clamped to [0.1, 10] on import (the UI's own clamp). If absent, the covalent default `BOND_TOL·(r_A + r_B)` is used. |
| `radius_A` | number (Å) | **AUTHORITATIVE (0.6.76+): absolute bond cylinder radius in Å.** Emitted for every pair on export. Clamped to [0.02, 1.0] on import. When present, `radius_scale` is ignored. Additive optional key — format stays `version: 1` under the unknown-keys-ignored rule, so pre-0.6.76 readers skip it harmlessly. |
| `radius_scale` | number | **LEGACY (derived).** Exporters (0.6.76+) always emit `radius_scale = radius_A / 0.15` (0.15 Å = the navigator's default cylinder radius constant, `DEFAULT_BOND_RADIUS_A`) so 0.7.9-era Blender — which reads only `radius_scale` — keeps working. On import it is used only when `radius_A` is absent: the effective radius is `clamp(0.3, 3.0, radius_scale) × 0.15 Å`. A pair with neither key follows the importer's live Å default. |
| `color` | `"#rrggbb"` | Custom bond color (`bondCustomCol` + `bondColorMode: "custom"`). Absent = default split endpoint-species coloring. |

## Cross-tool display-geometry convention (0.6.76)

All display geometry is denominated in **absolute Angstrom** in both tools
(navigator scene units are Å 1:1; the Blender add-on builds in Å):

- **Balls**: display radius = **0.34 × the Cordero covalent radius** (Cordero
  et al., *Dalton Trans.*, 2008 — the same table the payload's `species_style`
  carries), rounded to 2 decimals in the UI. Shared **by value** with the
  Blender extension: Sn 0.47 Å, Se 0.41 Å. The navigator's `BALL_A_PER_COV =
  0.34` and Blender's ball factor are the same cross-tool constant — changing
  either alone breaks geometric congruence.
- **Bond cylinders**: absolute Å, default `DEFAULT_BOND_RADIUS_A = 0.15 Å`
  (both the navigator's global default and the divisor in the `radius_scale`
  derivation above).

Because both tools now agree on ball and cylinder radii in Å, a side-car
renders with the **same proportions** everywhere — e.g. a 0.45 Å Se–Sn
cylinder visibly engulfs the 0.47 Å Sn balls in the navigator exactly as it
does in a Blender render. (Before 0.6.76 the navigator drew balls at 0.5 ×
Cordero and treated per-pair radii as multipliers of an unstated base, so one
file produced different proportions per tool.)

## Canonicalization rule

Pair keys are the two species symbols **lexicographically sorted**, joined with
an ASCII hyphen: `"Se-Sn"`, never `"Sn-Se"`. On import, keys are re-canonicalized
regardless of the order in the file (an en-dash separator is also tolerated), so
a `"Sn-Se"`-keyed file and a `"Se-Sn"`-keyed file produce identical state.
**Index-keying is forbidden**: atom indices break under supercell expansion and
cross-importer atom ordering; species pairs do not.

Internally the viewer keys its per-pair maps with an en-dash (`Se–Sn`, the
legend's display convention); the file always uses the ASCII hyphen.

## Distances are absolute

Every distance in the file is an **absolute cutoff in Å** — never a multiple of
`BOND_TOL` or of Cordero covalent radii. The *defaults* the viewer seeds are
derived from Cordero et al. (Dalton Trans., 2008) radii × `BOND_TOL = 1.2`, but
the file stores only the resulting absolute value, so the file's meaning does
not depend on the viewer's internals or radius-table version.

## Apply semantics

- **Replace, not merge**: the file's `bonds` object becomes the complete
  bond-type table. A default pair **absent** from the file is **deleted**
  (pair-not-in-table = no bonds detected for it) — this is how deletions
  round-trip. Per-pair radius scales and custom colors are likewise rebuilt
  from the file (stale entries cleared).
- **Loose structure binding**: if `material.formula`/`natoms` mismatch the
  loaded payload, a warning is shown in `#status` and the settings are applied
  anyway (legitimate case: same material, different calculation).
- **Rejection** (state untouched, message in `#status`): unparseable JSON, a
  non-object, a `format` value other than `"phonon-vision-bonds"`, or a missing
  `bonds` table.
- **Unknown keys** (top-level or per-pair) are ignored silently — forward
  compatibility.
- Duplicate pair keys that collide after canonicalization: last one wins
  (matching the Add row's update-in-place behavior).

## Known v1 limitation

An **empty** `bonds` table (`{}`) does not persist "no bond types": the viewer
lazily re-seeds the default table on the next rebuild when the table is empty
(the seeding contract predates this format). Delete pairs individually if you
need "almost none".

Global display toggles (`bondsOn`, live-break, the global bond radius) are not
bond-type curation and are not serialized in format v1.

## UI

Bonds panel → **Export bonds** (downloads `<formula>_bonds.json` via a Blob
object-URL) and **Import bonds** (file input + `FileReader`). Both are thin
wrappers over the pure functions `serializeBondSettings()` /
`applyBondSettings(obj)` (exposed for the jsdom suite via `window.__atoms`).
