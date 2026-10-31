# phonon-core JSON v1 Exchange Schema

**Status: v1.0 LOCKED.** This supersedes the earlier design draft and resolves
all of its open questions (see §8). The base layer is implemented in
`phonon_core.io.json_v1` and validated by `docs/json_v1_schema.json`. Phase A
covers Material + BandPath + MeshGrid; extension *payloads* are reserved (names
only — §7), not implemented.

A JSON v1 document describes the phonon data for **one material**: a `Material`,
an optional `BandPath` (dispersion along a q-path), an optional `MeshGrid` (a
q-grid), and free-form metadata. It is meant to be the single interchange format
for the Blender add-on, the future web viewer, and third-party tools.

---

## 1. Document model

```jsonc
{
  "schema_version": "1.0.0",
  "generator":   { "name": "phonon-core", "version": "0.1.7" },
  "conventions": { /* §3 — MANDATORY, all 8 keys */ },
  "metadata":    { /* free-form, optional */ },
  "core": {
    "material": { /* §4 */ },
    "band":     { /* §5 */ } | null,
    "mesh":     { /* §6 */ } | null
  },
  "extensions": { /* §7 — reserved names, ignore unknown keys */ }
}
```

One file = one material. `core.band` and `core.mesh` are independently optional
(either, both, or neither). `core.band.material` and `core.mesh.material` are the
same `core.material`; the material is serialized once.

---

## 2. Versioning (dual)

- **`schema_version`** — top-level semver string, `"1.0.0"` for this release.
  Readers **accept** unknown MINOR/PATCH within MAJOR 1 (forward-compatible:
  `"1.9.0"` reads fine, treating unknown additions as ignorable) and **reject**
  an unknown MAJOR (`"2.0.0"`) with a clear error rather than guessing at a
  breaking change.
- **`generator`** — `{"name": "phonon-core", "version": "<x.y.z>"}` provenance
  of the writer.

---

## 3. Conventions block — MANDATORY

Every file MUST carry a complete `conventions` block; a reader that finds it
missing or incomplete fails with a clear error (it will not guess units). The
values below are fixed by phonon-core's data model except `source_code` /
`source_version`, which are caller-supplied provenance.

```jsonc
"conventions": {
  "frequency_units": "THz",                  // BandPath/MeshGrid frequencies
  "eigenvector_normalization": "mass_weighted", // Σ|e_s|²=1 (Togo & Tanaka 2015)
  "q_point_format": "fractional",            // fractional reciprocal-lattice coords
  "lattice_vectors": "rows",                 // each ROW is a lattice vector (phonopy)
  "dynamical_matrix_fourier": null,          // reserved: "with_tau" | "without_tau"
  "nac_method": "none",                      // reserved: "gonze" | "wang"
  "source_code": "phonopy",                  // upstream code that produced the data
  "source_version": null                     // string | null
}
```

**Citations.** Lattice/Bloch conventions follow Maradudin & Vosko, *Rev. Mod.
Phys.* **40**, 1 (1968); the mass-weighted eigenvector normalization (dividing
the polarization vector by √mₛ so Σ|e_s|² = 1) follows Togo & Tanaka, *Scr.
Mater.* **108**, 1 (2015) — the phonopy convention. Frequencies are signed THz
(negative = imaginary/soft mode), matching phonopy's native output.

---

## 4. `core.material`

| Field | Type | Units | Notes |
|---|---|---|---|
| `lattice` | 3×3 float | Å | rows are a₁,a₂,a₃ |
| `symbols` | list[str] | — | chemical symbol per atom |
| `masses` | list[float] | AMU | one per atom |
| `fractional_coords` | (N,3) float | — | in [0,1) (see `positions_were_wrapped`) |
| `positions_were_wrapped` | bool | — | True iff the loader wrapped input into [0,1) |

`reciprocal_lattice` is **not** stored: it is derivable and is recomputed on
read as B = (A⁻¹)ᵀ (phonopy reciprocal, no 2π). The recomputed value differs
from a phonopy-original by ~1e-9 (finite stored precision), which is acceptable
because it is not a v1 schema field.

---

## 5. `core.band` (BandPath; optional)

| Field | Type | Units | Notes |
|---|---|---|---|
| `storage` | str | — | `"inline"` (only implemented value); `"sidecar"` reserved |
| `q_points` | (N_q,3) float | fractional | path q-points |
| `distances` | (N_q,) float | Å⁻¹ | cumulative path distance |
| `segment_nqpoint` | list[int] | — | q-count per high-symmetry segment (Σ = N_q) |
| `labels` | list[[str,str]] | — | (start,end) label per segment, e.g. ["Γ","M"] |
| `frequencies` | (N_q,N_branch) float | THz | signed |
| `eigenvectors_real` | (N_q,N_branch,N_atom,3) float \| null | — | Re part (§9) |
| `eigenvectors_imag` | same shape \| null | — | Im part (§9) |

`N_branch = 3·N_atom`. Eigenvectors are optional (both null) but if present BOTH
real and imag must be present with identical shape.

---

## 6. `core.mesh` (MeshGrid; optional)

| Field | Type | Units | Notes |
|---|---|---|---|
| `storage` | str | — | `"inline"`; `"sidecar"` reserved |
| `mesh` | [int,int,int] | — | grid dimensions (nx,ny,nz) |
| `q_points` | (N_q,3) float | fractional | |
| `weights` | list[int] \| null | — | symmetry weights (all 1 if unfolded) |
| `frequencies` | (N_q,N_branch) float | THz | |
| `eigenvectors_real` / `eigenvectors_imag` | (N_q,N_branch,N_atom,3) float \| null | — | §9 |

---

## 7. `extensions` — reserved namespace (NOT implemented in Phase A)

`extensions` is an object of named, optional, independently-versioned future
payloads. Phase A writes `{}`; base-layer readers MUST ignore unknown keys here
(forward compatibility). Reserved names (this list IS the forward-compat
contract — adding one of these later does NOT bump the major version):

| Key | Intended payload |
|---|---|
| `dynamical_matrix` | D(q) (uses `conventions.dynamical_matrix_fourier`) |
| `nac` | `born_charges` (N,3,3), `dielectric_tensor` (3,3) (uses `nac_method`) |
| `group_velocities` | ∂ω/∂q per mode |
| `dos` | phonon density of states |
| `thermodynamics` | F, S, Cv vs T |
| `thermal_displacements` | U_ij ADP tensors (Maradudin & Vosko 1968) |
| `irreps` | irreducible representation labels per mode |
| `ipr` | inverse participation ratio per mode |
| `species_weights` | per-species projection weights |

---

## 8. Phononwebsite interop (0.1.6)

Interop lives entirely in adapters; the `core` schema (§1–§7) is unchanged. The
`core` block is an **information superset** of
[phononwebsite](https://github.com/henriquemiranda/phononwebsite/)'s JSON.

### 8.1 Exporter — `to_phononwebsite_json(material, bandpath)`

Emits **exactly** phononwebsite's *internal* format field set and nothing else,
so `phononjson.js::getFromJson` (lines 681-690) routes it to
`getFromInternalJson` — it carries no `@class` (pymatgen branch) and not the
`qpoints`+`frequencies`+`eigendisplacements`+`structure` quartet (Materials
Project OpenData branch). Field set verified against `phonon.py::write_json`
(lines 219-232); `line_breaks` is additionally kept (optional, read by
`phononjson.js::getLineBreaks`, line 1063).

| phononwebsite field | source | conversion |
|---|---|---|
| `name`, `formula` | symbols | Hill-style count formula |
| `natoms`, `atom_types`, `atom_numbers` | symbols | Z from periodic table |
| `lattice`, `atom_pos_car`, `atom_pos_red` | material | Å; Cartesian = frac @ lattice |
| `qpoints`, `distances` | bandpath | fractional q |
| `eigenvalues` | frequencies | **THz → cm⁻¹** (×33.35641) |
| `vectors` | eigenvectors | `[n_q][n_branch][n_atom][3][2]`, last axis `[Re,Im]`, **mass pre-scaled** (§8.3) |
| `highsym_qpts` | segments+labels | list of `[int q_index, str label]` pairs |
| `line_breaks` | segments | list of `[start,end]` inclusive q-index ranges |
| `repetitions` | — | default view hint (3,3,3) |

The "bohr" comment on `lattice` in `phonon.py` is stale — the loader and the
real `data/localdb/*/data.json` files treat `lattice` as Å.

### 8.2 Importer — `from_phononwebsite_json(source)`

Reads phononwebsite's **internal** format into our dataclasses
(`PhononDocument` with `material` + `band`). Converts `eigenvalues` cm⁻¹ → THz,
`vectors` `[…][3][2]` → complex, keeps `qpoints` fractional. Masses are looked
up from `atom_numbers` via phononwebsite's own `atomic_mass` table
(`atomic_data.js`), so they match the masses phononwebsite uses. Segments are
reconstructed from `highsym_qpts` (half-open boundaries, last segment takes the
remainder, Σ = N_q); tick grouping at shared boundaries is therefore
approximate while q-points/frequencies/eigenvectors are exact — this importer
targets visualization, not a byte-exact path round-trip. The pymatgen
(`@class`) and OpenData formats are detected (same routing as `getFromJson`) and
raise `NotImplementedError` — flagged as a follow-up rather than mis-parsed.

### 8.3 Mass convention (the important one)

phononwebsite's **JSON render path applies no mass factor**: `vibcrystal.js`
displaces atoms by the (per-mode unit-normalized) eigenvector components, and
`phononwebpage.js::getZeroPointAmplitudeAngstrom` (lines 64-85) — its only mass
term — is guarded by `mode_amplitude_convention === 'avg-mass-normalized'` and
sets only an overall scale. Its writer (`phonon.py`) stores **raw** mass-weighted
e_s. Its **YAML** path, however, pre-scales each atom by √(M_avg/m_s)
(`phononyaml.js` lines 259-292) precisely so the no-mass-factor renderer shows
physical motion. Our renderer displaces by e_s/√m_s (Togo & Tanaka 2015).

Therefore the exporter **pre-scales `vectors` by √(M_avg/m_s) per atom** —
identical to phononwebsite's YAML path. `phononjson.js::normalizeEigenvectors`
(lines 281-315) unit-normalizes each mode on load, so the constant √(M_avg)
cancels; the result is e_s/√m_s up to a per-mode constant, i.e. exactly our
physical displacement. The importer does the inverse expectation: internal-format
`vectors` are raw e_s, stored as-is as our native mass-weighted eigenvectors (our
renderer applies 1/√m_s). **Consequence:** do not re-import our own export — it
is pre-scaled, and re-importing would double-apply the mass factor. The export is
an adapter presentation transform; core eigenvectors and kinematics are untouched.

The export also **declares** this convention explicitly, exactly as
phononwebsite's own YAML loader does after scaling (`phononyaml.js` lines
291-292): `mode_amplitude_convention = "avg-mass-normalized"` and
`average_mass = mean(masses)` (AMU). These are purely declarative — they add no
further per-atom scaling — and they unlock
`phononwebpage.js::getZeroPointAmplitudeAngstrom` (lines 64-85) for our files,
giving a physically-meaningful default display amplitude. There is no
double-scaling risk: the per-atom re-scaler
`applyModeAmplitudeConventionToEigenvectors` (`phononjson.js` line 465) also
requires `dynamical_matrix.masses` (we emit none) and is called only on the
runtime-compute path (line 199), never on stored vectors read by
`getFromInternalJson`; and we ship no `dynamical_matrix`, so
`canComputeEigenvectorsOnDemand()` (line 352) is false.

This is a **one-way** export (added v1 info — masses, wrap flag — is dropped);
the importer targets phononwebsite-native files.

---

## 9. Eigenvectors — real/imag split

JSON has no complex type. Eigenvectors are stored as two nested float arrays,
`eigenvectors_real` and `eigenvectors_imag`, each shaped
`[n_q][n_branch][n_atom][3]`. This maps directly onto `Float32Array`s for the
future three.js web viewer. On read they recombine to `complex128` exactly.

**Losslessness.** Floats are written with the stdlib `json` default, which uses
Python's round-trippable `repr` (since 3.1, `float(repr(x)) == x` for all
float64). Therefore `read(write(x))` reproduces every eigenvector real/imag
value **bitwise** — verified on WSe2 and SnSe2 in `tests/test_json_v1.py`.

---

## 10. Reader / writer behavior

**Writer** (`write_json_v1`): always emits `schema_version`, `generator`, a
complete `conventions` block, `core.material`, and `extensions: {}`; band/mesh
when present. `indent=None` by default (compact, cheap to parse).

**Reader** (`read_json_v1`): raises `JsonV1Error` on (a) missing/non-string
`schema_version` or unknown MAJOR, (b) missing/incomplete `conventions`, (c)
`storage: "sidecar"` (reserved, not implemented), (d) missing required `core`
fields, (e) eigenvectors with only one of real/imag. Returns a `PhononDocument`
(`material`, optional `band`, optional `mesh`, `metadata`).

---

## 11. Resolved open questions (from the draft)

1. **THz vs cm⁻¹ in-file** → **THz** (`conventions.frequency_units`). cm⁻¹ only
   appears in the one-way phononwebsite export.
2. **Canonical phase on write** → **not applied**. v1 stores eigenvectors as-is
   for a lossless, bitwise round-trip; phase canonicalization is a render-time
   choice, not a storage transform, and is not part of v1.
3. **Inline vs sidecar** → **inline only**; `"sidecar"` reserved (§5/§6).
4. **Extension namespace** → top-level `extensions` object (§7).
5. **Material id / hash** → **resolved (0.1.6, relayed decision):** an optional
   `metadata.material_id` (string) is stored **verbatim and uninterpreted** — no
   hash scheme in 1.0. It rides in the free-form `metadata` object and
   round-trips like any other metadata; readers do not validate or compute it.
6. **Semver** → §2.

---

## 12. Interpretation notes

These were under-specified in the locked decisions; the implementation made the
minimal defensible choice and records it here rather than inventing contract:

- **"core" block layout.** Decision 6 says the *core block* supersets
  phononwebsite (reusing their field names), while decisions 4 and 7 dictate
  explicit names (`symbols`, `fractional_coords`, `eigenvectors_real/imag`,
  `frequencies`) that do **not** match phononwebsite (`atom_types`,
  `atom_pos_red`, `vectors`, `eigenvalues`). Resolved by treating decisions 4/7
  as authoritative for the in-file `core` layout and using `to_phononwebsite_json`
  as the compatibility bridge — i.e. "superset" means *information* superset, not
  *field-name* identity. This is the adopted reading.
- **`reciprocal_lattice` recomputed, not stored** (§4) — consistent with decision
  7's Material field list, accepted ~1e-9 deviation from phonopy original.
- **`repetitions` in the phononwebsite export** defaults to (3,3,3) (their own
  default); the v1 document does not carry a supercell hint, so it cannot be
  derived from the data.

---

## Appendix A. `extensions.engine` (added 1.1.0) — evaluation provenance

**Status: ADDITIVE (schema 1.1.0, phonon-core 0.6.79, Level-2 Stage 2 Bump 1).
The v1.0 core (§1–§12) is LOCKED and unchanged; this appendix only populates
the `extensions` namespace §7 reserved.** `engine` joins the reserved-names
forward-compat contract the same way the §7 table entries do: adding it is a
MINOR bump (1.0.0 → 1.1.0), never a major one.

### Semantics — the single-source declaration

`extensions.engine` states, machine-readably, that this document's `core.band`
and/or `core.mesh` (whichever are marked `evaluated: true`) were **computed by
the named engine from the named force constants** — not parsed from two
unrelated phonopy output files. Band-within-mesh exactness (the Level-2 goal)
is therefore a provenance-backed claim: same engine `kind`, same `fc_md5` ⇒
same D(q) behind both blocks. The arc: Stage 1 shipped the engine itself
(`engine/dq_phonopy.py`, 0.6.78, kind `"phonopy-A1"`; the A2 numpy engine will
declare `"numpy-A2"`); Bump 2 (0.6.80) makes the exporter emit this block.

### Wire example

```json
"extensions": {
  "engine": {
    "kind": "phonopy-A1",
    "evaluated": {"band": true, "mesh": true},
    "phonopy_version": "4.3.1",
    "fc_md5": "0123456789abcdef0123456789abcdef",
    "phonopy_yaml_md5": null
  }
}
```

### Keys

| Key | Required | Type | Meaning |
|---|---|---|---|
| `kind` | **yes** | non-empty string | Engine identifier (`"phonopy-A1"`; later `"numpy-A2"`). |
| `evaluated` | **yes** | object `{band: bool, mesh: bool}` | Which core blocks were engine-evaluated. |
| `phonopy_version` | no | string \| null | phonopy version driving the engine. |
| `fc_md5` | no | 32 lowercase hex \| null | md5 of the `FORCE_CONSTANTS` file. |
| `phonopy_yaml_md5` | no | 32 lowercase hex \| null | md5 of the `phonopy.yaml`. |

### Compatibility (both directions, tested)

- **1.0.0 files remain valid**: `extensions` absent/`{}`/engine-less reads back
  with `engine = None`. Other `extensions` keys keep the untouched v1.0
  contract (ignored).
- **Old readers ignore the block**: a 1.0.0-era reader accepts a 1.1.0 file and
  skips `extensions.engine` entirely (§7 readers MUST ignore unknown keys) —
  proven by the forward-compat negative control in the 0.6.79 CHANGELOG.
- **Minor-compat inside the block**: readers are STRICT on the required
  keys/types above but IGNORE unknown extra keys inside `engine`, so a future
  1.2.0 file that grows the block does not break 1.1.0 readers. Malformed
  required keys raise `JsonV1Error` naming the offending key.
