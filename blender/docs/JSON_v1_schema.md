# Phonon Vision — JSON v1 Exchange Schema (design)

**Status:** Design draft, no code yet. This document is for review before
we commit to a wire format. Once you sign off (or push back), I'll
implement it as `phonon_core.io.json_v1` with parallel writers and
readers.

**Goals**

1. Single source of truth for "the phonon data for a material" usable by:
   - The Blender addon (already imports `phonon-core`)
   - The future web viewer (a static Vite + react-three-fiber site)
   - Any third-party tool — phononwebsite, Materials Cloud, etc.
2. Cross-tool compatibility: our JSON should be loadable by
   [phononwebsite](https://github.com/henriquemiranda/phononwebsite/)
   and vice-versa, modulo extensions we add.
3. Future-proof: room for thermal ellipsoids, Born charges, polarization
   vectors, dispersion surfaces, IPR, projection weights — without
   reformatting on every addition.

---

## 1. The format split — base + extensions

A `phonon-vision JSON v1` document is two layers stitched together at the top
level:

```jsonc
{
  // Layer 1: base — strictly compatible with phononwebsite's schema
  "name": "WSe2",
  "natoms": 3,
  "lattice": [[...], [...], [...]],
  "atom_types": ["W", "Se", "Se"],
  "atom_numbers": [74, 34, 34],
  "atomic_numbers": [74, 34, 34],
  "chemical_symbols": ["W", "Se", "Se"],
  "formula": "WSe2",
  "qpoints":     [[0,0,0], [0.01, 0, 0], ...],   // fractional
  "repetitions": [3, 3, 1],                       // suggested supercell
  "atom_pos_car": [[...], [...], [...]],          // Cartesian, Å
  "atom_pos_red": [[...], [...], [...]],          // fractional
  "eigenvalues": [[...], [...], ...],             // shape (N_q, N_bands), cm⁻¹
  "vectors":     [[[[[...]]]]],                   // (N_q, N_bands, N_atoms, 3, 2) [Re,Im]
  "highsym_qpts": {"0": "Γ", "33": "M", ...},
  "line_breaks":  [[0, 32], [33, 65], ...],       // segment q-index ranges
  "distances":    [...],                          // (N_q,), Å⁻¹

  // Layer 2: extensions — namespaced, optional. Tools that don't
  // understand them MUST silently ignore them per RFC.
  "phonon_vision_extensions": {
    "schema_version": "1.0",
    "source": {
      "phonopy_version": "2.21.2",
      "calculator": "vasp",
      "born_file_present": false,
      "force_constants_file_present": true,
      "calculation_date_utc": "2026-05-15T03:42:00Z"
    },
    "conventions": {
      "frequency_unit": "cm-1",
      "eigenvector_norm": "mass_weighted",
      "imag_freq_convention": "negative",
      "canonical_phase_applied": false,
      "mass_weighting_baked_in": true
    },
    "thermal_displacements": null,
    "born_charges": null,
    "dielectric_constant": null,
    "polarization_vectors_gamma": null,
    "ipr": null,
    "species_weights": null,
    "license": "CC-BY-4.0",
    "citation": "Doe et al., PRB 2025"
  }
}
```

**Strict rule:** every base-layer field MUST be present and match
phononwebsite's interpretation byte-for-byte. The base layer is not a
place where we innovate — innovation goes in the extensions block.

That means if a user drops our JSON into the phononwebsite hosted viewer
at `henriquemiranda.github.io/phononwebsite/`, it just works. And if they
drop a phononwebsite JSON into our tools, the extensions block is absent
(treated as `{}`) and everything falls back to base behavior.

---

## 2. Base layer — the phononwebsite-compatible block

Field by field, with the **convention** we'll use when writing and the
**failure mode** if a reader gets nonsense:

| Field | Type | Units | Convention | Failure mode |
|---|---|---|---|---|
| `name` | string | — | chemical formula or user-given title | reader uses "Untitled" |
| `natoms` | int | — | primitive-cell atom count | reader recomputes from `atom_types` |
| `lattice` | (3, 3) | Å | rows are a₁, a₂, a₃ (Cartesian) | reader rejects with clear error |
| `atom_types` | list[str] of len natoms | — | chemical symbols, e.g. ["Sn", "Se", "Se"] | reader rejects |
| `atomic_numbers` | list[int] of len natoms | — | Z per atom | reader can compute from symbols |
| `chemical_symbols` | list[str] | — | same as `atom_types`, kept for compat | optional duplicate |
| `formula` | string | — | "WSe2", "SnSe2", etc. | reader can compute |
| `qpoints` | (N_q, 3) | fractional reciprocal | **fractional** in primitive-cell reciprocal basis, no 2π | reader rejects if shape wrong |
| `repetitions` | (3,) int | — | suggested supercell for visualization (3,3,3 default) | reader can default |
| `atom_pos_car` | (N_atoms, 3) | Å | Cartesian | reader recomputes from `atom_pos_red` |
| `atom_pos_red` | (N_atoms, 3) | fractional | in primitive lattice | reader rejects if missing both |
| `eigenvalues` | (N_q, N_bands) | **cm⁻¹** signed | negative = imaginary, matches phononwebsite | reader rejects if shape wrong |
| `vectors` | (N_q, N_bands, N_atoms, 3, 2) | — | complex stored as [Re, Im] pairs; mass-weighted-normalized so Σ\|e\|² = 1 | reader rejects if shape wrong |
| `highsym_qpts` | dict[str(int), str] | — | maps q-index (as string!) to label; LaTeX-style "Γ" with no $...$ wrappers | reader can default to {} |
| `line_breaks` | (N_seg, 2) | — | segment q-index ranges [start, end] inclusive | reader treats as single segment |
| `distances` | (N_q,) | Å⁻¹ | cumulative path distance from start | reader can default to range(N_q) |

**Frequency unit choice.** phononwebsite uses cm⁻¹. We'll convert on write:
internally `phonon-core` stores THz (matches phonopy), and write to cm⁻¹
via `units.thz_to_cm1`. The conventions block records which unit is used
in the JSON so a reader can re-convert.

**Eigenvector layout.** Phononwebsite stores `(N_q, N_bands, N_atoms, 3, 2)`,
where the last axis is `[Re, Im]`. We'll match that exactly. Our internal
`(N_q, N_bands, N_atoms, 3) complex128` becomes:

```python
v = eigenvectors.view(np.float64).reshape(..., 2)  # zero-copy view
```

…modulo memory layout; in practice a `np.stack([real, imag], axis=-1)` is
safer (handles arbitrary array memory layouts).

**Canonical phase.** This is the **biggest interop question**.
Phononwebsite's renderer assumes raw phonopy eigenvectors (with the
arbitrary global phase from numerical diagonalization). Our
`phonon-core.canonicalize_eigenvector_phase` rotates eigenvectors so the
largest-magnitude component is real and positive — making snapshots
reproducible (see Dove §3.5; Maradudin & Vosko 1968 §2).

Option A: write canonicalized eigenvectors, record `"canonical_phase_applied": true`
in extensions. Phononwebsite still renders the animation correctly
(canonicalization commutes with the full-period animation). But other
tools that compare instantaneous snapshots may see different values
than from a raw phonopy file.

Option B: write raw eigenvectors, set `"canonical_phase_applied": false`.
Phononwebsite-compatible but loses our reproducibility benefit unless
the reader re-canonicalizes.

**Proposal: Option A.** Document the convention prominently; the
canonicalization is a one-liner if a reader wants to undo it. The benefit
(reproducible static snapshots across phonopy runs) is genuine and worth
the small interop note.

---

## 3. Extensions layer — `phonon_vision_extensions`

All fields optional. A reader walks the block and consumes what it
recognizes. Unknown fields ignored, **never** an error.

### 3.1 `source` — provenance

```jsonc
"source": {
  "phonopy_version": "2.21.2",     // for bug-chasing
  "calculator": "vasp",             // 'vasp', 'qe', 'crystal', 'abinit'
  "born_file_present": false,       // was a phonopy BORN file fed in?
  "force_constants_file_present": true,
  "calculation_date_utc": "2026-05-15T03:42:00Z"  // for reproducibility audit
}
```

### 3.2 `conventions` — what was done to the data

```jsonc
"conventions": {
  "frequency_unit": "cm-1",        // 'cm-1' | 'THz' | 'meV'  — what eigenvalues are in
  "eigenvector_norm": "mass_weighted",  // 'mass_weighted' | 'cartesian'
  "imag_freq_convention": "negative",   // phonopy emits negative for soft modes
  "canonical_phase_applied": true,  // rotated so largest component real+positive
  "mass_weighting_baked_in": true,  // already 1/sqrt(m_s) divided in eigenvectors
  "supercell_in_repetitions": [3, 3, 3]
}
```

### 3.3 Optional physics fields — schema reservations for the future

These slots are **reserved now, populated later**. Keeping them in the
schema means we don't have to bump the version when we implement
thermal ellipsoids etc.

```jsonc
"thermal_displacements": {
  "temperatures_K": [300.0, 500.0],
  // (N_T, N_atoms, 3, 3) symmetric tensors in Å²
  // ADP convention (Maradudin & Vosko 1968 eq. 5.18; Dove §4.2)
  "u_matrices_angstrom_sq": [[[[...]]]]
},
"born_charges": [
  // (N_atoms, 3, 3) Z*_s tensor, in |e|
  // phonopy's BORN file convention; sum_s Z*_s should be 0 (acoustic sum rule)
  [[1.5, 0, 0], [0, 1.5, 0], [0, 0, -3.0]]
],
"dielectric_constant": [
  // (3, 3) high-frequency dielectric tensor
  [4.5, 0, 0], [0, 4.5, 0], [0, 0, 4.5]
],
"polarization_vectors_gamma": [
  // (N_bands, 3) computed dipole at Γ = sum_s Z*_s · e_s(Γ) / sqrt(m_s), real
  // Zero magnitude = IR-inactive; large = IR-active
  [0.0, 0.0, 0.0], [0.0, 0.0, 0.5], ...
],
"ipr": [
  // (N_q, N_bands) inverse participation ratio per mode
  // 1/N_atoms = delocalized, near 1 = localized on single atom
  [0.33, 0.33, 0.33, ...], ...
],
"species_weights": {
  "species_order": ["W", "Se"],
  // (N_q, N_bands, N_species) projection weights, sum to 1 along last axis
  "weights": [[[0.7, 0.3], [0.3, 0.7], ...], ...]
}
```

### 3.4 Compression hints

For large meshes the JSON gets heavy. Two strategies the schema
supports (writer chooses, reader handles both):

- **Inline.** All arrays as plain JSON lists. Largest file size,
  trivially readable in JS.
- **Sidecar.** Big arrays referenced by filename:
  ```jsonc
  "vectors": {
    "@external": "WSe2_vectors.bin",
    "dtype": "complex128", "shape": [606, 9, 3, 3], "order": "C"
  }
  ```
  Reader fetches the binary file. Saves ~3× on the wire for the
  eigenvector array. Optional, only used when explicitly requested.

I'd default to inline for v1 — keep the format simple. Sidecar can be
v1.1 once we have web-side bandwidth concerns.

---

## 4. Validation & error handling

The reader must:

1. Reject files where `lattice`, `qpoints`, `eigenvalues`, or
   `vectors` are missing or shape-inconsistent.
2. Warn but proceed when optional base-layer fields are missing
   (compute them from what's available; e.g., `atom_pos_car` from
   `atom_pos_red @ lattice`).
3. Silently ignore unknown fields in `phonon_vision_extensions`.
4. Pin to `schema_version` "1.0" — refuse `2.x` until we have explicit
   migration support.

The writer must:

1. Always populate all required base-layer fields (recompute if needed).
2. Set `phonon_vision_extensions.schema_version` to the current value.
3. Round-trip cleanly: `read(write(data)) == data` for any data we can
   write.

---

## 5. Module layout in `phonon-core`

```
phonon_core/
  io/
    __init__.py
    json_v1.py        # write_json_v1, read_json_v1
    _phononwebsite_compat.py   # base-layer mapping, kept separate
```

Public API:

```python
from phonon_core.io import write_json_v1, read_json_v1

# Write
write_json_v1(band_path, "WSe2.json",
              canonicalize=True,
              include_ipr=False,
              include_species_weights=True)

# Read
result = read_json_v1("WSe2.json")
# result is a BandPath (if line_breaks present) or a MeshGrid (if not),
# plus the extension dict on result.extensions
```

---

## 6. Open questions

These I want your input on before implementing:

1. **THz or cm⁻¹ in the JSON?** Phononwebsite uses cm⁻¹. Phonopy uses
   THz internally. Spectroscopists publish in cm⁻¹. Going cm⁻¹ is the
   interop-friendly choice. Confirm?

2. **Canonical phase: on or off by default in writes?** I'd default to ON,
   record the fact in conventions. Disagree?

3. **Inline eigenvectors or sidecar binary?** For v1 I'd do inline only.
   Adds maybe 30% file size vs binary but the schema stays simple.
   Acceptable?

4. **Where does the `phonon_vision_extensions` namespace live for
   third-party tools?** Some JSON exchange formats use `x-` prefixes
   (HTTP-style) or `_meta` (CommonMark-style). I prefer the explicit
   namespace name because it's self-documenting. Disagree?

5. **Material identifier — UUID / hash / nothing?** Some users will want
   to reference "this JSON came from the same DFT run as this other
   JSON." A SHA-256 hash of the POSCAR/CIF input would tie things
   together. Optional in v1 or skip entirely?

6. **Schema versioning.** I'd use semver in `schema_version`: `"1.0"`
   for this release, `"1.1"` for additive changes (new extension fields),
   `"2.0"` if we ever break the base layer (which would also break
   phononwebsite compat — let's not). Confirm?

---

## 7. Implementation phases once approved

Phase A: writer + reader for the base layer only. Round-trip tests on
WSe2 and SnSe2.

Phase B: extensions for `source`, `conventions`, `ipr`,
`species_weights`. These derive directly from `phonon-core` modules
already.

Phase C: extensions for `thermal_displacements` once we have a
`phonon_core.thermal` module. Same for `born_charges` once we have
BORN-file parsing.

Phase D: sidecar binary format. Only if we hit a real bandwidth wall on
the web side.
