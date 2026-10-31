# Phonon Vision — JSON v1 — Decisions (May 18 2026)

This document records the answers to the open questions in
`JSON_v1_schema.md`, so we have a single source of truth when
implementation starts. Read this *together with* the schema doc — the
schema doc still describes the full structure, this just locks in the
parts that were undecided.

## Decisions

### 1. Frequency units in the wire format

**Decision: store in THz. Display unit is a UI preference, converted
client-side. Supported display units: THz (default), cm⁻¹, meV, eV, Ha.**

Rationale:
- Materials Project / pymatgen's canonical `PhononBandStructure` stores
  frequencies in THz; the docstring is explicit: *"frequencies – list of
  phonon frequencies in THz as a numpy array with shape (3·natoms,
  nqpoints)"*. The default phonon plotter writes `ylabel='Frequencies
  (THz)'`. The plotter accepts a `units` argument (`thz`, `ev`, `mev`,
  `ha`, `cm-1`, `cm^-1`) — convention is **store one unit, convert at
  display time**. We mirror that exactly.
- Phonopy native unit is THz.
- Phonondb / Petretto et al. high-throughput phonon JSON records
  (the data behind MP's phonon section) are pymatgen serializations
  → THz.

Note that this diverges from phononwebsite's wire format, which uses
cm⁻¹. The phononwebsite viewer is downstream of pymatgen/phonopy in MP's
own stack — Materials Project converts THz → cm⁻¹ at the moment it hands
data to the viewer, because of the viewer's I/O contract. The cm⁻¹
choice is a quirk of that one viewer, not a community standard.

Display-time conversion factors (compile-time constants):

```js
const FREQ_CONVERSIONS = {
  THz:  { factor: 1.0,        label: "THz",  decimals: 3 },
  cm1:  { factor: 33.35641,   label: "cm⁻¹", decimals: 1 },  // CODATA 2018
  meV:  { factor: 4.13567,    label: "meV",  decimals: 2 },
  eV:   { factor: 0.00413567, label: "eV",   decimals: 4 },
  Ha:   { factor: 1.51983e-4, label: "Ha",   decimals: 5 },
};
```

### 2. Canonical phase

**Decision: ON by default.**

`phonon_vision_extensions.conventions.canonical_phase_applied = true`
will be the standard. Eigenvectors are rotated so the largest-magnitude
component is real and positive (Maradudin & Vosko 1968 §2; Dove §3.5).
This gives reproducible static snapshots across phonopy runs of the
same material.

The `phonon-core.canonicalize_eigenvector_phase` utility will be
called in the writer; if a reader needs raw phonopy phases it can
opt out by setting `canonical_phase=False` on the writer call.

### 3. Sidecar binary

**Decision: inline-only for v1. Sidecar deferred to v1.1.**

If the database grows to thousands of materials and we hit a bandwidth
wall in the web viewer, we'll add a sidecar binary mode that writes
the eigenvector array to a separate `.bin` file referenced via the
`{"@external": "file.bin", "dtype": ..., "shape": ..., "order": "C"}`
descriptor. The v1.1 reader will handle both modes transparently.

### 4. Extension namespace name

**Decision: `phonon_vision_extensions`.**

Self-documenting, no clash with any base-layer key, and the underscore
prefix makes it scan as "this is our extension, not phononwebsite's
base format" at a glance in the JSON.

### 5. Material identifier

**Decision: Include BOTH a UUID and a content hash.**

```jsonc
"material_id": {
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "content_sha256": "a3f8d6...",
  "content_hash_source": "POSCAR"
}
```

- `uuid` — generated at first write of a calculation result. Stable
  across re-runs of the same DFT calculation (writer reuses if
  present), unique across different calculations. Lets the defect
  database refer to "this DFT result" by a stable handle even if the
  POSCAR text changes (e.g., comment-line update).
- `content_sha256` — SHA-256 of the canonical input structure file
  (POSCAR by default; see `content_hash_source`). Lets us detect
  "two calculations starting from the same structure" — useful for the
  defect database where you'll have many similar structures.
- `content_hash_source` — which file was hashed; "POSCAR" is the
  default but could be "CIF" or "phonopy_disp.yaml".

Both fields are optional in the schema — readers must handle absence
gracefully — but the writer populates both by default.

### 6. Schema versioning

**Decision: Semver in the `schema_version` string. Confirmed.**

- `1.0` — this initial release
- `1.1` — additive (new optional extension fields, no base-layer change,
  no required-field change)
- `2.0` — breaking change to base layer or required-field set (we're
  trying very hard to never bump to 2.0)

Reader enforces:
- Refuses to read `0.x` (none exists; reserved for pre-stable)
- Reads `1.x` for any x (ignores unknown extension keys)
- Refuses to read `2.x` until we have a migration path

## New concept: "Units" UI section in the webapp

Parallel to the existing UI sections (Atoms, Supercell), the webapp
will have a **Units** panel where the user picks:

- **Frequency**: THz (default) / cm⁻¹ / meV
- **Distance**: Å (default) / nm / pm / Bohr
- **Q-points**: fractional (default) / Cartesian in 2π/a units

Units affect *display only*. Stored data, calculations, and exports
all use the wire format's units (THz, Å, fractional). The user-chosen
display unit is a UI preference, not stored in the JSON itself.

This parallels how VESTA/Jmol handle colors: a color choice for "all
oxygens" is a viewer preference, not a property of the data. Same for
units.

The conversion factors (compile-time constants in the webapp):

```js
const FREQ_CONVERSIONS = {
  THz: { factor: 1.0,      label: "THz",  decimals: 3 },
  cm1: { factor: 33.35641, label: "cm⁻¹", decimals: 1 },
  meV: { factor: 4.13567,  label: "meV",  decimals: 2 },
};
```

For the addon side, the same options should eventually appear in a
Units panel — though for v0.4.x it's THz only (matching phonopy
default). I'll bring it up as a separate change later — not a JSON v1
concern.

## Implementation greenlight

With these decisions locked, I can now implement Phase A (base layer
writer + reader + round-trip tests on WSe2/SnSe2). Estimated
~4 hours of careful work plus tests. Will propose a write/read API
sketch before I start.

But: per your message, **Phase A is not the next thing**. Bondings are
broken in 0.4.0, that's the urgent fix. After 0.4.1 is verified by
you, the priority is your SnSe₂ image-atoms issue, not JSON.
