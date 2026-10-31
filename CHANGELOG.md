# phonon-vision v2 changelog

## 3.6.0 — CRYSTAL23 provenance: fort.25 band structures (honest tier)

- **THE FORMAT ANSWER, settled by the data** (the user's open question):
  CRYSTAL's fort.25 BAND blocks carry path FREQUENCIES ONLY — no
  eigenvectors, no dynamical matrix. The .out adds geometry, the path
  definition and per-k frequency lists but still no off-Γ eigenvectors;
  FREQINFO.DAT stores symmetry-IRREDUCIBLE gradient sets (4 of 240 flags on
  the CsSnI3 example), not a directly parseable Hessian. So atom-WEIGHTED
  bands and 3D surfaces are impossible from these files, and this importer
  does not pretend otherwise.
- NEW `src/phonon_core/crystal_import/`: `read_fort25` (regex field-cutting
  — CRYSTAL's fixed-width E-fields fuse without spaces and token-splitting
  shreds them), `read_crystal_out_structure` (primitive cell recovered by
  inverting the printed EXPANSION MATRIX and folding the supercell atoms
  with a WRAP-SAFE quantization — round-after-mod split the seam and gave 6
  "distinct" atoms of 3; the fold count is asserted; slab z-in-Å convention
  handled; the run's OWN isotopic masses parsed, never guessed), and
  `convert_crystal_bands` → `{meta, band}` payload with REAL Γ-M-K-Γ path
  structure (labels matched from the .out's integer/denominator segments).
- **The honesty contract is in the payload**: eigenvector and character
  arrays are exactly ZERO and `meta.conventions.eigenvectors` says why; the
  viewer's blend falls back to its uniform species mix (the `tot > 0`
  guard) and animation shows zero displacement rather than invented motion.
- **Cross-file oracle**: fort.25's Γ row equals the frequencies the .out
  itself prints in DISPERSION K POINT 1, digit for digit (< 5e-4 cm^-1,
  print precision) — two outputs of one run through two parsers. Unit
  conversion is the frozen THZ_TO_CM1 (the group's PhononBands.py hard-codes
  its 4-digit reciprocal, asserted equivalent).
- Real data: 1T-SnSe2 monolayer, r2SCANh, 7x7 SCELPHONO — 9 branches x 2998
  path points; hex cell a = 3.797 Å recovered from the supercell (vs 3.78 in
  the PBE flagship payload — functional difference, structure identical).
  Browser gate: boots with zero console errors, 11 traces, ticks at M/K,
  Tier-1 correctly HIDDEN (no mesh_modes — nothing false is offered).
- Tests (+6): block structure, the cross-file Γ oracle, the unit-constant
  equivalence, hex/slab geometry pins (120°, c = 500 Å, masses), payload
  honesty (zeros + stated reason + monotone distance + band continuity +
  path closure), loud mismatch/truncation refusals.
- FORWARD (recorded in the module docstring): weighted bands + 3D surfaces
  for CRYSTAL need the supercell Hessian — decode FREQINFO's symmetry
  expansion, or re-run with a Hessian-writing option; from a Hessian,
  `mlip_import.convert_hessian` already supplies the entire remaining
  pipeline. The PBE0 example is CsSnI3 — the same crystal as the QE .dyn
  data, a ready-made cross-code comparison once its Hessian is reachable.

## 3.5.0 — QE .dyn → mesh_modes: thermal ellipsoids for 3D (and polar) materials

- **ONE CONCERN: a ph.x `.dyn` set becomes a viewer payload.** NEW
  `qe_import/dyn_payload.py::convert_qe_dyn` and
  `dyn_file.py::dyn_modes` (eigenvectors join the frequencies: `eigh`
  columns of the mass-weighted matrix are already the project convention —
  Dove 4.16, Σ|e|²=1 — reshaped to parser layout and phase-canonicalized;
  eigen-residual on real data 1.9e-16). `dyn_frequencies` is now a thin
  wrapper so there is one spelling.
- **This is the first 3D material with thermal ellipsoids**, and the first
  polar-QE data on screen. The payload is `{meta, band, mesh_modes}`: no
  surface (bands-over-BZ is a 2D concept), no engine provenance
  (`meta.source` carries origin qe-dyn, `arbitrary_q: false` + reason). The
  viewer needed ZERO changes — every absence was already presence-keyed, and
  the Tier-1 gate reads `mesh_modes` directly. Exporter untouched
  (md5 fe2db504).
- **The band panel shows an honest SPECTRUM, not a fake path**: x = |q|
  (1/Å, ascending), markers only — the panel's native mode, so no invented
  connectivity is ever drawn. 64 grid points are not a band path and none is
  pretended; click-to-animate works on every marker.
- **The equal-weight licence is VERIFIED, not assumed.** Tier-1's ADP
  average requires equal-weight nodes; a star-expanded set qualifies exactly
  when it fills the uniform grid. `convert_qe_dyn` infers the grid from the
  q-set, requires on-grid q (measured 1.8e-09) and a bijection onto the
  cells, and REFUSES incomplete, duplicated or off-grid sets (three
  negative controls with teeth). Real CsSnI3: 64/64.
- **The physics payoff, measured then pinned** (CsSnI3, 300 K, through the
  payload): Cs U_iso = 0.72 Å² vs Sn 0.032 — the A-site RATTLER at 23× —
  and the tensors carry the exact tetragonal site symmetries (Sn/Cs/apical-I
  Uxx=Uyy to 1e-6; the in-plane iodines are each other's x↔y mirror), facts
  the pipeline cannot fake if the eigenvector layout, mass mapping or grid
  weighting were wrong. The iodine ellipsoids stand perpendicular to the
  Sn–I bonds — the octahedral-tilt signature, visible in the browser gate's
  screenshot.
- Two test pins corrected against measurement, both recorded in-file:
  the 6-sig-fig mesh/band overlap floor is ~1e-6 not 1e-8 (test arithmetic);
  and the instability count is 63/960 modes (6.6%) — an entire soft branch,
  not the 3 Γ negatives a first version extrapolated from. For THIS
  geometry the ADP is qualitative (|ν| treatment of soft modes): anisotropy
  and the rattler contrast are robust, absolute magnitudes are not.
- Frequencies: full precision in both blocks, bitwise-identical between
  them; eigenvectors full precision in `band` (the animator's source),
  6-significant-figures in `mesh_modes` (the 0.6.81 size policy).
- Browser gate (headless Chromium + swiftshader, real payload): boots with
  zero console errors, Tier-1 visible, ADP reports "averaged over 64
  q-points" with U_iso 0.3076 Å² — exactly the Python-side mean — and
  max axis ratio 4.2:1 = √(0.512/0.029), the in-plane iodine.
- Tests (+11): structure/oracle-through-payload (<2e-6 THz), spectrum
  honesty, species character partition summing to 1, eigenvector fidelity,
  determinism, the three licence refusals, group-a refusal, the perovskite
  physics pins, and js/json write-out.

## 3.4.0 — the ph.x `.dyn` route: exact QE phonons, polar materials included

Diagnosing the 3.3.1 gap disproved the hypothesis recorded with it and
produced a better route.

- **The 3.3.1 attribution was wrong, and the measurement says so.** Applying
  an ASR projection to the converted QE force constants moves the gap from
  0.6975 to 0.6974 THz — nothing — even though the projection demonstrably
  works (row-sum drift 3.6e-02 -> 1.5e-14; three exact Gamma zeros appear).
  Enabling phonopy NAC with the file's own Born/dielectric data moves
  frequencies by only ~0.03 THz, because eps_infinity ~ 19.7 screens LO-TO
  heavily. Neither term explains 0.79 THz.
  **The real cause is structural**: for a polar material `q2r.x` SUBTRACTS
  the long-range dipole-dipole contribution before Fourier transforming, so
  the `.fc` carries only the short-range remainder and `matdyn.x` adds the
  long part back with its own Ewald parameters. That reconstruction — not a
  sum rule, not a single unit constant — is the pending work. Both documents
  that carried the wrong attribution are corrected.
- **NEW `src/phonon_core/qe_import/dyn_file.py`** — the route with no such
  caveat. ph.x `.dyn` files carry D(q) as computed, so they are EXACT at the
  computed q, need no interpolation or sum-rule choice, and **work for polar
  materials today**. `read_qe_dyn`, `read_qe_dyn_set`, `dyn_frequencies`.
- **The oracle is the file itself.** Every `.dyn` carries both D(q) and the
  frequencies ph.x derived from it, so re-deriving them checks units, mass
  convention, block indexing, hermiticity and the signed square root against
  the file's own answer with nothing external assumed. Measured across the
  group's real CsSnI3 output — **18 files, 64 q-blocks: worst disagreement
  1.69e-06 THz**, the files' printed precision.
- `RY_THZ` is DERIVED from CODATA primitives (RY_EV * e / h), asserted
  against the known 3289.842 rather than typed as it.
- Three real-format findings, each caught by the data and now covered:
  1. **A file carries the whole STAR of q** (up to 8 blocks in one file), so
     blocks are collected as a list.
  2. **ph.x diagonalizes ONCE per file**, for the star's representative — a
     file with 8 q-blocks prints one 3N-frequency list. An earlier version
     required one block per q and silently produced `None` for every starred
     file.
  3. **Concatenating a set destroys the frequency pairing** unless the index
     is remapped: measured 1.19 THz where the per-file check was 1.7e-06.
     `frequency_index` now records which q-blocks the printed rows belong to.
  Plus: Born-charge lines read `atom #    1`, so the index is the LAST token.
- **The two QE files legitimately disagree on Born charges** and only one may
  be asserted neutral: `.dyn` carries RAW DFPT charges (sum 0.389), while
  `q2r` imposes charge neutrality writing the `.fc` (sum 3e-08). Documented,
  and the tests assert accordingly per file type.
- Tests (+25): the derived-constant check; a synthetic `.dyn` built to a
  closed-form frequency (pins the Ry/bohr^2 -> THz chain independently of any
  real file); malformed-input refusals; and the real-data suite —
  per-file self-consistency across all 19 files, the set-level check with the
  remapped index, structure/species/dielectric validation, and D(Gamma)
  hermiticity + reality.

## 3.3.1 — real q2r data lands: four reader defects the analytic oracle could not reach

Data preparation, and what it immediately proved. Three real q2r `.fc` files
were found on this machine (the group's CsSnI3 EPW package, stages cubic /
epi / epirelax, each with its own `matdyn.in` + matdyn frequency output).
Every one of them broke the reader.

- **The reader failed on real input in four distinct ways**, each now fixed:
  1. **Quoted species labels contain spaces** — `'Cs '`, and one that is
     entirely blank (`'   '`). A whitespace token split shreds them. The
     header is now parsed LINE-oriented (the FC section stays token-wise:
     it is regular and free-format).
  2. **Labels contradict masses.** The file carries type 2 with a BLANK
     label at 118.710 amu (Sn) and type 3 labelled `'Sn '` at 126.904 amu
     (iodine). Masses are load-bearing — they enter D(q) — so species are
     now resolved BY MASS via `_resolve_symbol`, and how each was decided is
     reported in `symbol_sources` (`label` / `blank-label` /
     `label(Sn)-vs-mass`). Resolved composition: Cs + Sn + 3 I.
  3. **The polar block layout was wrong in the reader**: each atom's Born
     tensor is preceded by its own INDEX line, and the `T` flag line carries
     a trailing number. Both are now handled.
  4. **ibrav=6** (simple tetragonal) was not implemented.
- **The independent check that the polar fix is right**: the Born charges now
  satisfy charge neutrality, sum |Z*| = 0 to machine precision on ALL THREE
  files (2e-16, 7e-08, 3e-08). A mis-parse scrambles the tensors and that sum
  stops vanishing — physics the reader cannot fake.
- The SYNTHETIC polar fixture had been built on the same wrong layout, so it
  was corrected from the real file. Recorded because it is the clearest
  possible illustration of why an analytic oracle cannot replace real data:
  it validated the reader against a format that does not exist.
- NEW `tools/stage_qe_fixture.py` GENERATES an acceptance fixture from a
  q2r/matdyn directory (`.fc` + `matdyn.in` + `reference_freqs.csv` +
  `PROVENANCE.md`). The `.freq` stride is DERIVED from the declared
  nbnd/nks and the token count, not assumed — this project's own reference
  carries four numbers per q (the band-path distance `q_in_band_form` adds),
  and a file without it parses equally well.
- Two cases staged (`CsSnI3_cubic`, `CsSnI3_epirelax`, 15 q x 15 bands each).
  Both are POLAR, so the acceptance test now splits honestly:
  `test_real_q2r_reader` validates structure, mass-resolved species,
  charge neutrality and dielectric positive-definiteness on real data;
  `test_real_q2r_frequency_oracle` SKIPS polar cases BY NAME, because
  matdyn applied NAC (and `asr='crystal'`) and comparing would measure the
  missing physics rather than the reader.
- **The remaining gap measured** (converted without NAC vs the matdyn
  reference): max 0.786 THz overall, 0.698 at Gamma, median 0.216. The
  attribution stated here first — "dominated by the acoustic sum rule" — was
  an inference, and 3.4.0 measured it to be WRONG; see that entry. Recorded
  in `tests/fixtures/qe/README.md`.
- **The MLIP real-data skip is closed.** `test_real_force_constants_round_trip`
  now expands the SnSe2 dataset's COMPACT force constants to full format with
  phonopy (checking pair symmetry, 4.4e-16, before using it) and round-trips a
  genuine 192-atom supercell Hessian — the size and structure a real MLIP
  Hessian has, which the 16-atom toy cannot represent. No new stored bytes:
  the data stays single-sourced in `Phonopy-examples`.

## 3.3.0 — Quantum ESPRESSO provenance: q2r force constants

- **ONE CONCERN: `convert_qe_fc` — the q2r `.fc` route.** NEW
  `src/phonon_core/qe_import/`: token-wise parser for the q2r real-space
  force-constant file (header, ibrav 0–4 lattices, Ry-a.u. masses, alat
  positions, the (j1,j2,na1,na2)×(m1,m2,m3) FC blocks), unit conversion
  DERIVED in code from CODATA 2018 primitives (Ry/bohr² → eV/Å²; masses
  /(m_u/m_e)/2 — never a hand-typed pre-multiplied decimal), and the
  supercell assembly Φ[(k1,c1),(k2,c2)] = fc[k1,k2,(c2−c1) mod nr] with an
  integer-residue bijection assertion on the atom mapping (the
  position-gauge lesson). phonopy writes the yaml (the sanctioned
  `mp_import.pheasy` pattern).
- **The oracle problem, faced honestly.** No QE is installed here, so the
  importer discipline's real-data oracle cannot run yet. The substitute is
  the strongest available: `.fc` fixtures BUILT FROM THE FORMAT SPEC for
  spring models with CLOSED-FORM dispersions — an anisotropic simple-cubic
  toy (distinct kx≠ky≠kz pins the m-nesting and block order) and an
  ibrav=0 two-sublattice toy (distinct masses pin the Ry-a.u. conversion) —
  compared through the full chain at every supercell-commensurate q to
  <1e-9 THz. What this CANNOT settle is listed as PENDING in the module
  docstring: real-file header quirks, and the Fourier SIGN (time reversal
  makes ω(q)=ω(−q) in any crystal, so NO frequency oracle can pin it — it
  is e ↔ e*, invisible to frequencies and |e|², visible only to
  phase-sensitive observables). `tests/fixtures/qe/README.md` carries the
  acceptance protocol;
  `test_real_q2r_acceptance` activates the moment real matdyn-verified
  data lands.
- **Polar files are REFUSED, not guessed**: the dielectric/Born block is
  parsed, and conversion raises with the reason — activating NAC needs the
  QE↔phonopy NAC unit constant pinned against real matdyn output first
  (the pheasy-NAC lesson: a plausible wrong constant costs 1e-2–1e-1 THz
  and only an oracle catches it). Deliberately NO override flag.
- Tests (+8): the two analytic-oracle chains; A1↔A2 agreement on the
  written triple; the reader proven INDEX-driven, not line-order-driven
  (swap two full lines → bitwise-identical physics; swap values only →
  >0.1 THz shift — the teeth); loud truncation/ibrav errors; header unit
  round-trip; polar refusal; the real-data acceptance hook (skips, named).
- phonopy rejects fictitious element symbols ('X'), so the toys use real
  ones with explicit masses — recorded because the error surfaces deep in
  the yaml-writing path.

## 3.2.0 — MLIP provenance: supercell Hessians join the common ground

- **ONE CONCERN: `convert_hessian` — the machine-learned-potential route.**
  NEW `src/phonon_core/mlip_import/` (pure numpy, NO phonopy import:
  phonopy's full-format FORCE_CONSTANTS *is* the supercell Hessian in
  eV/Å², so this is validation + layout, not a physics transform). Input:
  the (3N,3N) or (N,N,3,3) Cartesian Hessian of a supercell an existing
  `phonopy.yaml` describes (autograd / HVP / finite differences over an ASE
  calculator); output: the phonopy triple every engine here consumes.
- Refusals and reports, per the importer discipline: units must be eV/Å²
  (anything else refused — no silent unit guess); asymmetry ‖H−Hᵀ‖∞
  measured, symmetrized below `asym_tol`, REFUSED above; optional
  `atom_order` permutation; ASR drift reported always, enforced on request.
- **Two wrong ASR repairs found and documented by the suite, then a third
  finding that reshaped the toy:**
  (i) the textbook diagonal correction zeroes ROW sums but breaks pair
  symmetry, so the engine's hermitization re-imports the violation — D(Γ)
  loses its translation nulls (0.069 eV/Å² leakage, fake Γ acoustics at
  1–3 THz); (ii) alternating it with re-symmetrization does NOT converge —
  the antisymmetric diagonal residue is an exact 2-CYCLE (rows 9e-16, asym
  stuck at 5.12 for 200 passes). The correct repair is the symmetric
  projection Φ′ = Φ − rᵢ/N − cⱼ/N + T/N², which zeroes rows AND columns in
  one shot and provably preserves pair symmetry; (iii) with the projection
  in place Γ STILL failed — because the toy itself was unphysical: the
  engine reads only PRIMITIVE rows and hermitizes, so Γ closure needs the
  input to be TRANSLATION-CONSISTENT (Φ[Ti,Tj]=Φ[i,j]), which a periodic-
  supercell Hessian is (to solver noise) and a raw random matrix is not.
  The toy is now translation-averaged; the TI expectation is documented in
  the module as a scope statement, not enforced.
- Tests (+7): round trip vs the reference engine (doctrine, <1e-10 THz);
  flat-(3N,3N) ≡ (N,N,3,3) bitwise; the atom-order trap with the exact
  0.6.89 signature (a SAME-primitive image swap: Γ-invisible <1e-6,
  generic q >0.05 THz, and `atom_order` restores exactness — a first
  version swapped ACROSS primitives, which moves Γ too and tests nothing);
  asymmetry refusal + sub-tolerance reporting; loud bad-input errors; ASR
  → exact Γ acoustic zeros; a fixture-guarded real-data round trip (skips:
  the SnSe2 dataset ships compact FC, and a Hessian is full-format by
  nature).

## 3.1.0 — band-following: connect branches by maximum eigenvector overlap

- **ONE CONCERN: branch indices become physical identities.** An eigensolver
  sorts by frequency, so at every band crossing the "branch" silently jumps
  between physical bands. NEW `src/phonon_core/band_connect.py` connects
  branches along the path by gauge-invariant squared eigenvector overlap
  (greedy best-first assignment on the near-permutation overlap matrix;
  deterministic tie-break). The connection is a RELABELLING only — a
  permutation table, frequencies and eigenvectors untouched — and it NEVER
  crosses a path discontinuity (each segment restarts sorted; the
  disconnected-path lesson).
- Exporter: opt-in `--band-connect` bakes a presence-keyed `band.connect`
  block (perm table + min matched overlap + reorder count + segment starts +
  a prose convention line), computed from the band block's OWN arrays so the
  table can never disagree with the data beside it. Default output is
  byte-unchanged (key-set pins elsewhere in the suite stay valid).
  `pv_navigator_export.py` md5 494936ed → fe2db504.
- Tests (+10): the crossing toy where connected bands must equal ANALYTIC
  straight lines exactly; an adiabatic avoided-crossing model where identity
  is required even though the end-point eigenvectors are exchanged; gauge
  invariance under random per-mode phases; permutation validity +
  determinism on random unitary modes; the path-break rule; degenerate-
  subspace validity; a content-flip negative control with constant
  frequencies; real-SnSe2 connectivity; and the exporter bake (opt-in +
  exact agreement with the library).
- THREE first-run test failures, all in the TESTS, each instructive:
  (i) the avoided-crossing "ends exchanged" precondition needed a narrower
  gap; (ii) the negative control flipped modes at a crossing toy's own swap
  point, which UNDID the sorted swap — the module correctly returned
  identity and the control was testing nothing (rebuilt on non-crossing
  constant bands); (iii) a blanket 0.5 overlap floor is WRONG on real data —
  degenerate subspaces legitimately split pair overlap (1/2 two-fold, 1/3
  three-fold; SnSe2 measures 0.368, between the two). The repaired pin is
  the discriminating one: every weak connection must be explained by
  near-degeneracy (< 0.25 THz gap), and the floor drops to the 3-fold value.
- Viewer consumption of the baked table (recolour/step by connected branch,
  trajectory `branch=` disambiguation) is FORWARD work, recorded in
  CLAUDE.md's roadmap.

## 3.0.1 — package assembly complete: governance + the Blender add-on

- NEW `CLAUDE.md` — the governance document for the 3.x line: versioning and
  ship convention (`../pv3_ships/phonon_vision3_<Mmp>.zip`), the privacy
  ruling (unpublished materials STAY in the test data; the tree is
  unpublishable as-is and `ship_guard` FAIL is the expected state), the
  battery commands and skip-signature health rule, viewer rules, physics
  conventions, the importer discipline every new provenance must follow, and
  the v2-specific trap ledger. Deliberately carries NO prose "current
  version" stamp — pyproject + the CHANGELOG heading + the highest ship zip
  are the truth (the stale-stamp lesson).
- **The Blender add-on joins the package** (user ruling 2026-07-30): the
  0.8.8 tree copied to `blender/` (61 files, 40 MB, wheels included; the
  ancestor `../phonon_vision-0.7.7/` is untouched and frozen). Its guards
  run against THIS tree: `check_stamps.py` exit 0 (three stamps agree at
  0.8.8) and `check_vendored_parity.py --core-src ..` exit 0 — **17 vendored
  modules byte-identical to this package's `src/` + 2 pinned packaging
  stubs**, so the add-on folds in with zero drift. The add-on keeps its own
  version line (jumping it to 3.x would falsify its changelog).
- **Version unification**: `src/phonon_core/__init__.__version__` moves
  0.7.4 → 3.0.1 in lockstep with `pyproject.toml` (the package is the
  versioned thing now; the data layer's 0.6.x–0.7.4 lineage is history in
  this file). No test pins the literal — `test_json_v1` reads the live
  value, verified before the bump.
- 3.0.0 sealed BEFORE this work at `../pv3_ships/phonon_vision3_300.zip`
  (19,015,736 bytes, md5 013c34d0) — the never-develop-over-an-unsealed-
  version rule now applies to this line.
- Ancestor CLAUDE.md files get a deprecation banner pointing here (the only
  edit either ancestor receives).

## 3.0.0 — the merge: v2 becomes the complete package

Recorded retroactively (the merge itself was executed under a
no-file-modification constraint, so this entry lands with 3.0.1).

- `phonon-core` 0.7.4's data layer (`src/`, 33 files), tools (exporter,
  launcher server), full test battery with fixtures (18 MB), JS DOM suite,
  and docs merged ADDITIVELY into this tree; `phonon-core/` itself
  untouched. v2's viewer (the 4103-line superset) and modules kept.
- NEW `src/phonon_core/adp.py` (packaged ADP; `python/phonon_adp.py`
  retained untouched), NEW `tests/test_adp_parity.py` pinning the two
  bitwise, NEW `pyproject.toml` (name `phonon-vision`, MIT), NEW
  `PACKAGE.md`.
- Acceptance: python **384 passed / 4 skipped** (the healthy skip
  signature); JS **163 passed** (the core's own 135-test DOM suite green
  against the merged viewer + v2's 28); END-TO-END the tree baked its own
  SnSe2 payload (7.3 MB valid json_v1) from its own fixtures with its own
  exporter — self-sufficiency proven, not asserted.
- Privacy status CHANGED: the tree carries unpublished-material references
  and no longer passes `ship_guard` (193 findings, expected).

## 2.2.0 — Tier 3: figure output and interoperability

The last tier of the programme. Neither feature changes what the viewer
computes; both change what can leave it.

### Publication figures

- **View along a / b / c / the a-b plane normal.** Orthographic and
  axis-aligned, so on-screen distance stays proportional to distance in the
  crystal. Re-orienting keeps the current zoom rather than re-fitting, so
  choosing an axis does not also throw away the framing. Built from THREE
  vector methods only, per the standing rule that `index.html` carries no
  trigonometry of its own; the up-vector is orthogonalized against the view
  direction with a fallback for the parallel case (hexagonal c along a1).
- **Save figure (PNG)** at 1x / 2x / 4x with a transparent background. The
  panel on screen is untouched: the backing buffer is resized with
  `updateStyle = false`, rendered, read, and restored in a `finally`.
- **The renderer has no `preserveDrawingBuffer`** — it costs on every frame —
  so the buffer is only valid until the browser composites. `render()` and
  `toDataURL()` therefore run in the SAME synchronous turn, with no `await`
  or rAF between them. Verified rather than assumed: the exported image
  decodes to 47,874 opaque coloured pixels against 1,975,246 transparent
  ones at 2662x760. A blank export would have been the silent failure here.

- **A framing bug the screenshot caught.** The first version preserved the
  current zoom when re-orienting, on the reasoning that choosing an axis
  should not also reset the framing. That is wrong whenever the structure's
  extent along the new screen axes differs from the old: SnSe2's tiled block
  spans ~38 A along c but ~11 A in plane, so carrying the top-down framing
  into the view-along-a orientation put every atom outside the window and
  drew an empty panel. The view now re-fits, using the same corner-projection
  extents as the initial fit. Verified numerically rather than by eye: all
  four axis views project 27/27 atoms inside the frustum.

### Trajectory export

- NEW `web/trajectory.js` — extended-XYZ (extxyz) writer. **Save trajectory
  (.xyz)** writes the tiled block, one frame per phase over a full period,
  which opens directly in OVITO, ASE and VMD.
- **Per-atom displacement vectors are written as a `disp` column**
  (`Properties=species:S:1:pos:R:3:disp:R:3`). OVITO renders those as arrows,
  and it makes the file self-describing: the motion is recoverable from a
  single frame without differencing.
- **The exported lattice is the TILED block's**, not the primitive cell's.
  The frames contain tiled atoms, so a primitive lattice would contradict
  them and any tool applying periodic boundary conditions would be wrong.
- **Coordinates are un-centred.** `r0c` is centroid-shifted for display; the
  export adds `cCell` back so the atoms sit in the crystal frame the written
  lattice describes.
- Loop phases exclude the endpoint, so a looping animation does not stutter
  on a duplicated frame.

- **A units bug the validation caught.** The first version wrote the internal
  `amp` under the key `amplitude_A`. That value is mode-normalized
  (`peak_A / peakUnit`), not a length: it read 2.8563 where the real peak was
  0.1999 A. The file now writes the slider's physical peak as `amplitude_A`
  AND the peak actually realized by the sampled frames as `peak_disp_A`, so a
  reader can verify the claim against the `disp` column without trusting the
  header.

### Verification

- **Validated against ASE, the real consumer**, not against our own
  assumptions. A 12-frame SnSe2 export parses as 12 frames x 27 atoms,
  formula Se18Sn9, cell = 3a1 x 3a2 x 1a3, `disp` read as a per-atom array,
  all 11 metadata keys typed. Three assertions on the physics:
  `peak_disp_A` equals the file's own max |disp| exactly; frame 0 differs
  from the last frame (the loop has no duplicate); and positions minus disp
  give the SAME equilibrium in every frame to 1.0e-06 A, which is the file's
  own write precision — so the position and displacement columns are
  mutually consistent.
- JS **28 passed** (20 + 8 trajectory); Python **17 passed** unchanged.
  Viewer guard battery clean; `kinematics.js` still byte-identical.

## 2.1.0 — Tier 2: model styles, radius styles, edge completion

### Radius styles (covalent / atomic / van der Waals)

- NEW `tools/gen_radius_tables.py` GENERATES `web/radii_data.js` from
  pymatgen's `periodic_table.json` (Slater 1964 atomic; Bondi 1964 /
  Alvarez 2013 van der Waals). The table is not hand-typed: a 96-element
  radius set is exactly where a transcription error is easy to make and
  invisible until it silently changes a bond count. Shipped coverage is 96
  vdW and 88 atomic elements.
- The generator SANITY-CHECKS before writing: seven independently known vdW
  values, plus the requirement that vdW > atomic for every shared element.
  If a future pymatgen reshuffles its schema and the script starts reading
  the wrong field, it exits instead of emitting a plausible wrong table. The
  same two checks are repeated in the JS suite against the SHIPPED file.
- `covalent` remains **bit-for-bit the historical default**
  (`BALL_A_PER_COV` = 0.34 x the payload's Cordero radius). That constant is
  shared BY VALUE with the Blender add-on, so it is untouched; the new styles
  are additional, never a redefinition. Atomic and vdW are raw table values —
  a space-filling sphere IS the van der Waals radius, not that radius times a
  ball-and-stick shrink factor.
- Missing entries return `null` rather than a guess. The viewer falls back to
  covalent and NAMES the elements it did that for.
- Switching style clears per-species manual radius overrides, which were
  absolute Angstrom values chosen against the old table; keeping them would
  silently mix two conventions in one scene.

- **No ionic style, deliberately.** A single ionic radius per element does not
  exist: Shannon's radii are indexed by oxidation state and coordination
  number, and selenium is 0.64 A as Se(4+) but 1.84 A as Se(2-). In SnSe2 the
  selenium is Se(2-), so pymatgen's per-element average of 1.013 A describes
  no real chemistry. Choosing the state per structure is a chemistry
  inference, not a radius table, and is left as an explicit non-goal.

### Model styles

- One `Model` preset drives sphere size, radius table, bonds and polyhedra
  together: Ball-and-stick, Space-filling, Polyhedral, Wireframe.
  Ball-and-stick reproduces the historical default exactly.
- `Custom` appears automatically the moment any underlying control is touched
  directly, so the selector never claims a state the scene is not in.

### Edge-completion boundary images (the parked VESTA-style item)

- An atom on the surface of the tiled block looks under-coordinated, but that
  is an artefact of where the tiling stops. This draws the missing neighbours.
- **The inclusion rule is chemical, not geometric.** A ghost is drawn iff it
  is BONDED to a real atom under the same cutoff table the bond renderer
  uses. A fractional-window rule would both include images that complete
  nothing and miss genuinely bonded ones just outside the window.
- Ghosts are translucent so they cannot be mistaken for the block, follow the
  live radius style and model scale, and each carries the integer cell it
  occupies — so `kinematics.js` gives it the Bloch phase of that cell. A
  ghost that moved with its parent would be wrong at any q away from Gamma.
- Measured on SnSe2 at 3x3x1: 42 image atoms, 70 completing bonds. Survives
  a model switch and a retile.
- This reuses the image-aware search built for Tier 1 polyhedra, which is why
  the Tier 1 report named it the natural foundation for this item.

### Verification

- JS **20 passed** (16 Tier 1 + 4 radius-table); Python **17 passed**
  unchanged. Viewer guard battery clean: 0 `Math.*`, one `CHAR_SCALE`, 0
  `localStorage`. `kinematics.js` still byte-identical.
- Browser: default boots `ball/covalent` (historical default preserved);
  Space-filling switches to vdW with bonds off; Polyhedral enables polyhedra;
  boundary completion reports 42/70 and persists across a model change. No
  console errors.

## 2.0.0 — Tier 1: thermal ellipsoids and coordination polyhedra

Built on the v1 viewer (`phonon-core` 0.7.4, `web/index.html` 6c3b267b). Both
features are presence-keyed on `mesh_modes`: a payload without a q-mesh hides
the whole section rather than showing dead controls.

### Thermal ellipsoids (ADPs)

- NEW `python/phonon_adp.py` — the reference implementation and the oracle for
  the browser module. Harmonic mean-square displacement tensors
  `U_s^ab(T)` averaged over the q-mesh, plus `principal_axes` and the ORTEP
  probability scaling.
- NEW `web/adp.js` — the numerical twin, with a cyclic-Jacobi symmetric 3x3
  eigensolver (chosen over a closed-form solver because it stays accurate on
  the near-degenerate tensors an isotropic site produces, and returns an
  exactly orthonormal frame to feed a rotation matrix).
- Live temperature slider, 50/90/99% probability levels, adjustable mode floor.
  `U` depends only on (T, floor), so the tensors are cached and changing the
  probability or opacity never recomputes the mesh average.
- **ORTEP display rule.** At ordinary temperatures the ellipsoid is smaller
  than the drawn ball — SnSe2 at 300 K gives semi-axes of 0.14-0.33 A against
  a 0.47 A Sn radius — so enabling ellipsoids shrinks the balls to 0.25x.
  Found by looking at the render, not by reasoning: the first working build
  drew the ellipsoids entirely inside the spheres.
- Sampling facts are REPORTED, never silent: q-point count, modes dropped
  below the floor and the fraction that represents, and the softest mode kept.

### Coordination polyhedra

- NEW `web/polyhedra.js` — convex hull, polyhedron assembly, per-frame vertex
  packing, and the volume (via the divergence theorem) that turns the
  qualitative picture into a number to watch.
- Hull TOPOLOGY is fixed at equilibrium and only the vertices follow the
  animation. Recomputing the hull per frame would let faces flip whenever a
  vibration makes four vertices momentarily coplanar, which reads as flicker.
- **Coplanar facets are triangulated once, as a fan.** A square facet has four
  supporting triples, each covering half the square; emitting all of them
  double-covers it. Caught by the cube test, which reported volume 16 for a
  cube of volume 8 — a clean factor of two.
- **Neighbour search is periodic-image aware.** SnSe2's lower Se sits at
  fractional z = 0.96, so three of the six octahedral neighbours exist only as
  images one cell down along a3, which a 3x3x1 tiling never instantiates. The
  bond-list version of this search found 3 neighbours and drew nothing; the
  image-aware version recovers the full SnSe6 octahedron (6 vertices, 8 faces,
  V0 = 25.853 A^3 against 25.87 A^3 for a regular octahedron of the same
  vertex distance).
- Each vertex carries the integer cell it lives in, so `kinematics.js` gives an
  image its OWN Bloch phase (`2*pi * n_int . q_frac`) instead of copying its
  parent's displacement. One engine call per frame covers every vertex of
  every polyhedron.
- Cutoffs are read from the same editable `bondTypes[]` table the bond
  renderer uses, so widening a pair's distance still reshapes the polyhedra.
- When nothing can be drawn, the panel reports the coordination it actually
  found and what to change, instead of a bare "nothing to draw".

### Verification

- Python: **17 passed**. Every claim is checked against something independent
  of the implementation — the closed-form single-oscillator result, the
  classical `k_B T / m w^2` limit (matched to 1 part in 10^5 at 20000 K), the
  chi-squared definition of the ORTEP factors, `U ~ 1/m` scaling, and tensor
  symmetry/positive-definiteness.
- JavaScript: **16 passed**, including a parity suite against the Python
  oracle (5 cases, agreement to 1e-10 relative) and hull geometry checked
  against exact volumes of a regular octahedron (4/3) and a cube (8).
- Browser (headless Chromium + swiftshader, the only place this is testable —
  jsdom initializes no WebGL, so the atoms panel that owns these controls
  never builds): both toggles drive clean, no console errors, U_iso 0.0214 A^2
  at 300 K rising to 0.0638 at 900 K, reported axis ratio 2.4:1 = sqrt(5.64),
  consistent with the Python tensor's U_zz/U_xx.

### Notes

- `index.html` stays free of `Math.*` calls; all new arithmetic lives in
  `adp.js` / `polyhedra.js`, the rule `kinematics.js` already followed.
- Two internal guards use hoisted `var` (`tier1Ready`, `adpAtomScale`) rather
  than `let`, because `buildTiling` appears earlier in source than the Tier-1
  block and reading a `let` before its declaration executes is a TDZ throw.
- `kinematics.js` is untouched.
