# phonon-vision add-on changelog

## 0.8.8 — the conformance gate reaches the arbitrary-q engine path (and becomes a tie-fix regression test)

Closes the gap surfaced by the 0.8.7 HPC acceptance run: `conformance_headless.py`
could only exercise the PRECOMPUTED-file render path (`band.yaml` + `q_index`
into a fixed grid), never the ENGINE path — so it proved nothing about
`evaluate_dq_np` / `dq_numpy`, the very code the 0.6.87 tie fix lives in. One
concern, one file: `tools/conformance_headless.py` gains an engine-mode branch.

**What the gate now does.** A fixture whose `"input"` is a `phonopy.yaml`
(+ sibling `FORCE_CONSTANTS`) is detected as ENGINE mode (`props.data_kind ==
"ENGINE"`), and the gate drives the SAME seam the GUI q-scrub uses —
`props.q_frac_*` → `EngineSource.set_q` → `evaluate_dq_np` (the vendored A2 /
`dq_numpy` code path) — at an explicit, arbitrary `"q_frac"`. Two checks:

- **RENDER conformance** (as before, now for the engine data source): Blender's
  displayed atom world positions == `motion.positions_at_frame` for the
  engine-evaluated mode, over a full animation cycle. Proves the render path
  faithfully DISPLAYS whatever the engine computes.
- **PHYSICS conformance** (new): the engine's frequencies at `q_frac` are
  compared to a phonopy-A1 golden baked into the fixture, under the metric
  doctrine (`|Δν|` above 1e-3 THz, `|Δλ|` below). Proves the deployed engine
  COMPUTES the right thing on the cluster — not just that the fixed code is
  present (the install script's `hasattr(DEFAULT_SYMPREC)` check remains that).

**On the private triclinic acceptance dataset this is a genuine tie-tolerance regression test in production.**
At the generic, off-grid q `(0.137, 0.291, 0.053)` the fixed wheel (0.2.1,
`dq_numpy` `d1059152`) reproduces phonopy-A1 to **9.4e-15 THz**; a pre-0.6.87
relative-tie wheel (0.2.0, `80bef1ef`) DIVERGES by **1.2e-2 THz** — a
12-decade fail-then-pass, verified by running the gate against both wheels:

| wheel | render check | physics check | RESULT | exit |
|---|---|---|---|---|
| 0.2.1 (deployed, tie-fixed) | OK (~4.7e-7 Å) | max\|Δν\| 9.4e-15 THz → OK | PASS | 0 |
| 0.2.0 (pre-0.6.87 tie bug) | OK (~4.7e-7 Å) | max\|Δν\| 1.2e-2 THz → FAIL | FAIL | 1 |

The render check passes on BOTH wheels (it tests display fidelity, not engine
correctness) — which is exactly why the physics check is a separate, necessary
assertion. Verified on Blender 4.5.0 (cp311) AND 5.2.0 (cp313); exit codes
propagate unpiped. (High-symmetry materials have exact ties and cannot catch
the tie bug — SnSe2 reproduces on both wheels; the private pseudo-symmetric
dataset is the one that reaches it.)

**Private-dataset scope (user ruling, 2026-07-15).** The acceptance dataset
lives only on the private cluster and is excluded from any public package —
same discipline as the core line's public cut; this tree carries no copy of
it.

**No physics/render source changed** — `motion.py`, `evaluate_dq_np`, the
vendored wheel (still 0.2.1), the frame handler are all byte-identical. This
is a TEST-GATE change: it certifies the engine path that already shipped at
0.8.0/0.8.7, and does so without touching it.

**Scope vs 0.8.7:** `tools/conformance_headless.py` (`40748db7` → `d8fd875a`),
`__init__.py` + `blender_manifest.toml` (stamps), CHANGELOG + CLAUDE.md.
Parity guard exit 0 unchanged (no wheel move). Stamps exit 0.

## 0.8.7 — vendored wheel re-sync: phonon_core 0.2.0 → 0.2.1 (the tie-tolerance fix reaches Blender)

Closes the parity break DISCOVERED (and deliberately not folded in) at 0.8.6.
The vendored wheel still carried core 0.6.86's `engine/dq_numpy.py`
(`80bef1ef`) — the **pre-0.6.87 RELATIVE shortest-vector tie tolerance**,
i.e. the exact physics bug the core line fixed at 0.6.87 (`d1059152`,
absolute-Ångström phonopy `symprec`). Consequence until now: Blender's
engine mode (live q-scrub 0.8.0, path walking 0.8.5) evaluated D(q) with the
buggy rule; on pseudo-symmetric cells (image gaps in the
1e-5…1e-4 Å window) frequencies were wrong by ~0.1–0.2 THz, invisible at Γ.
High-symmetry and generic cells were unaffected (the core line proved the
fix bitwise-null there), which is why no golden ever caught it locally.

**The re-cut, by the sanctioned path:** `tools/_subset.py` `WHEEL_VERSION`
0.2.0 → 0.2.1; `tools/build_vendored_wheel.py --core-src ../phonon-core`
(core at 0.6.91) → `wheels/phonon_core-0.2.1-py3-none-any.whl` with
`dq_numpy.py` = `d1059152`, phonopy-absent A2 smoke PASS inside the wheel;
top-level packaging stub re-pinned IN THE SAME COMMIT (`d3638416` →
`35f9ac1b`, version-string change only; engine stub unchanged `4bdbbc56`);
old 0.2.0 wheel deleted; manifest entry updated.
**Parity: fail-then-pass on record** — `check_vendored_parity.py` exit 1
before (68 diff lines in dq_numpy), exit **0** after (17 modules
byte-identical to core 0.6.91 + 2 pinned stubs).

**A consumer trap found by the re-cut itself (the 0.6.79 lesson, again):**
four test files hardcoded the literal wheel filename
`phonon_core-0.2.0-py3-none-any.whl` (5 sites: test_motion,
test_job_roundtrip ×2, test_jump_targets, test_path_walk) and broke the
moment the wheel moved — `ModuleNotFoundError`, plus a cascade
AssertionError in test_jump_targets (hsp couldn't read supercell_matrix
through the missing wheel). Bumping the literal would re-arm the trap at
0.2.2; instead all five sites now derive the filename from
`_subset.WHEEL_VERSION` — the same single-source the builder and parity
guard already share (`check_vendored_parity.py` was already doing it right,
reading the manifest). Fail-then-pass: battery 3× exit 1 pre-patch, 70/70
post-patch.

**Regression proof:** `test_motion.py`'s float64-hex goldens (which
zip-import the tree's wheel and drive the REAL A2 engine) pass BIT-UNCHANGED
under the new dq_numpy — matching core 0.6.87's own proof that the tie fix
is bitwise-null off the pseudo-symmetric regime. Full battery 70/70
(bonds 25, motion 8, job_roundtrip 11, jump_targets 21, path_walk 5);
stamps exit 0.

**Ship:** `extension validate` + `build` clean under Blender 5.2.0 LTS;
installed via `install-file` into BOTH lines (4.5 LTS and 5.2 LTS user
repos) and headless-smoked on both — real PNGs, engine mode running the
FIXED physics on both interpreters (cp311 + cp313 dual bundle from 0.8.6,
unchanged). Also from this pass: the 0_8_6 snapshot zip was repacked to the
root-level layout every other snapshot uses (it had wrapped a top-level
`phonon_vision-0.7.7/` dir — audit-tooling breakage, artifact-only, no code
involved).

**Scope:** `tools/_subset.py` (WHEEL_VERSION + stub re-pin),
`wheels/phonon_core-0.2.1-py3-none-any.whl` (replaces 0.2.0),
`blender_manifest.toml` (wheel entry + stamp), `__init__.py` (stamp),
4 test files (wheel-filename single-sourcing), this CHANGELOG entry,
CLAUDE.md (ledger + provenance). PyYAML/h5py wheels untouched (both ABI
sets, 20 files, byte-identical to 0.8.6). No add-on runtime source touched.

## 0.8.6 — Blender 5.2 LTS: the wheel refresh (dual-bundled, not a cutover)

Blender 5.2.0 LTS shipped 2026-07-14, confirming exactly what the 0.7.13
"Blender 5.2 readiness" checklist predicted: it bundles **Python 3.13.13**
(VFX Platform 2026), which silently ignores every cp311 binary wheel
(PyYAML, h5py). This version closes that gap. **`blender_version_min` STAYS
at 4.5.0** — this is an ADDITION, not a cutover: 4.5 LTS (supported to July
2027) and 5.2 LTS are both served from ONE package.

**The decision.** Two options were on the table: cut over to cp313-only
(raise `blender_version_min` to 5.0, strand every 4.5 user immediately) or
dual-bundle cp311+cp313 in one manifest. Confirmed from Blender's own
extension-platform source (`bl_extension_utils.py`'s
`python_versions_from_wheels` / `paths_filter_wheels_by_platform` in
`addons_core/bl_pkg`) that the extension installer already filters bundled
wheels PER INTERPRETER at install time — a manifest listing both ABI tags
for the same package is a supported, sanctioned pattern (the "upstream
add-ons ship this way" option the checklist named). Architect-ruled:
dual-bundle.

**`download_wheels.sh`: a second landmine, found migrating.** Running
`--py-tag cp313` failed outright for BOTH pinned packages on macOS x86_64
with "No matching distribution" — not a "latest resolves differently" case:
CPython 3.13 raised its own macOS baseline, so PyPI's cp313 wheels for
PyYAML 6.0.2 and h5py 3.14.0 are tagged `macosx_10_13_x86_64`, while cp311
wheels are tagged `macosx_10_9_x86_64` (the script had this hardcoded).
Verified against the live PyPI JSON index (`pypi.org/pypi/<pkg>/<ver>/json`)
before touching anything. Fixed with a `MACOS_X86_64_TAG` pin keyed by
`PY_TAG` (same reproducibility discipline as the version pins — an explicit
per-cp-tag table, not auto-detection) — the script now correctly produces
a complete 5-platform set for EITHER tag, and refuses with a named error for
an unpinned one rather than guessing.

**Wheel set: 10 wheels → 20** (`phonon_core-0.2.0` is `py3-none-any`,
unaffected either way). `blender_manifest.toml`'s `wheels = [...]` gained the
five cp313 PyYAML + five cp313 h5py entries alongside the existing cp311
ten; `platforms` unchanged (linux-x64, linux-arm64, macos-x64, macos-arm64,
windows-x64 — both ABI sets cover all five).

**Verification (both LTS lines, both through the REAL install path — not
the tree-import shortcut).** `tools/pv_headless_smoke.py`'s own docstring
claims "no install step needed" via a manual `sys.path` + `importlib`
trick; that claim is FALSE for wheel-bundled dependencies — a manually
`register()`-ed add-on never triggers Blender's wheel-extraction machinery,
which only fires through the real extension install pipeline. This was
invisible under 4.5 only because a stale `install-file`-installed copy from
a prior session happened to leave `phonon_core` reachable on that
interpreter's path; a fresh Blender 5.2 profile has no such leftover and
the tree-import shortcut fails loudly (`ModuleNotFoundError: No module
named 'phonon_core'`) — a real, generalizable gap in the smoke script's
self-containment claim, now on record.

The valid procedure, run for both versions: `extension validate` →
`extension build --source-dir phonon_vision` (this is also where the
manifest string-length cap is enforced — `permissions.files` measured at
58 chars, under the 64-char limit, clean) → `extension install-file -r
user_default -e <pkg>.zip` → smoke-test the INSTALLED copy.

| | Blender 4.5.0 (regression) | Blender 5.2.0 LTS |
|---|---|---|
| bundled Python | 3.11.11 | 3.13.13 |
| wheel ABI used | cp311 (unchanged) | cp313 (new) |
| `extension build` | — | clean, 40.75 MB package |
| `extension install-file` | (pre-existing) | STATUS Installed |
| headless smoke | exit 0, 1920×1080 PNG, 1.18 MB | exit 0, 1920×1080 PNG, 1.15 MB |

Both renders load the same `band.yaml` (SnSe₂), build the scene, and write a
real PNG end-to-end. Plain-Python battery re-run clean, unaffected by any of
the above (all bpy-free): `test_bonds.py` 25/25, `test_motion.py` 8/8,
`test_job_roundtrip.py` 11/11, `test_jump_targets.py` 21/21,
`test_path_walk.py` 5/5. `tools/check_stamps.py` exit 0 (all three stamps
agree at 0.8.6).

**A local dev-tooling quirk, not a shipping one.** The source tree's
directory is still named `phonon_vision-0.7.7` (a fossil name, same
pathology already fixed once on the private core line) — this breaks
`pv_headless_smoke.py`'s tree-import trick, which derives the Python package
name from the directory basename and expects it to match the manifest id.
Confirmed this does NOT affect the actual shipped package: `extension
build` names the installed copy from `blender_manifest.toml`'s `id` field,
never the source directory, and the built zip carries no top-level folder
at all. Worked around LOCALLY with a non-destructive symlink
(`../phonon_vision -> phonon_vision-0.7.7`, not part of any shipped
artifact) rather than renaming the tree — that rename is a separate concern,
out of scope for this bump, flagged for its own future one-line version bump
if the fossil name keeps costing dev time.

**DISCOVERED, NOT CAUSED — a pre-existing parity break, unrelated to Blender
5.2.** `tools/check_vendored_parity.py --core-src ../phonon-core` now exits
**1** (previously exit 0, verified 2026-07-12 against core 0.6.86): the
private core line has moved through 0.6.87 (the A2 tie-tolerance fix,
`engine/dq_numpy.py` `80bef1ef` → `d1059152`) since the vendored wheel
(`phonon_core-0.2.0`) was last cut, and the wheel was never re-synced. This
predates this session and is orthogonal to the Python-ABI/wheel-tag work
here — fixing it means re-running `tools/build_vendored_wheel.py` and
re-pinning `tools/_subset.py`, a physics-engine-parity concern, not a
Blender-API-compat one. Per the one-concern-per-bump discipline it is
**flagged, not folded in**: this build is Blender-5.2-ready but is
**NOT parity-clean for release** until a follow-up wheel re-cut lands.
CLAUDE.md's "Parity before release" gate is intentionally left unsatisfied
here, on record, rather than silently patched.

**Scope:** `download_wheels.sh` (macOS tag fix, documented), 20 new wheel
files under `wheels/` (10 cp313 PyYAML+h5py; cp311 set untouched),
`blender_manifest.toml` (wheel list + version stamp), `__init__.py`
(version stamp), this CHANGELOG entry. `blender_version_min` **unchanged**
at 4.5.0 (dual-bundle, not a cutover — no reason to raise it). No Python
source touched (`api.py`, `bonds.py`, `operators.py`, `ui.py`, etc. all
byte-identical — this is a wheels-and-manifest-only concern, matching the
0.7.13 audit's own prediction: "the code needs nothing for 5.x, the wheels
are the real work").

## 0.8.5 — PATH WALKING: carry q continuously along the declared band path

The engine navigator gains its second control. TWO models, side by side:
**jump** = the unique high-symmetry BZ points, deduped by wrapped q (0.8.1–0.8.4);
**walk** = the band file's DECLARED route (SeeK-path or any custom `BAND=`), its
directions PRESERVED. One `path_s` slider (0→1) carries q CONTINUOUSLY along
that route, with EXACT D(q) evaluated at every position — a continuum, not
precomputed samples. This is a pure PARAMETERIZATION LAYER over engine mode:
s → segment → linearly interpolated q_frac → the EXISTING guarded q update. No
new physics, no evaluator change, no kinematics change, no jump-row change.

- **The segment table (state.remember_band_hsp EXTENDED).** Alongside the jump
  pairs, the ordered table `[(label_start, label_end, q_start, q_end, length)]`
  is stashed, with `length = |Δq @ reciprocal_lattice|` (Cartesian, so s tracks
  the plotted band-structure x-axis) and the cumulative total. Labels are the
  per-segment names (0.8.4 prettify) with PRIMES PRESERVED — a route's identity
  is its DIRECTION (Γ→X and Γ→X′ are different walks to wrapped-equal points),
  unlike the dedup'd jump row. The lattice route (Source 2) declares no path
  and clears the table.
- **`path_s` FloatProperty (0..1, `_on_path_s_changed`).** s·total → locate the
  segment by cumulative length → t within it → q = (1−t)·q_start + t·q_end
  (FRACTIONAL interpolation, phonopy's own band convention) → write q1/q2/q3
  UNDER the existing suppress guard → ONE evaluation, ONE redraw via the
  SHARED `_engine_evaluate_and_refresh` (reused, not duplicated). Walking
  straight through Γ where segments meet is seamless (shared endpoints are
  bitwise-identical from both sides).
- **ONE-WAY FLOW.** The slider writes q; q is NEVER back-projected onto the
  path (it self-intersects at Γ — projection is ill-posed). Moving q1/q2/q3
  directly, or clicking a jump button, flips the readout to "off-path";
  grabbing the slider resumes on-path. A "Walk path" row (engine source, only
  when a path exists via a same-material sibling band) shows the s slider and
  the readout `SEG_START → SEG_END · NN%` from the current segment's label pair.
- **Honest caveat (tooltip + here):** Band N is the Nth SORTED eigenvalue at
  each q; walking through a band crossing follows the SORT ORDER, not the
  connected band. Band-following (max-overlap connectivity) is future work.
- **Job schema:** engine jobs gain OPTIONAL `path_s` (serialized alongside
  q_frac). `q_frac` remains AUTHORITATIVE on replay — path_s is a UI
  restoration hint only, and replay reconstructs from the phonopy.yaml + FC
  and does NOT require the band sibling to exist.

### Tests (bpy-free + real datasets)

- NEW `tools/test_path_walk.py` (5): the segment table on a REAL 14-segment
  band.hdf5 (total > 0, primes preserved, cumulative strictly
  increasing, s=0→Γ, s=1→V2′ exactly, the seg3/seg4 shared-Γ boundary
  bitwise-seamless); interpolation == closed-form to 1e-12 and evaluate_dq_np
  deterministic on the walk's q; the SnSe2 connected-path regression; the
  one-way-flow flag (direct q → off-path, slider → on-path + overwrites q);
  NEGATIVE CONTROL (fail-then-pass) — the pre-0.8.5 state.py (from the 0.8.4
  snapshot) has no segment-table API, the current one builds 14.
- `tools/test_job_roundtrip.py` (+1, now 11): a job's `path_s` round-trips and
  restores; replay from q_frac WITHOUT the band sibling (path_s inert).
- Headless (4.5.0, exit 0): SnSe2 engine (Source-1 sibling) → has_path, 3
  segments; walking s ∈ {0, .25, .5, .8, 1} carries q along Γ→M→K→Γ exactly
  (engine cache follows, readout tracks each segment); direct q → off-path,
  slider → on-path; one frame rendered while walking.

### Forward map

Band-following (max-overlap connectivity) — resolve band crossings so branch N
follows the CONNECTED band through a crossing rather than the sort order — is
the named future refinement.

Wheels 11/11 byte-identical; parity untouched (17+2 vs core 0.6.86); core
`../phonon-core` untouched.

## 0.8.4 — jump targets from BandPath.labels, not tick_locations()

FIELD DEFECT (0.8.3 acceptance, private triclinic dataset): the jump row showed
Γ, "X|Y", "Z|R₂", "T₂|U₂", "V₂|Γ" — five buttons with COMPOUND labels that are
not point names at all, and the one labelled "X|Y" actually jumped to X.
ROOT CAUSE: `remember_band_hsp` read `BandPath.tick_locations()` — matplotlib
x-axis TICK strings, which FUSE path discontinuities into 'X,|Y'-style plot
furniture — and paired each with the segment's last q. NOT the 0.8.1 dedup
(that logic is correct and unchanged) and NOT basis-related. (Architect note:
this was misdiagnosed as wrapped-q merging one turn earlier; the real cause is
the data source.)

- **The fix — the RIGHT BandPath field.** `state.remember_band_hsp` now builds
  (label, q) pairs from `BandPath.labels` — the PER-SEGMENT `(start, end)`
  high-symmetry point NAMES (the truth). Each segment i contributes BOTH
  endpoints: `(labels[i][0], q_points[seg_start])` and
  `(labels[i][1], q_points[seg_start + n − 1])`. These go to the UNCHANGED
  `assemble_jump_targets` (wrapped-q dedup, first spelling wins), so primed
  variants (X′, Y′, …) absorb onto their unprimed points. hsp.py's two Sources,
  the provenance line, the dedup, and the ENGINE wiring all stay as shipped.
- **Label prettify (pure, unit-tested): `state._prettify_point_label`.** Strips
  a trailing ','/whitespace (the comma-separated BAND_LABELS artifact, e.g.
  'X,' → 'X'); strips '$'; '\Gamma' → 'Γ'; '\prime' → '′'; collapses TeX
  sub/superscript braces ('$R_2$' → 'R2', '$X^{\prime}$' → 'X′'). Anything
  unrecognised is left VERBATIM — never invent a name. (The existing display
  `units.prettify_label` is idempotent on these clean names, so ui.py is
  untouched.)
- **RESULT (real 14-segment triclinic band.hdf5):** the jump row is EXACTLY
  [Γ, X, Y, Z, R2, T2, U2, V2] with q's matching the SeeK-path table bitwise
  (X=(½,0,0), Y=(0,½,0), Z=(0,0,½), R2=(−½,−½,½), T2=(0,−½,½), U2=(−½,0,½),
  V2=(½,−½,0)); the primed points wrap onto their unprimed twins and NO
  compound '|' label survives. SnSe2 (connected path, 3 segments) is unchanged
  at [Γ, M, K] — the row Axel accepted does not regress.

### Tests (bpy-free, tools/test_jump_targets.py 17 → 21)

+4: `_prettify_point_label` units (trailing comma, '$R_2$'→'R2', prime,
verbatim fallback); the real triclinic row == [Γ,X,Y,Z,R2,T2,U2,V2] with q's to
1e-9 and no '|'; the SnSe2 regression; NEGATIVE CONTROL (fail-then-pass) — the
verbatim pre-edit tick_locations()-based builder yields compound '|' labels and
misses Y/R2/T2/V2 as distinct points, while the labels-based builder does not.
Headless (4.5.0, exit 0): register + triclinic band load → clean row; SnSe2 engine
(Source 1 sibling) → [Γ,M,K] / band.hdf5.

### Ledger line

TICK labels are PLOT FURNITURE — the matplotlib x-axis strings fuse path
discontinuities into compound 'X,|Y' tokens and drop intermediate points; they
are NEVER a source of physics point names. Enumerate high-symmetry points from
`BandPath.labels` (per-segment start/end), not `tick_locations()`.

Wheels 11/11 byte-identical; parity untouched (17+2 vs core 0.6.86); core
`../phonon-core` untouched.

## 0.8.3 — close the honesty gap in the lattice jump route (slab vs 3D bulk)

ARCHITECT DEFECT (the 0.8.2 PROMPT, not the implementation): the documented
2D-only limit — "3D returns `[]`; no spglib, no guessing" — was FALSE. The
2D detector is "c* ⟂ a*,b*", which every ORTHOGONAL 3D family satisfies, so the
route already returned in-plane points for 3D crystals, labelled with a 2D
Bravais name. Measured with the 0.8.2 code:

    simple cubic / tetragonal   -> bravais "square",      targets [Γ, X, M]
    orthorhombic                -> bravais "rectangular", targets [Γ, X, Y, S]
    3D hexagonal (graphite)     -> bravais "hexagonal",   targets [Γ, M, K]

The q values are NOT wrong — they are genuine k_z=0 high-symmetry points of
those 3D zones, correctly labelled per S&C — but they are INCOMPLETE (no
R/Z/A/L/H) and the provenance line named a 2D Bravais TYPE for a 3D crystal.
Physics fine, claim false. 0.8.3 fixes the claim and the labelling WITHOUT
weakening the feature.

- **A real slab detector, from a signal already in hand.** phonopy.yaml carries
  `supercell_matrix` — the very key the engine load sniffs on. `hsp.py` gains
  `read_supercell_matrix` (via the vendored `_yaml_loader`, never re-parsed by
  hand) and `is_slab_supercell`. **EXACT RULE: is_slab iff the THIRD ROW of
  `supercell_matrix` is `[0, 0, 1]`** — phonopy builds `A_super = S·A_prim`
  (lattice vectors as rows), so the supercell's c-vector equals the primitive's
  (no replication, no shear along c) exactly when `S[2] == [0,0,1]`. This
  generalises the common diagonal `[N, M, 1]` slab and accepts in-plane shear;
  it is consistent with the c*=b3 (axis-2) decoupling the classifier already
  assumes. Unreadable → None (treated as the conservative "not a slab" branch).
- **The contract, corrected and honest.** `lattice_jump_targets` gains an
  `is_slab` argument and RETURNS THE SAME in-plane points in every regime (they
  are correct k_z=0 points for slab and orthogonal bulk alike); the new
  `lattice_provenance` shapes the honesty:
    - **slab** (`is_slab True`) → "lattice symmetry (<name>)" — canonical: these
      ARE the 2D BZ points and <name> IS the crystal's Bravais type;
    - **orthogonal 3D bulk** (`is_slab` False or None) → "in-plane (k_z=0)
      symmetry — 3D zone; k_z≠0 points need spglib" — the correct-but-partial
      truth, NO 2D Bravais type claimed for a 3D crystal;
    - **non-orthogonal c*** (triclinic / monoclinic-ish) → `[]` (Γ only), the
      one genuinely-undetectable case, unchanged.
  `operators._set_engine_jump_targets` passes the slab flag (it has the yaml
  path already). Offering correct partial points beats offering nothing OR
  pretending they are the whole story. k_z=0 labels per **Setyawan & Curtarolo,
  Comput. Mater. Sci. 49, 299 (2010)** in BOTH regimes.

### Tests (bpy-free, tools/test_jump_targets.py 13 → 17)

+4: SnSe2 data_task `supercell [8,8,1]` → is_slab True, slab targets/provenance
unchanged from 0.8.2 (the shipped behaviour Axel accepted does not move); a
fabricated cubic bulk `supercell [2,2,2]` → is_slab False, targets STILL
[Γ,X,M], provenance contains "k_z=0" and NOT "square"; non-orthogonal c* → []
(unchanged). NEGATIVE CONTROL (fail-then-pass): the 0.8.2 provenance for a 3D
cubic ("lattice symmetry (square)") is reproduced from `lattice_bravais_name`
and shown to differ from the 0.8.3 honest line. (`test_lattice_3d_returns_empty`
comment corrected — [] is now the non-orthogonal-c* case, not "2D-only".)

### Blender headless acceptance (4.5.0; run, exit 0)

Three loads: SnSe2 slab with a sibling band.hdf5 → `band.hdf5` (Source 1,
unchanged); SnSe2 slab, phonopy.yaml+FC alone → `lattice symmetry (hexagonal)`
(Source 2 slab, unchanged); the SAME material with its supercell third row
edited to `[0,0,2]` (a fabricated 3D bulk) → same `[Γ,M,K]` points but
provenance `in-plane (k_z=0) symmetry — 3D zone; k_z≠0 points need spglib`.

Wheels 11/11 byte-identical; parity untouched (17+2 vs core 0.6.86); core
`../phonon-core` untouched.

## 0.8.2 — engine jump targets from the MATERIAL, never session load order

FIELD DEFECT (0.8.1 acceptance): loading ONLY a phonopy.yaml showed a bare Γ
jump row. **Root cause is NOT the 0.8.1 wrapped-q dedup** — that is verified
correct and RETAINED (fed SnSe2's raw `['$\Gamma$','M','K','$\Gamma$']` it
still returns exactly `[Γ, M, K]`) — but `operators.py`'s post-load block,
which fired `remember_band_hsp(data)` only `if data_kind == "BAND"`. So the
ENGINE navigator's jump row was populated only when a band file happened to be
loaded EARLIER in the same Blender session. Load order must not decide what the
panel offers. Two layered, material-derived sources now feed the (unchanged)
`assemble_jump_targets`, which dedups them by wrapped q.

- **NEW bpy-free module `phonon_vision/hsp.py`** (chosen over adding to
  state.py: it is pure numpy + LAZY vendored parsers, so the tests import it
  DIRECTLY without the bpy stub state.py needs, and state.py stays focused on
  Blender state). The engine load path (`operators._set_engine_jump_targets`,
  a new `elif data_kind == "ENGINE"` branch) ALWAYS refreshes the store from
  the loaded material, so a prior material's targets can never leak.
- **SOURCE 1 (authoritative)** `hsp.discover_sibling_band`: a band file next to
  phonopy.yaml (`band.hdf5` then `band.yaml`, first hit wins — the engine's own
  sibling-FORCE_CONSTANTS idiom), parsed with the vendored parsers, accepted
  only if it is the SAME material (symbols + reciprocal lattice `np.allclose`,
  the exporter's idiom). NON-FATAL at every step — absent / unparseable (h5py
  may be missing) / different material → log once and fall through to Source 2;
  a bad sibling never blocks an engine load that already succeeded.
- **SOURCE 2 (fallback)** `hsp.lattice_jump_targets`: high-symmetry points from
  the reciprocal lattice alone, **pure numpy — Blender ships neither scipy nor
  spglib**. **2D-ONLY BY DESIGN**: if c* is not decoupled (the third reciprocal
  vector not orthogonal to the other two) the material is treated as 3D and the
  route returns `[]` (Γ only) — classifying the 14 3D Bravais lattices to
  spglib grade is out of scope, and **a WRONG label is worse than no label**.
  For the decoupled-c* (layered) class — SnSe2, hBN — the in-plane b1,b2
  classify (rel tol 1e-4 lengths, 0.1° angles), conventions per **Setyawan &
  Curtarolo, Comput. Mater. Sci. 49, 299 (2010)**:
  hexagonal→Γ,M(½,0,0),K(⅓,⅓,0); square→Γ,X(½,0,0),M(½,½,0);
  rectangular→Γ,X(½,0,0),Y(0,½,0),S(½,½,0); centred-rect→Γ,X,S [S approx];
  oblique→Γ only. **VERIFIED on the real SnSe2 phonopy.yaml: it is hexagonal
  with ∠(b1,b2)=60.0°, and the lattice route's M, K reproduce the q's in its
  own band.hdf5-labelled path to M=0.0, K=3.33e-07 rlu (< 1e-6).**
- **Data panel provenance** stays visible: "Jump targets: band.hdf5" (Source 1)
  / "lattice symmetry (hexagonal)" (Source 2) / "Γ only" (oblique or 3D).
- **assemble_jump_targets (0.8.1) UNCHANGED** — it is what makes the two
  sources composable (Source 2 returns Γ first; the wrapped-q dedup absorbs it
  under the standing Γ).

### Tests (bpy-free, tools/test_jump_targets.py 4 → 13)

+9: SnSe2 lattice route == its band.hdf5 M/K to 1e-6 (the load-bearing one);
square/rectangular/centred + hexagonal 60°&120° label sets; oblique → [Γ];
3D-coupled → [] and `dim_hint=3` → []; the same-material guard (unit + real
sibling: SnSe2 band.hdf5 accepted, a C₂ material rejected → falls back);
Source-1 raw `$\Gamma$` through assemble still [Γ,M,K] (0.8.1 intact).
NEGATIVE CONTROL (fail-then-pass): with nothing remembered, `assemble([])` is
a bare Γ (the defect); `assemble(lattice(hex))` is [Γ,M,K] (the fix).

### Blender headless acceptance (4.5.0; run, exit 0)

Scenario 1 — FRESH session, load ONLY `data_task/phonopy.yaml`: jump row
`[Γ, M, K]` via `band.hdf5`; clicking M sets q=(0.5,0,0), readout follows
(1.900 THz). Scenario 2 — `phonopy.yaml` + `FORCE_CONSTANTS` copied ALONE to a
temp dir (no band sibling): jump row STILL `[Γ, M, K]`, now via
`lattice symmetry (hexagonal)`.

## 0.8.1 — engine jump row: deduplicate by q, not label string

FIELD DEFECT (0.8.0 user acceptance): the engine Mode Navigator's **Jump row
showed TWO Γ buttons**. The standing Γ was seeded as the literal glyph
`"Γ"` (ui.py) while the band path's start-Γ arrives raw from
`BandPath.tick_locations()` — on the real SnSe2 band file that raw spelling is
`"$\Gamma$"` (TeX with `$` delimiters) — so the old **label-string** dedup
(`lab not in [t[0] for t in targets]`) never matched the two spellings and
rendered the path Γ beside the standing one.

- **THE RULE: q is identity, the first label wins.** New pure/bpy-free
  `state.assemble_jump_targets(band_hsp) -> [(label, q), ...]`: starts with
  the standing Γ `("Γ", (0,0,0))`, then appends each band-file `(label, q)`
  only if its q — **each component wrapped to its image in [-0.5, 0.5)**, so
  q and q+G (reciprocal lattice vector) are the same point — differs from
  every already-accepted target's wrapped q by more than 1e-6 in max-norm.
  The label plays NO part in identity, so the standing Γ absorbs any path Γ
  whatever its raw spelling; and the FIRST spelling of a q wins (a `M'` at
  −0.5 merges into the `M` at +0.5, keeping "M" and its q). ui.py now calls
  this instead of the inline seed + label-string loop.
- **Zero physics, zero UI surface beyond the corrected row.** Wheels 11/11
  byte-identical (phonon_core-0.2.0 not rebuilt); parity contract untouched
  (17+2, exit 0 vs core 0.6.86); core `../phonon-core` untouched.
- **Tests** (bpy-free, NEW `tools/test_jump_targets.py`, 4; drives the REAL
  shipped `state.assemble_jump_targets` via a minimal bpy stub — the function
  is itself pure): (1) `["\Gamma", "M", "K", "\Gamma"]` at their q → exactly
  `[Γ, M, K]`, the Γ entry the standing one, one Γ not two; (2) zone-boundary
  alias `M'(−0.5,0,0)` after `M(0.5,0,0)` → merged, "M" and q=(0.5,0,0)
  retained; (3) same label "X" at DISTINCT q `(0.5,0,0)` vs `(0,0.5,0)` →
  BOTH kept (label is not identity). NEGATIVE CONTROL (fail-then-pass, in
  suite): the verbatim pre-edit label-string assembly produces
  `[Γ, "\Gamma", M, K]` — two Γ — and so FAILS test (1)'s assertion, shown
  side by side with the shipped function passing on identical input.
  Confirmed on REAL DATA: `fatbands_SnSe2/band.hdf5` raw labels
  `['$\Gamma$', 'M', 'K', '$\Gamma$']` → deduped `['Γ', 'M', 'K']`,
  Γ count 1.

## 0.8.0 — ENGINE MODE: exact D(q) at ANY q, with live q-scrubbing

MINOR bump, justified: the add-on gains its FIRST capability beyond file
playback — load phonopy.yaml + FORCE_CONSTANTS and evaluate EXACT modes at
any fractional q via the vendored A2 numpy engine (wheel 0.2.0, NOT rebuilt
this release; parity contract unchanged at 17+2; platform wheels 11/11
byte-identical). Core `../phonon-core` stays 0.6.86. Zero new physics:
motion.py's Bloch phase always handled arbitrary fractional q (q·R =
2π·q_frac·n); kinematics untouched.

### The feature

- **LOAD**: the .yaml branch content-sniffs BEFORE the band/mesh attempts —
  a top-level `supercell_matrix:` key (line-anchored, first 64 KB) marks a
  phonopy STRUCTURE file and routes to `motion.load_engine_source` (sibling
  FORCE_CONSTANTS / force_constants.hdf5 discovery mirrors the engine's own
  rule; the RESOLVED path is displayed and serialized). Engine errors —
  missing FC, parse failure, and the A2 **BORN refusal** ("A2 has no
  non-analytic correction yet; use the A1 engine") — surface VERBATIM as the
  operator error. Data panel: "ENGINE (A2 numpy) — exact D(q) at any q" +
  the FC file name. NEGATIVE CONTROL (headless Blender, fail-then-pass):
  the pre-0.8.0 tree loading data_task/phonopy.yaml dies with the CONFUSING
  band-yaml fall-through ("missing 'phonon' section"); post-edit → kind=
  ENGINE, source=EngineSource.
- **LIVE q-SCRUBBING (the acceptance feel)**: `q_frac_1/2/3` FloatProperties
  (r.l.u., soft ±1, precision 4), each update re-evaluating D(q) through the
  module-level engine cache and refreshing the readout + animation source
  CONTINUOUSLY — no Evaluate button. Measured: raw evaluation 0.18
  ms (architect's spike: 0.07 ms — machine delta), full property-set →
  evaluate → scene refresh → overlay chain 2.4 ms median over a 20-set
  scrub in headless Blender. `EngineSource` (bpy-free, in motion.py — the
  engine-call-chain charter) duck-types the exact `get_mode`/`material`/
  `num_qpoints` seam file sources satisfy, so `update_atom_positions`, the
  frame handler, the freq readout and the imaginary badge consume engine
  modes UNCHANGED (threshold semantics identical — signed THz).
- **MODE NAVIGATOR (engine source)**: band prev/next + slider reused; the
  readout gains the honesty line "exact engine mode — q = (a, b, c) r.l.u."
  (the navigator's Stage-3 language); **no fake 2D-path position is ever
  shown for an engine q** (the q-index row is hidden). Jump buttons: Γ
  always; the high-symmetry labels of the last BAND file loaded this
  session (captured via `state.remember_band_hsp` from
  `BandPath.tick_locations`) become additional jump targets that set q to
  those q_frac triples — one refresh per jump (suppressed writes + a single
  `_on_engine_q_change`).
- **ANIMATION**: engine modes route (q_frac, eigenvector) into the SAME
  motion/frame_handler path as file modes (the `get_mode` seam — no routing
  change was needed; frame_handler.py and api.py are UNTOUCHED). Changing q
  mid-animation re-sources displacements exactly like q_index does today.
- **JOB SCHEMA v3**: optional top-level `engine` block
  `{source: "engine", phonopy_yaml, force_constants (RESOLVED), q_frac [3],
  band_index}`; writer emits 3, reader accepts 1–3. **Eigenvectors are
  NEVER serialized** — the FC file is the source of truth; the bpy-free
  `job.load_engine_source_from_job` reconstructs + re-evaluates, and replay
  matches a fresh evaluation at **0.0 (bitwise)** — asserted at 1e-12 in
  tools/test_job_roundtrip.py. Missing FC at load → named
  FileNotFoundError. apply_job_to_scene re-syncs the loaded engine cache to
  the job's q (prop writes stay suppressed — one evaluation, no callback
  storm).

### Tests (bpy-free counts: bonds 25, motion 7→8, job 7→10)

- test_job_roundtrip +3: engine-block roundtrip (incl. the cache re-sync
  assert), replay-vs-fresh (bitwise 0.0), missing-FC named error. The v2
  bonds test's `schema_version == 2` literal was updated to the CONSTANT
  (the 0.6.79 lesson — a literal re-arms the trap at every bump; my own new
  test initially repeated it and was fixed the same way).
  NEGATIVE CONTROL: the engine-roundtrip test vs the PRE-edit job.py (from
  the 0_7_14 snapshot) fails `KeyError: 'engine'`, exit 1; post-edit ALL 10.
- test_motion +1: `evaluate_dq_np` output (vendored wheel) at incommensurate
  q = (0.137, 0.291, 0) through `positions_at_frame`; the Bloch relation
  reconstructed from two quarter-cycle frames (Z = d(φ=0) − i·d(φ=π/2)) and
  asserted EXACTLY between cells related by integer n:
  Z_n2 = Z_n1·e^{2πi q·Δn} on 12 cell pairs at 1e-12 relative.
- SANCTIONED RIDER (audit defect 0.7.14): the wheel-build smoke toy-writer
  interpolated numpy scalars with !r — under numpy>=2 that emits
  "np.float64(3.1)" and poisons the YAML. Every scalar now passes through
  float(), and the writer ASSERTS the generated YAML contains no
  "np.float64". This env: numpy 1.26.4 (repr benign here; the assertion
  guards the >=2 case). Smoke re-run: PASS.

### Headless Blender acceptance (4.5.0; run, exit 0)

Load data_task/phonopy.yaml → kind=ENGINE; freqs @ (0.137, 0.291, 0) =
[1.5274, 2.6208, 3.6107, 3.8001, 4.4857, 5.443, 5.5839, 5.9604, 6.9159] THz
(digit-identical to the bpy-free evaluation); honest readout + badge=False;
20× property-scrub with no error (2.35–2.75 ms/set); job round-trip applied
and ONE FRAME RENDERED via the job path (1,468,678 B PNG). Known trap
re-confirmed: FloatProperty storage is float32 (q = 0.137 stores as
0.13699999…; readout shows 4 decimals — resolution ~1e-7 r.l.u., physically
irrelevant).

## 0.7.14

Wheel plumbing ONLY — no feature, nothing user-visible beyond the wheel
version stamp. Engine-mode UI is 0.8.0. Core tree `../phonon-core` untouched
(stays 0.6.86). h5py + PyYAML wheels byte-identical (10/10).

### The headline: the FIRST REPRODUCIBLE wheel build

- **NEW `tools/build_vendored_wheel.py`** — until now the vendored wheel was
  built BY HAND (0.1.4 had no committed recipe; the subset existed only
  implicitly, inside the artifact). The recipe: assemble the staging package
  from `--core-src` using the SHARED subset list (**NEW `tools/_subset.py`**
  — ONE constant, TWO consumers: this builder and the parity guard, no
  drift), write the two packaging stubs from templates IN the script, build
  (PEP-517 `python -m build --wheel`, with a `pip wheel` fallback producing
  identical package contents), print the INNER-FILE md5 manifest (the
  reproducibility contract — the zip's own md5 varies with timestamps; the
  contents must not), then SMOKE the built wheel in a subprocess whose
  sys.path front-loads a shadow `phonopy` raising ImportError.
- **Wheel `phonon_core-0.1.4` → `phonon_core-0.2.0`**: now carries the A2
  pure-numpy D(q) engine (`engine/dq_numpy.py`, byte-identical to core
  0.6.86, md5 80bef1ef) + a NEW templated `engine/__init__.py` stub exposing
  `load_engine_np / evaluate_dq_np / evaluate_dq_np_batch` ONLY. **The
  phonopy-backed A1 engine is DELIBERATELY ABSENT**: phonopy cannot be a
  dependency inside Blender, and a module that can never import is worse
  than one that does not exist (core's engine/__init__.py re-exports A1
  names, so it is not vendorable as-is — hence the template). 0.1.4 DELETED.
- **Stub pins**: the single 0.1.4-era pin `07478e09` is RETIRED; the two
  0.2.0 stubs are pinned in `tools/_subset.py` — top-level `d3638416…`
  (`__version__ = "0.2.0"`, the unchanged 4 subset import lines + __all__),
  engine `4bdbbc56…`. Re-pin in the same commit whenever a template moves.

### Parity contract extended (16+1 → 17+2, set-complete)

- `tools/check_vendored_parity.py` now verifies the SHARED list: all 17
  subset modules byte-identical to `--core-src` + BOTH pinned stubs, AND
  set-completeness both ways (a wheel module not in the list is a break
  exactly like a missing one — exit 2). Run TODAY against `../phonon-core`
  (0.6.86): **"17 modules byte-identical to core 0.6.86 + 2 pinned packaging
  stubs", exit 0** — full output in the ship report.

### Proofs

- **PHONOPY-ABSENT SMOKE (the point of A2, fail-then-pass):** the smoke
  builds a tiny 2-atom toy (hand-written phonopy.yaml + translationally
  consistent isotropic-spring FORCE_CONSTANTS with exact ASR — core's Tier-1
  pattern), then under the phonopy shadow: `import phonon_core.engine`,
  `load_engine_np`, `evaluate_dq_np` at 3 q → finite SIGNED frequencies,
  unit-norm eigenvectors, Γ acoustic [−0, 0, 0] THz + real optical gap
  (9.483 THz). AGAINST 0.1.4 (before deletion): `ModuleNotFoundError: No
  module named 'phonon_core.engine'`, exit 1 — the control bites. Against
  0.2.0: PASS, exit 0.
- bpy-free suites green AGAINST THE NEW WHEEL (test_motion zip-imports it):
  bonds 25, motion 7, job round-trip 7; `check_stamps` exit 0 at 0.7.14
  (and correctly FAILED exit 1 mid-edit while the CHANGELOG heading lagged
  the other two stamps — the guard works).
- Blender headless (4.5.0): extension rebuilt + reinstalled with the new
  wheel; `pv_headless_smoke.py` unchanged-behavior PASS + an A2-in-Blender
  probe (`from phonon_core.engine import load_engine_np`) — outputs in the
  ship report.

### The 0.1.x-expectation enumeration (consumers first — the 0.6.79 lesson)

Every code/config place the 0.1.x wheel expectation lived, all updated:
`blender_manifest.toml:19` (wheel entry), `tools/test_motion.py:33`
(zip-import filename), `tools/check_vendored_parity.py:42` (the retired
stub pin). Checked and NOT version-bearing: `tools/check_stamps.py` (no
wheel logic) and the `ui.py` ~89 wheel guard (`_PC_IMPORT_OK` is an
import-success flag, version-agnostic — nothing to update).
`scene_setup.py:309` says "phonon-core 0.1.4+" as a HISTORICAL since-marker
(comment), deliberately left.

## 0.7.13

Catch-up release: parity seam repaired + parity RESULT recorded, Blender 5.x
audit, ASR-fixed-dataset regression check. **No physics moved and the wheel
was NOT rebuilt** — `wheels/` is byte-identical to 0.7.12 (parity held; the
Level-2 core arc never touched a vendored module).

### Parity seam repaired (the core-tree rename broke every documented command)

- The sibling core tree was renamed `phonon_core-0.6.76-src` → `phonon-core`
  (post-core-0.6.84 housekeeping; zero in-core-tree references existed, but
  every parity command documented HERE pointed at the dead name).
  `--core-src` is a REQUIRED argument, so nothing ever mis-resolved silently —
  the commands just died. Updated: the docstring invocation in
  `tools/check_vendored_parity.py` (the only code line that carried the old
  path), CLAUDE.md's siblings/discipline/test-command references. `--core-src`
  stays REQUIRED with no default (explicit beats implicit for a cross-tree
  check). Historical CHANGELOG entries keep their original paths with a
  bracketed `[tree since renamed ../phonon-core]` annotation — the old text
  was accurate when written; history is annotated, not rewritten.
- NEGATIVE CONTROL (fail-then-pass): the OLD documented command
  (`--core-src ../phonon_core-0.6.76-src`) FAILS — `[parity] not a core src
  tree (no src/phonon_core)` — and the NEW (`--core-src ../phonon-core`)
  exits 0. Both outputs recorded in the ship report.
- **PARITY RESULT (recorded, reproduced today):** vendored physics verified
  at parity with core **0.6.85** — 16/16 modules byte-identical (kinematics,
  animation, expand, units, species, radii, projection, types + all 8 parser
  files) + the pinned packaging stub (`07478e09`), exit 0 UNPIPED. The
  Level-2 arc (0.6.77–0.6.85) moved exporter/json_v1/viewer/launcher/serve
  and added engine/ — none vendored. CLAUDE.md provenance line refreshed.
- Out of scope, flagged: `HPC/acceptance/phonon_vision_pkg/` (a sibling
  acceptance snapshot of the installed 0.7.11) also carries the old path; it
  is a frozen shipped artifact and was deliberately NOT edited.

### Blender 5.x audit (outcome: keep blender_version_min = 4.5.0)

- Audited the add-on's API surface against the 5.x line (static + release
  notes; only Blender 4.5.0 is installed on this machine — no 5.x binary to
  smoke against, stated plainly): registration/PropertyGroup/properties,
  frame_change_pre + @persistent, `instance_type="VERTS"` DupliVerts, the
  overlay (render STAMPS, not gpu draw-handlers), Principled BSDF sockets
  (presence-keyed fallbacks already in place), engine selection
  (`try BLENDER_EEVEE_NEXT except TypeError: BLENDER_EEVEE` — 5.0 renamed
  NEXT back to BLENDER_EEVEE, the existing fallback covers it), Freestyle
  (present through 5.1), no bpy.props dict-access (removed in 5.0;
  grep-verified unused), manifest `schema_version = "1.0.0"` (still current).
- **THE BLOCKER IS THE WHEELS, exactly as CLAUDE.md's standing checklist
  predicted:** Blender 5.0 bundles Python 3.11, but 5.1+ (and 5.2 LTS, July
  2026) bundle **Python 3.13** — all ten cp311 binary wheels (h5py 3.14.0,
  PyYAML 6.0.2 × 5 platforms) are ignored there; only phonon_core-0.1.4
  (py3-none-any) survives. A true 5.2 migration = the cp313 wheel refresh
  (checklist items 2–4), which changes `wheels/` — frozen by THIS release's
  scope — so it is deferred to its own bump.
- **Decision:** `blender_version_min` STAYS `"4.5.0"` (4.5 LTS is supported
  until July 2027; the Python code needs nothing for 5.x, and raising the pin
  without cp313 wheels strands 4.5 users for a 5.2 that still cannot run).
  Audit recorded in CLAUDE.md's "Blender 5.2 readiness" section.

### ASR-fixed dataset regression check (real corrected data; no code)

- Loaded the regenerated SnSe2 `band.hdf5`
  (Phonopy-examples/fatbands_SnSe2/data_task/, symfc-symmetrized FC — the
  core-0.6.84 physics finding) through the add-on's own loader in headless
  Blender 4.5 and checked the EXACT ui.py badge predicate
  (`mode.is_imaginary and abs(ν) > imag_freq_threshold`, default 0.01 THz):
  Γ readouts `0.000i / 0.000 / 0.000 / 3.357 / 3.357 / 4.885 / 4.885 /
  5.968 / 8.487 THz` — band 0 is sign-honest (`is_imaginary=True`, ν=−1e-06)
  yet badge-suppressed as sub-threshold dust; **zero badges at Γ and zero
  across all 8181 path modes** (global min −0.009385 THz). Mode selection +
  frame-handler animation verified moving atoms (Γ optical 3.357 THz:
  max|Δ| 0.0444 Å; non-Γ q_index 300: 0.1503 Å, frames 1→7).
- Headless gates on the same dataset: `pv_headless_smoke.py` exit 0 (PNG
  rendered, 1,176,477 B); `conformance_headless.py` exit 0 — PASS on 6
  frames, max deviation 1.769e-06 Å vs the bpy-free reference (atol 1e-4).

### Stamps

- 0.7.12 → 0.7.13 (`blender_manifest.toml` version + `bl_info["version"]`);
  `blender_version_min` unchanged at 4.5.0 per the audit decision.

## 0.7.12

Three named concerns (the color feature + two sanctioned riders), each below.

### Per-pair bond color from the side-car (last unapplied field; color parity)

- **Consume the side-car `color` field — cross-tool color parity with the
  navigator 0.6.76.** `color` is the LAST curated per-pair field the add-on
  did not apply (0.7.9 max/replace, 0.7.10 radius_A; `color` was parsed and
  stored since 0.7.10 but inert). Contract (verified in
  `../phonon_core-0.6.76-src/web/index.html` [tree since renamed
  `../phonon-core`, post-core-0.6.84]): a per-pair `color` is a
  DISPLAY-sRGB `#rrggbb` hex, emitted ONLY when that pair is in the
  navigator's custom-color mode (`serializeBondSettings` ~L1824); a pair in
  default mode emits no `color` key. The navigator's two color systems are
  MUTUALLY EXCLUSIVE per pair: `bondColorMode[key] == "custom"` (whole
  cylinder that hex) XOR absent (the per-species two-color split).
- **Semantics (mirrors the navigator's `recolorBonds` per-pair branch):** for
  each bonded pair, a curated color WINS where present — the ENTIRE cylinder
  (both halves) is that hex, overriding the per-species split FOR THAT PAIR
  ONLY; absent color leaves the "Two-color bonds (per species)" behavior
  unchanged (split at the midpoint, each half its atom's species color). The
  global two-color toggle governs only NON-curated pairs; a curated-color pair
  is solid regardless, matching the navigator where a custom-mode pair ignores
  the split.
- **sRGB -> linear is the color-match guarantee.** Blender material Base Color
  and `diffuse_color` are LINEAR; the navigator hex is display-sRGB
  (`hexToRgb01` is raw /255). A curated color is linearized via the standard
  sRGB EOTF (IEC 61966-2-1: `c/12.92` for `c <= 0.04045`, else
  `((c+0.055)/1.055)**2.4`) before assignment, or the Blender color would read
  wrong against the navigator. Pinned numerically: `#7f8fa6` ->
  `(0.21223, 0.27468, 0.38133)` linear (vs the raw sRGB `(0.498, 0.561,
  0.651)` — a ~0.28 difference in red).
- **Implementation reuses the 0.7.9 per-pair batching — no per-bond loop.**
  `bonds.py` (bpy-free): `parse_sidecar` now VALIDATES `color` against
  `^#[0-9a-fA-F]{6}$` (malformed warned + dropped == absent, the navigator's
  import rule); new `pair_colors(spec)` (validated hex per pair),
  `hex_to_linear(hex)` (sRGB EOTF), and `bond_material_plan(color, two_color,
  same_species) -> (n_cyl, style)` — the pure decision that a curated color
  yields `(1, "custom")` (one solid cylinder) and otherwise the prior
  grey/species/split. `scene_setup.create_bond_object` appends one
  `PV.BondColor.<pair>` solid material (Base Color + diffuse_color both linear)
  per curated pair AFTER the grey slot, and routes each bond through
  `bond_material_plan`; a curated pair is a single full cylinder (like the
  same-species collapse — no midpoint seam). The per-frame update is unchanged:
  materials are static (per-polygon `material_index`), only vertices move, and
  the radius batch still keys on the canonical pair.
- **Tests: tools/test_bonds.py +6 -> 25 total, plain python.** color parse
  (valid `#7f8fa6` kept; `#ZZZ`/`7f8fa6`/`#12345` dropped + warned; no-key
  absent); `pair_colors` helper; sRGB->linear color-match PIN (independent EOTF
  reproduction + black/white fixed points + "not the raw fractions");
  `bond_material_plan` override precedence (color wins over split AND over
  two_color-off/same-species); absent-color-keeps-split default (radius-only
  file -> no pair resolves "custom"); color round-trip through
  `json.dumps(spec)`.
- **NEGATIVE CONTROLS (fail-then-pass, scratch edits; restored -> ALL 25 PASS,
  exit 0):** (a) skip the sRGB->linear conversion (return the raw sRGB
  fractions) -> the color-match pin FAILS (exit 1, got `(0.498, 0.561,
  0.651)` vs expected `(0.212, 0.275, 0.381)`); (b) let a colored pair still
  split (`bond_material_plan` no longer returns `"custom"`) -> the override
  precedence test FAILS (exit 1, got `(2, "split")` vs `(1, "custom")`).
- **Headless integration verified** (real SnSe2 `band.yaml`, Blender 4.5.0,
  `--factory-startup`): a curated `Se-Sn` color builds a `PV.BondColor.Se-Sn`
  slot whose linear Base Color equals `hex_to_linear` to float32, ALL bonds
  single-cylinder on that slot; dropping the color restores the two-half
  species split (slots Sn/Se). Cosmetic display-only — bond detection, radius,
  and the kinematics render path (0.7.11 conformance) are untouched.

### Rider 1 (named): manifest `permissions.files` length fix

- **Shorten `permissions.files` 93 -> 58 chars** ("Read phonopy band/mesh
  files and write rendered animations"). Blender 4.5.9's `extension build`
  validator caps permission descriptions at 64 and FATAL-errored on the old
  string; `extension install-file` does NOT enforce that cap (which is why
  0.7.7–0.7.11 installed despite the 93-char value) and the length is
  runtime-inert. Fail-then-pass: at 0.7.11 `blender --command extension build`
  FATAL-errored (`"files": a value no longer than 64 characters expected,
  found 93`); at 0.7.12 it builds a package zip cleanly.

### Rider 2 (named): remove the dead `__init__.py` `__main__` block

- **Delete the duplicated `if __name__ == "__main__": register()` block**
  (two copies, lines 45–50; the long-standing wart flagged in CLAUDE.md).
  Confirmed genuinely unreachable before deleting: Blender always imports the
  add-on as a package (`__name__` is the package name, never `"__main__"`),
  and a direct `python __init__.py` fails at `import bpy` first — not an entry
  point. `register()`/`unregister()` still resolve and run (headless smoke).

## 0.7.11

- **Kinematics render-path conformance — prove the bpy plumbing preserves
  the engine's displacements.** FRAMING: scene_setup builds base positions +
  cell indices via `phonon_core.expand.expand_cell` (Å) and animates via the
  engine's `frame_to_phase` / `compute_bloch_phases` /
  `compute_displacements_at_phase` — all parity-guarded and golden-covered
  in the core line. The UNVERIFIED code was the add-on-local plumbing:
  (1) the `all_base_pos`/`uc_idx` flatten -> custom-prop ->
  reshape(n_slots, 3) round-trip; (2) the per-species
  `foreach_set('co', current_pos[slot_indices])` scatter; (3) the
  displacement clamp. 0.7.11 factors that plumbing into bpy-free
  `motion.py`, conformance-tests it against the tree's own vendored wheel,
  and closes the loop end-to-end inside real Blender. NO user-facing change;
  behaviour byte-identical to 0.7.10 (hex-exact regression golden below).
- **Displacement convention (for the record, with citations):**
  `u_{ℓs}(φ) = A · Re[e_s(q) · exp(i(q·R_ℓ + φ))] / sqrt(m_s)` — the
  mass-weighted eigenvector convention of Dove, *Introduction to Lattice
  Dynamics* (1993), eq. 4.16, in the dynamical-matrix/eigenvector convention
  of Togo & Tanaka, Scr. Mater. 108, 1 (2015) (phonopy), with Bloch phase
  `q·R_ℓ = 2π·q_frac·n` exactly (fractional q, integer cell index n).
  **THE CLAMP IS NON-PHYSICAL:** it rescales each over-long displacement
  vector independently to |u| = cap, distorting the eigenvector amplitude
  ratios `e_k/sqrt(m_k)` between atoms that define the mode shape; it exists
  to keep extreme-amplitude animations legible. The physics-faithful path is
  clamp OFF — tests pin that clamp=False matches the engine and clamp=True
  only ever shrinks vectors (exact-cap length, direction preserved,
  in-cap vectors untouched bitwise).
- **NEW bpy-free `motion.py`** (pure numpy, engine imports lazy — the
  bonds.py discipline): `pack_geometry`/`unpack_geometry` (C-order flatten/
  reshape PINNED and documented), `displacements_at_frame` /
  `positions_at_frame` (the 0.7.10 engine call chain + clamp lines moved
  VERBATIM, same float ops in the same order; NO unit scaling — 1 BU ≡ 1 Å),
  and `species_layout` (the exact flat `co` array `foreach_set` consumes).
  scene_setup's remaining bpy work in the animate path is ONLY: read custom
  props, call motion.*, foreach_set. The displacement is exposed separately
  from positions because the arrow builder consumes it (`current_pos =
  base + disp` stays the identical op). Module-top dead imports removed
  (`phonon_core as pc` — already unused — and the kinematics trio the
  refactor orphaned); the bond-build site unpacks through the same pinned
  `motion.unpack_geometry`.
- **Tests: tools/test_motion.py NEW (7), plain python; the engine comes from
  the tree's own VENDORED WHEEL (zip-imported), so the conformance target is
  exactly what ships.** ROUND-TRIP (pack->unpack exact, dtypes pinned);
  ENGINE CONFORMANCE over a full period, frames 0..frames_per_cycle
  inclusive, for BOTH a Gamma and a non-Gamma mode (q = (0.5, 0.25, 0)) on a
  2x2x1 `expand_cell` supercell, reference recomputed INDEPENDENTLY as
  `base + A·Re[engine disp at unit amplitude]` with the engine's own
  `compute_bloch_phases` (atol 1e-13, rtol 0; non-trivial Bloch envelope
  asserted); Å->BU PIN (mass-4 / amplitude-2 / unit-eigenvector fixture
  moves the atom by EXACTLY 1.0 BU, bitwise); CLAMP bounded-distortion
  contract; SCATTER interleaving pin (x,y,z per vertex in slot_indices
  order); 0.7.10 REGRESSION GOLDEN — `current_pos` captured from the
  PRE-refactor `update_atom_positions` lines (float64-hex, clamp OFF and
  clamp engaged on exactly 2 of 8 slots), asserted with NO tolerance.
- **NEGATIVE CONTROLS (fail-then-pass, scratch copies; all restored ->
  ALL 7 PASS, exit 0):** (a) F-order reshape in `unpack_geometry` -> the
  round-trip FAILS (exit 1); (b) disp scaled by 1.0001 -> the Å->BU pin
  FAILS (atom moves 1.0001 BU, exit 1); (c) Bloch phase zeroed -> the
  NON-Gamma conformance FAILS at frame 0 while the Gamma test STILL PASSES
  (proves the per-cell phase envelope is exercised, exit 1); (d) `uc_idx`
  cell rows permuted -> non-Gamma conformance FAILS (exit 1); (e) END-TO-END
  gate control: `current_pos` scaled by 1.001 in scene_setup ONLY (the
  bpy side the reference does not share) -> conformance_headless FAILS
  (max|world − reference| ≈ 4e-2 Å > atol, exit 1).
- **NEW tools/conformance_headless.py — headless end-to-end gate.** Builds
  the real add-on scene for a fixture JSON (input phonopy file, supercell,
  q/band, amplitude, frames), advances frames via `scene.frame_set` (the
  registered frame handler = the actual render path), reads every atom
  object's vertex WORLD positions (`obj.matrix_world @ v.co`), and asserts
  allclose to `motion.positions_at_frame` (atol 1e-4 Å — mesh coordinates
  are float32). Robust to a versioned checkout dir name and to Blender's
  exception-swallowing (every failure routes through sys.exit; probe
  confirmed code propagation). Includes a dead-handler belt (atoms must
  MOVE between non-cycle-aligned frames). RUN IN-HARNESS on Blender 4.5.0
  (matches `blender_version_min` 4.5.0): WSe2 band fixture, NON-Gamma
  q = (0.4187, 0.1625, 0), band 4, 2x2x1, 6 frames spanning a cycle — PASS,
  max deviation 1.9e-6 Å, exit 0; clamp-ON variant (amplitude 30, cap
  0.3 Å) — PASS, exit 0. HPC acceptance invocation (Blender 4.5 LTS):
  `blender --background --python tools/conformance_headless.py -- --job
  <fixture.json>`.

## 0.7.10

- **Consume side-car `radius_A` — absolute-Å bond geometry (cross-tool
  geometric congruence with the navigator).** Context: navigator 0.6.76 made
  all display geometry absolute Å and its side-car now carries per-pair
  `radius_A` (AUTHORITATIVE, bond-cylinder radius in Å) alongside the DERIVED
  legacy `radius_scale` (= radius_A / 0.15, kept so 0.7.9-era readers keep
  working). At 0.7.9 Blender read only `radius_scale` against its own base
  (`bond_thickness × 0.06`), so a curated file rendered different cylinder
  proportions than the navigator. 0.7.10 consumes `radius_A` as an absolute
  Å cylinder radius (1 Blender unit ≡ 1 Å, verified: positions come from
  `expand_cell` in Å and are written verbatim, no scale factor anywhere), so
  ONE file yields geometrically congruent bonds in BOTH tools. Ball radii
  already match (both tools use 0.34 × Cordero: Sn 0.47, Se 0.41 Å) — atom
  sizing untouched. Per-pair color materials and kinematics-golden
  conformance remain SEPARATE later versions.
- **Resolution order mirrors the navigator's 0.6.76 import** (new bpy-free
  `bonds.resolve_radius`, per pair): `radius_A` finite -> clamp [0.02, 1.0] Å
  used VERBATIM as the cylinder radius — `props.bond_thickness` is FULLY
  BYPASSED for that pair (the display analogue of a loaded spec bypassing
  `bond_tolerance` for detection); else `radius_scale` finite ->
  base × scale (the legacy 0.7.9 path, base = bond_thickness × 0.06); else
  base. `parse_sidecar` surfaces `radius_A` (clamped [0.02, 1.0] on read,
  the navigator's import clamp; non-numeric warns and is dropped; survives
  the normalized-state round-trip the operators store). `bond_thickness`
  continues to govern ONLY bonds without a curated `radius_A`.
- **Geometry path**: `create_bond_object` stashes per-bond `pair_radius_A`
  (-1.0 sentinel = absent; parse clamps real values ≥ 0.02 so the sentinel
  cannot collide) next to the 0.7.9 `pair_radius_scale`; the per-frame
  update resolves each contiguous per-pair batch through
  `bondcore.resolve_radius` — the 0.7.9 per-pair batching is reused, the
  vectorized kernel is unchanged, and the value handed to
  `_compute_bond_verts_vectorized` is the RADIUS (ring verts at distance r
  from the axis; a 0.30 Å spec yields a 0.60 Å-diameter cylinder).
- **UI**: the "Bond thickness" slider greys out while every curated pair
  carries `radius_A` (any navigator ≥ 0.6.76 export) — it then drives
  nothing drawn under replace semantics — mirroring the `bond_tolerance`
  grey-out (new `bonds.all_pairs_have_radius_A`). A mixed/legacy spec keeps
  the slider live (it still governs radius_scale-only pairs).
- **Tests: tools/test_bonds.py 12 -> 19.** CONGRUENCE PIN (radius_A = 0.30
  -> resolved radius 0.30 BU at bond_thickness 1.0 AND 3.0 — independence is
  the cross-tool guarantee); precedence (both keys -> radius_A wins; only
  radius_scale -> base × scale; neither -> base — matches navigator 0.6.76);
  clamps (5.0 -> 1.0, 0.001 -> 0.02, at parse AND resolve); UNIT GUARD
  against the REAL mesh builder (scene_setup loaded with stubbed bpy, the
  test_job_roundtrip house pattern: ring verts of a 0.30 Å spec sit at
  0.30 Å from the axis — radius, not diameter); 0.7.9 REGRESSION GOLDENS
  (spec=None -> 0.06/0.18 at thickness 1.0/3.0; radius_scale-only 2.0/0.5 ->
  0.12/0.03, still scaling with thickness); normalized round-trip + bad-value
  warning; UI predicate. NEGATIVE CONTROLS (fail-then-pass, scratch copy):
  (a) letting bond_thickness scale the radius_A branch -> the congruence pin
  FAILS (thickness 3.0 resolves 0.90 ≠ 0.30, exit 1); (b) treating radius_A
  as a diameter (halving) -> the unit guard FAILS (ring distance 0.15 ≠ 0.30,
  exit 1); fix restored -> ALL 19 PASS (exit 0).
- **Contract doc refreshed**: `docs/bonds_sidecar.md` updated VERBATIM to the
  navigator 0.6.76 revision (md5 9e56e1c2 — adds the `radius_A` row, marks
  `radius_scale` LEGACY/derived, and the "Cross-tool display-geometry
  convention" section). Vendored parity re-verified against core 0.6.76
  (viewer-only release; 16 modules byte-identical, exit 0) — no wheel re-cut.

## 0.7.9

- **Navigator bond side-car import + job schema v2** — cross-tool bond
  consistency: curate bonds once in the web navigator, render identically in
  Blender, reproducibly through SLURM. Implements the BINDING interchange
  contract `docs/bonds_sidecar.md` ("phonon-vision-bonds" v1, shipped by the
  navigator since core 0.6.69; copied into this tree verbatim, md5 08490231):
  canonical pair keys (lexicographic sort, ASCII hyphen, en-dash tolerated on
  read, index keys forbidden), `max` = ABSOLUTE cutoff in Å (never multiplied
  by radii or tolerance), REPLACE semantics (pairs absent from the file form
  NO bonds; the global `bond_tolerance` criterion is fully bypassed — and its
  slider greyed out — while a spec is loaded), loose structure binding
  (formula/natoms mismatch warns, applies anyway), rejection leaves state
  untouched, unknown keys ignored, `color` read and STORED but not applied
  (0.7.10 does materials). An in-spec pair WITHOUT `max` falls back to the
  covalent default at the CONTRACT constant `DEFAULT_TOL = 1.2` (the
  navigator's fixed BOND_TOL), never the add-on's live tolerance.
- **NEW bpy-free `bonds.py`** (plain numpy; the covalent-radius function is
  injected): `canonicalize_pair`, `parse_sidecar` (normalizes + clamps: max
  [0.1, 10] Å, radius_scale [0.3, 3.0]; last-wins on canonical collisions),
  `structure_warnings`, `pair_scales`, and `build_bond_list` — the 0.7.8
  `scene_setup._build_bond_list` moved here VERBATIM (both the strict and the
  PBC-image branches) and extended with a per-species-pair cutoff matrix:
  spec=None reproduces the old criterion with the same float ops in the same
  order; a spec applies per-pair absolute cutoffs through ONE matrix used by
  BOTH branches, so image bonds inherit the parent pair's cutoff.
- **Per-pair geometry batching** (the deliberate 0.7.10 substrate): bonds are
  stored sorted by canonical pair; `create_bond_object` records per-bond pair
  keys + radius scales; the per-frame update runs ONE
  `_compute_bond_verts_vectorized` call per contiguous pair batch with
  cylinder radius = base × (spec `radius_scale` or 1.0) — uniform within a
  batch, so the vectorized kernel is unchanged and no per-bond loop appears.
  Without a spec every scale is 1.0 and a pre-0.7.9 object degrades to a
  single batch — the old single call exactly.
- **Job schema v2 (additive, presence-keyed)**: `SCHEMA_VERSION` 1 -> 2. New
  OPTIONAL top-level `bonds` key = the side-car EMBEDDED BY VALUE (never a
  file path — cluster jobs stay self-contained) + a `source` provenance block
  (filename + import date). Reader accepts schema_version 1 and 2; writer
  emits 2; `pv_render`'s schema warning accepts both. `bonds` is NOT
  partial-job keyed: absent -> the spec is CLEARED, so v1 jobs load unchanged
  and render byte-identically to 0.7.8 regardless of prior GUI state.
  `pv_inner` logs `curated bonds: N pair(s) (source: …)` so batch logs show
  which criterion a render used; the spec flows into the build via the new
  `bonds_spec_json` scene prop (docs/job_schema_v1.md gains the v2 section).
- **UI/operators**: `phonon_vision.import_bonds` (file browser; parse ->
  store spec + provenance -> `refresh_bonds`; reports pair count + every
  loose-binding warning; rejections report an ERROR and leave state
  untouched) and `phonon_vision.clear_bonds` (restores the default
  criterion). The Bonds panel shows "Curated: N pair(s) — file.json" while
  loaded.
- **Tests: tools/test_bonds.py NEW (12) + test_job_roundtrip.py 4 -> 7.**
  REGRESSION GOLDENS captured from the PRE-CHANGE 0.7.8 code path (verbatim
  extraction, vendored Cordero radii r_Sn 1.39 / r_Se 1.20) for BOTH branches
  — spec=None reproduces them exactly. REPLACE semantics (Cordero would bond
  Se-Se at 2.7 < 2.88; a spec omitting Se-Se yields zero Se-Se bonds; the
  passed tolerance is inert under a spec). CANONICALIZATION ("Sn-Se",
  "Se-Sn", en-dash files -> identical specs and bond lists). ABSOLUTE-Å PIN
  (max 2.99 vs 3.01 toggles exactly the d=3.00 bond — catches any unit/radius
  scaling). HALO INHERITANCE (the d=2.0 image bond obeys Sn-Sn max 2.5/1.9;
  the same matrix drops the d=2.7 INTERNAL Sn-Sn bond — one criterion, both
  paths). Parse rejections/warnings/clamps/stored-color; tolerant
  `spec_from_json`. Job: v2 bonds round-trip by value incl. source; a v1 job
  (and a v2 job without `bonds`) clears a stale spec.
- **Negative controls (fail-then-pass, bonds.py md5-pinned 94471de4 both
  restores):** (1) REPLACE -> MERGE (absent pairs fall through to Cordero) ->
  exit 1, the omitted pairs' bonds leak back in ([0,1],[0,4],[1,2],[2,3]
  reappear); restore -> 12/12. (2) canonical sort dropped -> exit 1 (spec
  keys stop resolving; the Å-pin and canonicalization tests fail); restore ->
  12/12.
- **Ground facts pinned during STEP 1** (recorded in CLAUDE.md): the bond
  criterion compares in Å end-to-end (1 Blender unit ≡ 1 Å; positions reach
  the criterion unscaled, so absolute side-car cutoffs apply verbatim), and
  in halo display mode the image-atom SLOT selection happens inside the
  FROZEN vendored phonon_core (`expand_cell(mode="halo")`, Cordero×τ) — a
  spec governs every bond DRAWN but cannot add image atoms beyond the
  Cordero halo (spec cutoffs longer than Cordero×τ may lack image partners;
  omitted pairs can leave bond-less halo spheres). Known limitation,
  documented.
- Guards: `check_stamps` exit 0; `check_vendored_parity` vs
  ../phonon_core-0.6.75-src [tree since renamed `../phonon-core`] exit 0
  (16 modules + pinned stub — no vendored
  module moved). Both measured WITHOUT pipes.

## 0.7.8

- **Discipline onboarding + release guards + wheel hygiene. NO behavior
  change** — no add-on/GUI/headless code path moved; every change is a guard
  tool, a doc, packaging metadata, or a bundled-dependency version.
- **CLAUDE.md created at the tree root** (this tree's first): component role
  (Blender extension consuming phonopy files directly via the vendored
  phonon_core parsers — it does NOT read JSON v1), sibling components and the
  private-tree relationship, an md5 ledger of every `.py` file +
  `blender_manifest.toml` at 0.7.8, the vendored-wheel provenance line, the
  discipline rules imported from the core project (one concern per version,
  three synced stamps, snapshot zip, fail-then-pass negative controls, scope
  proofs), test commands, known traps, and the standing "Blender 5.2
  readiness" checklist.
- **NEW `tools/check_vendored_parity.py`** (plain python, stdlib only) — the
  architect's audit found the vendored `phonon_core-0.1.4` wheel byte-identical
  to private 0.6.75 in all 15 non-`__init__` modules, guarded by nothing. Now
  guarded: the tool unzips the manifest's wheel (or `--wheel PATH`) and
  md5-compares every module against `src/phonon_core/<same path>` under
  `--core-src`. Exit 0 all-identical (prints module list + core version from
  the src pyproject); exit 1 with a per-file diff-count table on divergence;
  exit 2 when a wheel module has no src counterpart. The wheel's root
  `__init__.py` is the sanctioned packaging stub (own `__version__`, imports
  only the vendored subset — no io/conformance/unfolding/bz) and is checked
  against a PINNED md5 (`07478e09`) instead; re-pin deliberately when re-cut.
  Verified TODAY against `../phonon_core-0.6.75-src` [tree since renamed
  `../phonon-core`]: 16 modules MATCH +
  stub pinned, exit 0. Runs before EVERY add-on release.
  - Negative control (fail-then-pass): scratch copy of the core src with one
    line changed in `expand.py` → exit 1 naming `expand.py` (2 diff lines);
    scratch `expand.py` deleted → exit 2 NO-COUNTERPART; restored → exit 0.
- **NEW `tools/check_stamps.py`** (plain python, stdlib only) — the three
  version stamps (`bl_info["version"]` tuple, manifest `version`, newest
  CHANGELOG heading) had no sync check. The tool parses all three and exits 0
  iff they agree, else 1 naming the mismatch. Negative control: perturbing the
  manifest stamp in a scratch copy → exit 1; restored → exit 0.
- **h5py unified to ONE version, 3.14.0, across all five platforms**
  (mac x64 / mac arm64 / win: 3.16.0 → 3.14.0; linux x64 / arm64: already
  3.14.0, unchanged). 0.7.7 shipped the split because the old script let pip
  resolve "latest per platform" at download time. 3.14.0 is the newest release
  publishing cp311 wheels for ALL five platforms on the reachable PyPI index
  (verified via `pip index versions h5py` — 3.16.0 is not on the index — and
  the pypi.org JSON file list for 3.14.0); the three replacement wheels were
  fetched with `pip download --only-binary=:all:` and the manifest wheel list
  updated to match. PyYAML 6.0.2 was already unified — unchanged.
- **`download_wheels.sh` fully parameterized**: `PYYAML_VERSION` /
  `H5PY_VERSION` / `PY_TAG` knobs at the top (h5py now PINNED — the unpinned
  resolution was the split's root cause), a `--py-tag` override flag deriving
  `--python-version` from the tag, a loud non-zero exit when any platform
  wheel fails to download (the manifest lists all five — a silent skip ships
  a broken platform), and a header documenting the Blender 5.2 migration
  procedure (read 5.2's bundled Python; rerun with the new tag; update the
  manifest list; run the headless smoke under 5.2; bump `blender_version_min`
  ONLY after the smoke passes). `blender_version_min` stays 4.5.0.
- Baseline regression: `python tools/test_job_roundtrip.py` — 4/4 PASS before
  and after (no behavior change to guard).

## 0.7.7

- **Removed Blender's default startup objects from the headless render.** A
  cluster render showed the factory "Cube" as a white box at the origin (plus a
  stray default "Light"/"Camera") leaking into the background build — the GUI
  never shows these because the user starts from a clean scene.
- Fix (HEADLESS-ONLY, in `tools/pv_inner.py`): a new `_clear_factory_objects()`
  runs BEFORE `build_scene_from_data`, removing every pre-existing object whose
  name is not prefixed `PV.` (i.e. the factory Cube/Light/Camera). Because it
  runs before the build, no PV object exists yet — the supercell boundary box,
  arrows, bonds, PV camera (PV.CameraObj) and lights (PV.KeyLight, …) are created
  afterward and are never at risk. Each removal is guarded (no error if already
  clean); it logs what it cleared. The GUI render operator and the user's
  interactive scene are untouched.
- No add-on/GUI/phonon-core change beyond the headless orchestrator.

## 0.7.6

- **Sweep generator `tools/pv_sweep.py`** — one base job + variation axes → N
  job files + a SLURM array script. Pure system/conda Python (no bpy, no
  phonon_core); runs on a login/dev node, no compute.
- Sweeps `display.q_index` and/or `display.band_index` from a fixed base job.
  Axes take an explicit list (`400,403,410`) or a range (`start:stop[:step]`,
  Python-range / end-EXCLUSIVE, so `0:9` = 0..8). Pairing is `--mode product`
  (cartesian, default) or `--mode zip` (paired; errors on unequal lengths).
  `--format mp4|gif|png` sets output.format + animation; `--device`/`--samples`
  optionally override every job.
- Writes `<out-dir>/jobs/job_NNNN.<ext>` (each = base except q/band/output),
  `<out-dir>/sweep_manifest.tsv` (task_id, job_path, q, band, output_name; no
  header), and `<out-dir>/render_sweep.sb` — a `#SBATCH --array=0-<N-1>%<K>`
  script that mirrors the hero env block (module purge; Blender; Miniforge3;
  conda activate <env> for ffmpeg), sanity-checks the extension (exit 2),
  looks up the task row via awk, renders into per-task scratch with
  `pv_render.py … -o`, and copies back to `~/phonon-vision/renders/`.
- New `docs/sweep.md` with three worked examples (all bands at one q; one band
  along a q-range; a product matrix).
- No add-on/GUI/phonon-core change — generator tool + doc only.

## 0.7.5

- **Fixed integer-vector job round-trip** — a job's `display.supercell` was
  silently ignored, so a 9x9x1 job rendered the default 3x3x3. `supercell_n` is
  the only IntVectorProperty in job.py's `_MAP`. Two bugs combined: `scene_to_job`
  serialized every vector as `[float(x) …]` (exporting `[9.0, 9.0, 1.0]`), and
  `_apply_job_body` set vectors with no coercion — Blender's IntVectorProperty
  rejects a float tuple (TypeError), the `except` caught it, and `supercell_n`
  kept its (3,3,3) default. The old round-trip test passed only because it
  round-tripped that default.
- Fix is type-aware via `props.bl_rna.properties[prop].type` (NOT a blanket cast,
  which would break the FLOAT color/offset vectors arrow_color /
  overlay_text_color / overlay_background_color / camera_target_offset):
  - `_apply_job_body` "vec": INT -> `tuple(int(round(v)) …)`, else
    `tuple(float(v) …)`. The `round()` also makes EXISTING float-valued job files
    (like the user's `[9.0, 9.0, 1.0]`) apply correctly.
  - `scene_to_job` "vec": INT -> `[int(x) …]`, else `[float(x) …]`, so future
    exports write `supercell` as `[9, 9, 1]`.
- All `[pv.job]` "could not set …" warnings now go to stderr, so a silent
  fallback like this is visible in batch logs.
- New `tools/test_job_roundtrip.py`: a no-bpy regression test that sets a
  NON-DEFAULT integer vector and asserts it survives as ints through
  export+apply (and that float vectors stay float).
- GUI untouched; no phonon-core change.

## 0.7.4

- **Working animation output (mp4 + gif) in the headless/CLI path**, frames-first.
  HEADLESS-ONLY; GUI render operator and GUI playback untouched. No phonon-core
  change. Fixes the 0.7.3 failure where an MP4 job produced a 48-byte unplayable
  `name.png0001-0250.mp4` (Blender's default 250-frame range, mangled filename,
  no frames kept, render lost).
- Inner (render_headless / pv_inner): for `animation: true` the inner now renders
  a **PNG frame sequence** — never an in-Blender container — over exactly ONE
  seamless loop. The frame range is the SAME the GUI uses
  (`recommended_render_settings(frames_per_cycle, fps)`; seam frame not
  duplicated), and `scene.render.fps` is set from `loop.fps`. Frames go to a
  clean `<output_stem>_frames/` dir and are kept on disk. Prints
  `[pv] rendered N frames -> <dir>`. The still path (`animation: false`) is
  unchanged.
- Outer (pv_render): after Blender returns, assembles the frame sequence with
  **ffmpeg** into the exact `-o` path (no frame-range suffix, no double
  extension): MP4 = H.264/yuv420p/`-crf 18`; GIF = two-pass palettegen+paletteuse
  at `loop.fps`. Frames are kept by default (safety net); `--clean-frames`
  deletes them after a successful encode. If ffmpeg is missing, the frames are
  preserved and the launcher exits nonzero with a clear "frames rendered to
  <dir>; ffmpeg not found … then re-run" message (no silent empty container).
- `-o` extension and `output.format` are reconciled (a `.mp4`/`.gif` `-o` implies
  that container; otherwise derive from `output.format`). `loop.frames_per_cycle`
  and `loop.fps` are now honored for output. Schema + cli docs updated;
  `output.format` documents the mp4/gif/png behavior.

## 0.7.3

- **GPU (CUDA/OptiX) rendering in the headless/CLI path.** HEADLESS-ONLY: the GUI
  render operator and the user's GUI Cycles preferences are untouched — the
  interactive render keeps using whatever device the user set.
- `api.set_render_device(scene, device="CPU", backend="AUTO")` (in operators.py):
  CPU sets `scene.cycles.device='CPU'`; GPU/AUTO sets the Cycles addon prefs'
  `compute_device_type` (AUTO tries OPTIX then CUDA), enables the matching GPU
  devices, and sets `scene.cycles.device='GPU'`. Explicit GPU with no device
  warns and falls back to CPU; AUTO falls back silently; missing `cycles` prefs
  is guarded. NEVER calls `save_userpref()`, so changes stay ephemeral to the
  headless process. Returns a one-line summary for the log.
- `tools/pv_inner.py`: after `apply_preset`, reads `render.device` /
  `render.gpu_backend` from the job, calls `set_render_device`, and prints
  `[pv] render device: <summary>`.
- Job schema: optional `render.device` (CPU|GPU|AUTO, default CPU) and
  `render.gpu_backend` (AUTO|OPTIX|CUDA, default AUTO) — documented in
  docs/job_schema_v1.md. Default CPU = unchanged, so existing jobs render
  identically. `pv_render.py` gains `--device` and `--gpu-backend` overrides;
  examples/job_snse2.yaml gains a commented `device:` line; docs/cli.md shows
  `--device GPU`.

## 0.7.2

- **Phase 2: the `pv_render` outer launcher** — a plain-Python CLI wrapper. No
  add-on behavior change, no phonon-core change.
- `tools/pv_render.py` (system/conda Python, stdlib + PyYAML, no bpy): reads a
  friendly YAML or JSON job, validates it FAST before launching (schema_version,
  input.file exists, output.path present/-o), applies `-o`/`--samples`
  overrides, writes a temp JSON (Blender's Python has no PyYAML), resolves
  Blender (`--blender` > `$PV_BLENDER` > PATH), resolves the sibling
  `pv_inner.py`, and runs `blender --background --python pv_inner.py -- --job
  <temp.json>` (no `--factory-startup`). `--dry-run` prints the command + temp
  job without launching; `--keep-temp` retains the temp JSON; the launcher exits
  with Blender's return code so failures propagate to HPC job arrays.
- `examples/job_snse2.yaml`: a commented YAML twin of the JSON sample, with a
  clean top-down camera (`mode: MILLER, miller: [0,0,1], auto_frame: true`) for
  the vacuum-padded 2D cell.
- `docs/cli.md`: local + ICER usage; notes that PyYAML is required in the
  launcher env and the extension must be installed in the target Blender.

## 0.7.1

- **Fixed property-update-callback NameError spam** (hundreds of
  `File "<string>", line 1, in <lambda>` / `NameError: name '_on_*_change' is
  not defined` during headless runs as load_file/apply_job_to_scene set props).
  Root cause: state.py used `from __future__ import annotations` (PEP 563), so
  the PropertyGroup `prop: Type(update=lambda self, ctx: _on_x(ctx))` annotations
  became strings that were eval'd at registration in a namespace lacking the
  module-level `_on_*` functions — the lambdas resolved `_on_x` against the wrong
  globals and raised every time a prop was set (values still applied; only the
  callbacks threw).
- Fix: removed `from __future__ import annotations` from state.py so the `update=`
  lambdas are REAL module closures whose `__globals__` is state.py's namespace;
  they now resolve the `_on_*` functions at call time. (All annotations in the
  module are builtins/typing — eager evaluation is safe.) Scoped to state.py;
  other modules' Operator prop annotations have no `_on_*` callbacks.
- Generalized `_SuppressSpeciesUpdates` into `_SuppressUpdates`: a module-level
  `_suppress_all_updates` flag that every `_on_*` callback early-returns on
  (nesting-safe, restored in `__exit__` even on exception). `load_file` (the
  post-load index reset + species population) and `apply_job_to_scene` (the whole
  bulk write) now run inside it — no N redundant refreshes per job, and headless
  stays quiet.
- GUI live-update is unchanged: the frame-handler path (sync_appearance) and the
  palette/camera operators are untouched; the now-resolving callbacks restore the
  intended per-edit refresh.
- No phonon-core change, no rebundle.

## 0.7.0

- **Phase 1 of the HPC CLI: job engine + headless orchestrator.** GUI unchanged
  except for one added "Export Job File" button. No phonon-core change.
- `docs/job_schema_v1.md`: the full job schema (schema_version 1), grouped
  input / display / appearance / camera / render / overlay / loop / output.
  Diffed against state.py and proven exhaustive — all 54 PhononVisionProps props
  plus the species collection and Miller indices are mapped.
- `phonon_vision/job.py`: one bidirectional mapping table drives both
  `apply_job_to_scene(scene, job)` (pure prop-writing; call AFTER load_file to
  override the populated species defaults/indices) and `scene_to_job(scene)`
  (the inverse, powering Export). They cannot drift. Species and Miller indices
  are special-cased; output.path/animation are synthesized job-only fields.
- `tools/pv_inner.py`: the headless orchestrator. Run with
  `blender --background --python tools/pv_inner.py -- --job <job.json>` —
  **no --factory-startup, no second add-on copy**. It DISCOVERS the already
  installed/enabled extension at runtime (scans sys.modules for the
  `*.phonon_vision` package exposing `api`; no hardcoded `bl_ext.user_default…`),
  then runs load → apply_job → build → sync_appearance → apply_camera →
  apply_preset → set film/resolution → render_headless, with `[pv]` progress
  lines and a nonzero exit on failure.
- GUI: `PV_OT_export_job` + an "Export Job File" button in the Render panel —
  dial in the look, export JSON, batch it headless.
- `examples/job_snse2.json`: a complete, realistic SnSe2 job (supercell 3×3×1,
  spherical camera, Cycles PBR, species colors) to validate the path.

## 0.6.9

- **Phase 0 of the HPC CLI: headless-safe core refactor.** No CLI yet; this only
  frees the two pieces that were trapped in the UI layer so the same functions
  serve both the GUI and the future `blender --background` path. GUI behavior is
  byte-identical; no phonon-core change, no rebundle.
- Extracted `load_file(scene, path) -> data` (operators.py): validates, sniffs
  format, dispatches to the phonon_core loader, stores data, resets indices, and
  populates species overrides — with the same human-readable messages. It does
  NOT build the scene or touch viewports. `PV_OT_load_yaml.execute` is now a thin
  wrapper that keeps its path/extension guards + HDF5 "Tip" (byte-identical GUI
  messages), calls `load_file`, then does the scene build + orthographic toggle +
  INFO report exactly as before.
- Added `render_headless(scene, *, animation, output_path=None, file_format=None)`
  (operators.py): configures PNG/FFMPEG output and renders SYNCHRONOUSLY in EXEC
  context (`write_still=True` / `animation=True`) — no render window, no
  render_complete handler. The GUI `PV_OT_render_animation` operator is unchanged.
  No overlay burn-in (Phase 1) and no GPU device selection (Phase 3) here.
- Added `phonon_vision/api.py`: a pure re-export of the headless contract
  (`load_file`, `build_scene_from_data`, `apply_camera`, `sync_appearance`,
  `render_headless`, `apply_preset`). No logic.
- Added `tools/pv_headless_smoke.py`: the acceptance test (and skeleton of
  Phase 1's inner script). Run:
  `blender --background --python tools/pv_headless_smoke.py -- <band.yaml> <out.png>`
  — registers the add-on self-contained, then load -> build -> camera -> render a
  PNG with no GUI window.

## 0.6.8

- **Movable camera target** (offset the look-at / orbit center). New
  `camera_target_offset` (FloatVectorProperty, TRANSLATION/LENGTH, default
  (0,0,0)): `apply_camera` uses `effective_center = bbox_center + offset`
  everywhere placement was driven by the center.
  - Spherical/Miller: `location = effective_center + distance·direction`;
    the pole-safe look-at aims at `effective_center` — so the camera orbits AND
    aims at the moved point.
  - Presets: `location = effective_center + preset_offset`, preset orientation
    unchanged; in the orthographic camera a lateral offset shifts what's centered
    in frame (the along-view component is a no-op), which is the intended
    "move the framing" effect.
  - Offset (0,0,0) reproduces prior framing exactly (regression-checked).
- UI: three "Target offset X/Y/Z (Å)" boxes + a "Center on structure" button
  (`phonon_vision.center_camera_target`) that resets the offset to (0,0,0). The
  offset uses the same live refresh path as az/polar/radius, so dragging it
  re-aims immediately. (Also fixed a stale "Elevation" UI header -> "Polar".)
- No phonon-core change, no rebundle.

## 0.6.7

- **Fixed Spherical/Miller camera aiming at the poles.** 0.6.6 aimed via a
  TRACK_TO constraint with a fixed `up_axis='UP_Z'`, which is degenerate when
  the view direction is parallel to world Z (straight up/down) — the camera
  stopped pointing at the structure. Replaced with a directly-computed,
  pole-safe look-at: forward = (center − location); up = +Z, falling back to +Y
  when |forward·Z| > 0.999; `rotation_euler` from the (right, true_up, −forward)
  basis. The TRACK_TO constraint and the `PV.CameraTarget` empty are dropped
  entirely (both camera paths now use `rotation_euler`); any straggler target
  from a 0.6.6 scene is cleaned up on the next camera setup.
- **Spherical now uses polar angle from +Z** (matching the intended model), not
  elevation from the horizontal. `camera_elevation` -> `camera_polar`
  ("Polar (from +Z)", 0–180, default 90). Direction:
  `(sin(polar)·sin(az), −sin(polar)·cos(az), cos(polar))` — polar 0 = top-down
  (+Z), polar 90 / az 0 = −Y (matches Side), polar 180 = bottom-up (−Z); azimuth
  sweeps around +Z. The three Spherical boxes are now Az / Polar / Radius.
- Miller mode uses the same new look-at (benefits from the pole-safe aiming).
- Presets (Side/Top/Iso) keep their existing `rotation_euler` paths unchanged.
- No phonon-core change, no rebundle.

## 0.6.6

- **Two new camera modes: Spherical and Miller index**, alongside Side/Top/Iso.
  Add-on only. The Side/Top/Iso preset framing is byte-for-byte unchanged; the
  placement logic was refactored into a shared `apply_camera(scene, view)` that
  the operator and the live property callbacks both call.
- Spherical: `camera_azimuth` (0–360°), `camera_elevation` (−90–90°),
  `camera_radius` (Å). Direction (center→camera):
  `dir = (cos(el)·sin(az), −cos(el)·cos(az), sin(el))`.
- Miller: `camera_miller_h/k/l` (floats, for fine tuning — per ascii-phonons).
  Direction = the (hkl) plane normal `h·b1 + k·b2 + l·b3`, where b1,b2,b3 are the
  reciprocal-lattice vectors of the unit cell (Ashcroft & Mermin, Solid State
  Physics Ch.5, 1976; ported from ascii-phonons `camera.py::reciprocal`). The 2π
  and cell-volume scalars wash out under normalization — direction only.
- Spherical/Miller aim via a TRACK_TO constraint on a `PV.CameraTarget` empty at
  the structure center (robust for all directions incl. top-down). Switching back
  to a preset removes the constraint and the empty so the preset's direct
  `rotation_euler` is unaffected.
- Conventions (for reproducible figures):
  - Target = structure bbox center; up = world +Z (TRACK_TO falls back near
    vertical automatically).
  - Spherical: az0/el0 = view from −Y (Side-like); el+90 = top-down (+Z).
  - Miller (hkl) = looking along the (hkl) plane normal (reciprocal-lattice
    direction). Negative indices view from the opposite side. On non-orthogonal
    cells this is the plane normal, NOT the real-space [hkl] direction.
  - ORTHOGRAPHIC throughout: distance/radius only POSITIONS the camera; zoom is
    the ortho frame width (Auto-frame or manual Frame width), never the radius.
- No phonon-core change, no rebundle.

## 0.6.5

- **Drive appearance writes from the per-frame handler** (final attempt at
  live-during-playback color). 0.6.4 wrote materials in the property callback,
  but that still didn't surface mid-playback — so the write now also runs from
  `frame_change_pre`, the one path guaranteed to execute every frame during
  playback.
- `sync_appearance(scene)` (material writes ONLY: PV.ArrowMat + each PV.Mat.<sp>
  Base Color + Emission + diffuse_color + update_tag; no object replacement, no
  node rebuild) is called from `frame_change_handler` right after
  `update_atom_positions`. The depsgraph eval for that frame then shows the
  current colors. All static-case machinery (synchronous callback writes,
  deferred redraw kicks, deferred geometry) is retained.
- Note: this re-writes + update_tags materials every frame during playback
  (cheap value assignments). If a future build shows playback stutter, gate the
  per-frame write behind a cheap "colors changed since last frame" check.
- No phonon-core change, no rebundle.

## 0.6.4

- **Fixed: appearance changes (arrow color, vibrancy, per-species color) now
  apply WHILE A MODE ANIMATES**, not only when static. Root cause: 0.6.3 routed
  the material *write* itself through a `bpy.app.timers` callback, and timers are
  starved during continuous playback — so the write never ran until playback
  stopped.
- Fix: write the material SYNCHRONOUSLY in the callback (callback-safe), defer
  only the redraw kick.
  - `write_arrow_color_sync` / `apply_species_colors_sync` write the material
    (Base Color + diffuse_color + emission) and `mat.update_tag()` immediately,
    inside the property callback. `_apply_species_color` is a new in-place,
    color-only writer (no node rebuild, no object replacement).
  - The deferred timer now carries only `redraw_kick()` (`view_layer.update` +
    `tag_redraw`), debounced per key, for the static case. During playback the
    per-frame handler's depsgraph eval picks up the already-dirtied material.
  - Playback fallback: the `frame_change_pre` handler drains pending redraw
    kicks once per frame (timers are starved during playback). This path carries
    ONLY redraw kicks — never object replacement.
  - Object replacement (radius -> sphere-mesh rebuild) stays on the deferred,
    timer-only path (`refresh_species_visuals`); never synchronous, never
    per-frame.
- Untouched: the 3 preset palette operators (call `refresh_species_visuals`
  directly from `execute()`), physics, the per-frame position handler, and all
  datablock-removal logic.

## 0.6.3

- **Live appearance updates now actually refresh the viewport.** The 0.6.2
  `tag_redraw` was insufficient: it repaints the *cached* material without
  re-evaluating the depsgraph, so the new color stayed invisible until a
  rebuild. Root cause: property-update *callbacks* (arrow_color,
  species_vibrancy, per-species "free palette" color) only tagged a redraw,
  while the palette *operators* work because they run a depsgraph re-eval
  (`view_layer.update()`) in clean `execute()` context.
- Fix, applied to every non-live appearance-prop callback:
  1. `_invalidate_and_redraw(mats)` helper — `mat.update_tag()` each edited
     material (covers Solid `diffuse_color` AND Material/Rendered Base Color),
     then `bpy.context.view_layer.update()`, then the `VIEW_3D` `tag_redraw()`
     loop. `refresh_arrow_color` and `refresh_species_visuals` both end with it.
  2. Callbacks run the refresh **deferred** via `bpy.app.timers.register(...,
     first_interval=0.0)` so the depsgraph re-eval lands in a clean context, not
     mid-callback (Blender ignores a depsgraph update issued from inside a
     property-update callback). Debounced with a module-level pending-key guard
     so a color-swatch drag schedules at most one timer per key.
  3. `traceback.print_exc()` in these callbacks (now via the deferred runner),
     so a swallowed throw can't masquerade as "not updating".
- Untouched (still live, by design): the 3 preset palette operators
  (VESTA/Jmol/Auto), physics, the per-frame handler, and all datablock-removal
  logic. In-place material writes only.

## 0.6.2

- **Fixed arrow color not updating.** Two bugs, one root cause — arrow color was
  only fully written when the material was first *created*:
  - `create_arrow_object` set the Principled BSDF "Base Color" only inside the
    `if mat is None:` block, so on a rebuild the node color stayed stuck at the
    first color while only `diffuse_color` updated → Material Preview / Rendered
    never recolored, even after Rebuild Scene.
  - The live callback set both colors but never tagged the viewport for redraw,
    so a Solid-mode change wasn't shown until the next redraw (a Rebuild).
- Fix: a single shared helper `_apply_arrow_color(mat, rgba)` writes BOTH the
  BSDF "Base Color" and `mat.diffuse_color`; `create_arrow_object` now calls it
  on **every** build (not just creation); `refresh_arrow_color` calls it and then
  forces a 3D-viewport redraw via `area.tag_redraw()`. The update callback now
  prints a full traceback on error so a swallowed throw can't masquerade as
  "not live".

### Pattern notes for future appearance properties (consistency)

These two rules came out of the arrow-color bug and apply to any appearance
property (vibrancy and palette are candidates that share the same shape; not
changed here, out of scope):

1. **Apply color on every build, not only at material creation.** Setting a node
   input inside an `if mat is None:` creation block leaves it stuck at the first
   value on every later rebuild. Write appearance state unconditionally after
   get-or-create.
2. **`tag_redraw` is required for live viewport-display changes.** Editing
   `diffuse_color` (or other display-only values) in place does not repaint the
   viewport on its own; tag the `VIEW_3D` areas for redraw so changes show
   without a Rebuild Scene.

## 0.6.1

- Added user-selectable global arrow color (`arrow_color`, default = the old red
  0.90/0.10/0.10), with a Display-panel swatch and a live update callback.

## 0.6.0

- Unified supercell expansion moved into phonon-core (`expand_cell`); the PBC
  toggle dispatches halo vs strict. Bundles phonon_core 0.1.4.
