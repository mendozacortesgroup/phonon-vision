# Conformance & golden-vector harness

`phonon_core` is the hub: a Blender add-on and (later) a JavaScript web viewer
both depend on it producing the *same* phonon kinematics. Until now that physics
was validated only by looking at Blender renders. This harness makes the
conventions **executable**, so any future importer or the JS engine can be
checked against a frozen reference instead of by eye.

The harness is **additive**. It conforms *to* the existing engine —
`kinematics.py`, the parsers, and `io/json_v1` are the reference spec and are not
modified by any of this. Where a test surfaced a real property of the engine
(see "The canonical-phase tie" below), it is reported here, not patched away.

## The dangerous bug class

The error this harness exists to catch is a **per-atom phase error**: one atom's
complex eigenvector multiplied by a spurious phase. It is insidious because

- every *magnitude*-based metric is blind to it — species participation weights,
  the inverse participation ratio, the projected DOS all depend only on `|e|`,
  which a pure phase rotation leaves unchanged; and
- at the Γ point it is invisible, because every cell shares the Bloch factor
  `exp(i q·R) = 1`, so a per-atom phase only shows up as a wrong *relative* phase
  between atoms — i.e. in the complex eigenvector, with an effect on the Bloch
  animation that grows with `|q|`.

So a check that only compares magnitudes, or only looks at Γ, would wave this bug
straight through. The harness's load-bearing assertion is therefore a
**complex** eigenvector match at **finite q**.

## The golden vectors

`tests/golden_kinematics.npz` + `tests/golden_kinematics.json` freeze the output
of the current engine for a fixed, documented set of modes from the trusted
WSe2 `band.hdf5` (path Γ–M–K–Γ, atoms W/Se/Se):

| name | q-index | band | q (frac) | character |
|------|--------:|-----:|----------|-----------|
| `gamma_acoustic` | 0 | 2 | (0,0,0) | Γ acoustic (≈0 THz) |
| `gamma_optical` | 0 | 8 | (0,0,0) | Γ optical |
| `m_point` | 201 | 3 | (½,0,0) | M, finite q |
| `k_point` | 403 | 3 | (⅓,⅓,0) | K, finite q |
| `generic` | 100 | 4 | (0.249,0,0) | generic low-symmetry, finite q |

The three finite-q modes are non-negotiable; they are the only ones that expose a
phase-convention error. For each mode the `.npz` stores the raw and
canonical-phased eigenvectors, the per-slot atom-index and Bloch-phase arrays,
and the displacement at a fixed phase `phi` plus a full `[0,2π)` trajectory; the
`.json` sidecar carries the conventions block, the exact input spec (`amplitude`,
`phi`, supercell, `n_frames`, mass-weighting), the source file's SHA-256, and the
`phonon_core` version the fixture was frozen under.

### Regenerating

The fixture is regenerated only deliberately — never silently in CI:

```
HDF5_USE_FILE_LOCKING=FALSE python tools/gen_golden.py \
    --band /path/to/2d_WSe2_band.hdf5
```

It reads `phonon_core` and writes the two files; it does not modify the engine.
The regression test (`tests/test_kinematics_golden.py`) then reloads the fixture,
re-runs the kinematics on the identical inputs, and asserts reproduction within
`1e-10`, guarding the Python engine against silent drift.

## The three tiers

`conformance.check_against_reference` compares a candidate eigenvector set to a
reference in three tiers and reports *which* tier failed, so the caller learns
what kind of disagreement they have.

**Tier 1 — frequencies.** The scalar dispersion. A mismatch means the dynamical
matrix or its diagonalization differs and the eigenvectors are not comparable.
Skipped if no frequencies are supplied.

**Tier 2 — per-atom magnitude `|e|`.** The mode's shape in magnitude space:
which atoms move how much. Necessary but **not sufficient** — it is exactly the
quantity the dangerous phase bug leaves unchanged.

**Tier 3 — per-atom complex `e`, after canonical-phase alignment.** The full
polarization, compared after removing the one physically arbitrary degree of
freedom: the global phase that diagonalization leaves on an eigenvector. A
global phase is meaningless and is absorbed; a per-atom phase is a different mode
and is not. **Tier 3 evaluated at a finite-q mode is the assertion that actually
guards the physics.**

Why alignment is needed at all: diagonalizers return eigenvectors up to an
arbitrary `e^{iα}`. The canonical rule (`kinematics.canonicalize_eigenvector_phase`)
fixes it by rotating so the largest-magnitude component is real and positive. The
checker applies that rule — see the tie note below for the one robustness detail.

## The canonical-phase tie (reported finding)

While building tier 3, the harness surfaced a genuine property of the engine.
The `generic` WSe2 mode (q-index 100, band 4) is a real standing-wave mode whose
two largest Cartesian components are **exactly co-equal in magnitude and exactly
anti-phase** (`+0.218−0.582j` and `−0.218+0.582j`, magnitudes equal to the last
bit). The canonical rule selects its pivot with `argmax`, and under an exact tie
`argmax` is decided by floating-point noise.

Consequences, and what is and isn't a bug:

- `canonicalize_eigenvector_phase` is **deterministic for a fixed input** — same
  vector in, same vector out. That is correct and is asserted in the tests.
- It is **not** invariant across a global-phase rotation under such a tie: the
  rephasing multiply perturbs the tied magnitudes at ~1e-16, so `argmax` can flip
  to the other (anti-phase) tied component, and the two "canonical" forms then
  differ by ~unity. A naive tier 3 that re-derived `argmax` independently for
  candidate and reference would *falsely* fail on a pure global-phase difference.

This is the "canonical phase isn't deterministic" case, reported rather than
patched. The engine is left untouched (its per-vector behavior is correct and
adequate for fixing a single render's φ=0 snapshot). The **checker** is made
robust where it is free to be: tier 3 derives the pivot **once from the
reference** (`argmax|ref|`) and rephases both candidate and reference by that same
index. This is still the canonical rule — the reference defines "the largest-
magnitude component" — but it is invariant to a global phase for every mode, ties
included. The convention assertion (Deliverable 4) likewise checks the engine's
*chosen* pivot (`argmax|raw|`), not a re-derived `argmax|canon|`.

## Tolerances

Two very different tolerance regimes appear in this harness.

**Parser equivalence (yaml vs hdf5).** phonopy emits the same calculation as both
text (`band.yaml`) and binary (`band.hdf5`); they must carry the same physics.
Measured maxima and the documented ceilings:

| quantity | measured (band / mesh) | ceiling | note |
|----------|------------------------|---------|------|
| q-points | 4.97e-8 / 4.95e-8 | 5e-8 | path round-off |
| frequencies | 5.0e-11 / 5.0e-11 | 5e-11 | THz |
| eigenvectors | 7.05e-15 / 7.07e-15 | **1e-12** | text-vs-binary float floor |

The eigenvector ceiling is a principled float-floor gate, not a measured edge.
The prior hand-check recorded "~7e-15"; re-measured precisely, the maximum is
7.05e-15 (band) / 7.07e-15 (mesh) — text-vs-binary representation noise (mean
~3e-15, **every** component below 1e-13). A literal ~7e-15 ceiling is *below*
what real data hits and would fail the day it is enforced. The gate exists to
catch convention / sign / phase bugs — which are **order unity** at finite q —
not float noise, so it is set at 1e-12: ~9 orders below a real bug and
comfortably above the ~1e-14 floor. q-points (5e-8) and frequencies (5e-11) are
left at their measured values.

These pairs are committed under tests/fixtures/phonopy/ and run by default; the
mesh pair is the first 64 q-points of the real SnSe2 grid, sliced independently
from phonopy's yaml and hdf5 so it still compares the two emitters (the full
41 MB mesh.yaml is too large to vendor). `$PHONON_CORE_FIXTURES` optionally
points the same filenames at full-size local files.

**Physics conformance (`check_against_reference`).** The defaults are
`atol=1e-8, rtol=1e-5` — deliberately loose relative to the parser floor, because
a legitimately different *source* of the same mode is allowed to disagree at the
level of input-table differences while a real bug is enormous by comparison:

- phonopy's and pymatgen's default atomic-mass tables differ by ~1e-3 relative,
  which feeds the `1/√m` mass-weighting; an importer that picked a different table
  should still be judged conformant.
- a dropped or sign-flipped intra-cell phase is **order unity** at finite q.

So there is a wide, safe margin between "absorb a mass-table difference" (~1e-3)
and "catch a phase bug" (~1). The crown-jewel self-test lands a phase corruption
at ~1.9e-1 — far above the tolerance, far below nothing — exactly where a real
bug lives.

## Using the checker (future importer / JS engine)

A future importer test, or the JS validation harness, loads the reference modes
from the golden fixture and calls the checker on its own output:

```python
import numpy as np, json
from phonon_core import conformance

npz  = np.load("tests/golden_kinematics.npz")
side = json.load(open("tests/golden_kinematics.json"))
names = list(side["modes"])

reference = np.array([npz[f"{n}__eigvec_raw"] for n in names])  # (M,N,3) complex
qpoints   = np.array([side["modes"][n]["q_point"] for n in names])
masses    = npz["_material_masses"]

candidate = my_importer.load_eigenvectors(...)   # same modes, same order

res = conformance.check_against_reference(candidate, reference, masses, qpoints)
if not res.passed:
    tier = res.first_failed_tier
    raise AssertionError(f"importer is non-conformant at {tier.name}: {tier}")
# crucial: confirm a finite-q mode was actually exercised
assert res.has_finite_q
```

`res` is a `ConformanceResult`: `.passed`, `.first_failed_tier`, the three
`TierResult`s (each with `.passed` and `.max_abs_dev`), and `.finite_q_mask`.
Printing it gives a per-tier summary:

```
ConformanceResult(PASS; 5 modes, 3 atoms, finite_q=yes)
  [PASS] frequencies: max|Δ|=0.000e+00
  [PASS] per-atom |e|: max|Δ|=0.000e+00
  [PASS] per-atom complex e (canonical-aligned): max|Δ|=0.000e+00 LOAD-BEARING at finite q
```

Pass frequencies too (`candidate_freqs=`, `reference_freqs=`) to enable tier 1.
