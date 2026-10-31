# Quantum ESPRESSO acceptance fixtures

The in-suite QE tests validate the `.fc` reader against ANALYTIC spring
models built from the format spec. They deliberately CANNOT settle the
conventions only real q2r output can (see `qe_import/fc_file.py`'s
docstring). To close those pins, drop real data here:

```
tests/fixtures/qe/real/<case-name>/
    <anything>.fc            # the q2r.x output, non-polar for now
    reference_freqs.csv      # matdyn.x frequencies at chosen q
```

`reference_freqs.csv`: one row per q-point, comma-separated —
`q1, q2, q3, nu_1, ..., nu_3N` — with q in PRIMITIVE FRACTIONAL coordinates
(crystal units, the same q you gave matdyn with `q_in_cryst_coord = .true.`)
and frequencies in THz, ascending. Convert matdyn's cm⁻¹ with
nu_THz = nu_cm1 / 33.35641 (the project's frozen constant).

Protocol: run matdyn.x on the SAME .fc at a handful of q including Γ and a
zone-boundary point, with NO interpolation tricks (`asr = 'no'` for the
first pass, so the comparison is against the raw FC the file carries).
`tests/test_qe_import.py::test_real_q2r_acceptance` picks the case up
automatically and compares through the full chain at the metric doctrine.

Polar materials: parsed but conversion is refused until the NAC unit
constant is pinned — a polar acceptance case needs the dielectric/Born
handling landed first (roadmap).

---

## What real data settled (measured 2026-07-30)

Real cases are **not committed** here. Two CsSnI3 ph.x/q2r cases (cubic and
epitaxially relaxed), both **polar**, were used to validate the reader; drop
equivalent data into `real/<case-name>/` (protocol above) to re-activate the
acceptance tests.

Staged with `tools/stage_qe_fixture.py <source-dir> <case-name> --every 20`.

### What the reader half already proved

Real q2r output broke four assumptions the analytic fixtures could not
reach, each now fixed and covered:

1. **Quoted species labels contain spaces** (`'Cs '`, and an all-blank
   `'   '`), which a whitespace token split shreds. The header is now parsed
   line-oriented.
2. **Labels disagree with masses.** The CsSnI3 file carries type 2 with a
   BLANK label at 118.710 amu (Sn) and type 3 labelled `'Sn '` at 126.904
   amu (iodine). Masses are load-bearing — they enter D(q) — so species are
   resolved BY MASS and the substitution is reported in `symbol_sources`.
3. **The polar block has a per-atom INDEX line** before each 3x3 Born
   tensor, and the `T` flag line carries a trailing number. Charge
   neutrality (sum of Born charges = 0 to machine precision on all three
   files) is the independent check that this is now read correctly.
4. **ibrav=6** (simple tetragonal) was not implemented.

### The remaining gap, measured — and correctly attributed

Converting `CsSnI3_epirelax` WITHOUT NAC and comparing against its matdyn
reference (which used NAC **and** `asr = 'crystal'`):

| | THz |
|---|---|
| max abs difference, overall | 0.786 |
| max at Gamma | 0.698 |
| median | 0.216 |

**Correction to an earlier reading of this table.** It first looked as though
the Gamma discrepancy was dominated by the acoustic sum rule. Measurement
says otherwise: applying an ASR projection moves the gap from 0.6975 to
0.6974 THz — nothing. The projection itself works (row-sum drift 3.6e-02 ->
1.5e-14, and three exact Gamma zeros appear), and enabling phonopy's NAC with
the file's own Born/dielectric data moves frequencies by only ~0.03 THz,
because eps_infinity ~ 19.7 screens the LO-TO splitting heavily.

The real cause is structural: **for a polar material `q2r.x` SUBTRACTS the
long-range dipole-dipole term before Fourier transforming**, so the `.fc`
holds only the short-range remainder and `matdyn.x` adds the long part back
with its own Ewald parameters. Reconstructing that in phonopy's convention is
the actual pending work — not a sum rule, and not a single unit constant.

**This is why the `.dyn` route exists** (`qe_import.dyn_file`). The ph.x
dynamical matrices have no such subtraction: they are exact at the computed
q, valid for polar materials today, and each file self-validates. Where the
`.fc` route needs the reconstruction pinned, the `.dyn` route already works.

(matdyn's own lowest Gamma mode is -0.066 THz — a genuine soft mode. CsSnI3
is an unstable perovskite; the negative value is physics, not an artefact.)
