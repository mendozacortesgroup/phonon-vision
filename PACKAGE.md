# phonon-vision 3.x — the complete package

(Version truth: `pyproject.toml` + the CHANGELOG top heading. This file's
merge-acceptance numbers are the 3.0.0 record; later batteries are in the
CHANGELOG per version.)

This directory is now the whole tool: the physics data layer, the exporter
that bakes materials, the single-file web navigator, and the analysis layers
added in v2. Before this merge it was the viewer alone, and a new material
could only be baked in the separate `phonon-core` tree and copied across by
hand. That is no longer true — the package bakes its own payloads.

The goal this serves is a common ground: many packages produce phonons
(phonopy, pheasy, DFPT via the Materials Project, machine-learned potentials),
and each ships its own conventions, units and file formats. phonon-vision
takes those outputs, normalizes them through one validated data layer, and
presents them through one viewer, so results from different provenances can be
compared without re-deriving anyone's conventions.

---

## What is here

```
src/phonon_core/        the data layer — parsers, JSON v1 schema, kinematics,
                        BZ geometry, projection, unfolding, scattering,
                        the D(q) engines (A1 phonopy-backed, A2 pure-numpy),
                        band_connect.py (band-following, 3.1.0), adp.py, and
                        the provenance importers: mp_import (Materials
                        Project), mlip_import (supercell Hessians, 3.2.0),
                        qe_import (Quantum ESPRESSO q2r, 3.3.0)
tools/                  pv_navigator_export.py  — bake band/mesh -> payload
                        pv_serve.py             — localhost drag-and-drop tier
                        gen_radius_tables.py    — regenerate radii_data.js
web/                    the navigator: index.html plus the math modules
                          kinematics.js   displacements (FROZEN, byte-identical
                                          to the v1 engine)
                          adp.js          thermal-ellipsoid tensors
                          polyhedra.js    convex hull + coordination polyhedra
                          radii_data.js   GENERATED radius tables
                          trajectory.js   extended-XYZ writer
                        pv_launcher.html  the drag-and-drop front end
python/phonon_adp.py    the pre-merge standalone ADP copy (see below)
blender/                the Blender add-on (own version line, 0.8.8; joined
                        at 3.0.1 — vendored wheel parity-checked against
                        this tree's src/)
tests/                  the full python battery, including fixtures
tests-js/               the full JS battery (DOM, kinematics, launcher, v2)
docs/                   schema and format documentation
```

## Versions folded in

| Component | Version | Provenance |
|---|---|---|
| data layer + exporter | 0.7.4 | `phonon-core`, latest |
| viewer + v2 analysis layers | 2.2.0 | v2 Tiers 1–3 |
| **this package** | **3.0.0** | the merge |

`phonon-core` was **not modified** by this merge and remains a valid tree in
its own right. This package is a superset, taken at each component's latest
version.

## Merge acceptance

The merge was accepted on the evidence below, not on the copy completing.

| Battery | Result |
|---|---|
| Python (`pytest`) | **384 passed, 4 skipped** |
| JavaScript (`node --test`) | **163 passed** |
| End-to-end bake | payload written by this tree's own exporter |

The JS number is the load-bearing one. It is the core's own 135-test DOM
suite — which pins trace counts, DOM invariants and the viewer guard
contracts — running against the **merged 4103-line viewer**, plus v2's 28.
All green, so the three v2 tiers broke no invariant the core relied on.

The 4 skips are the expected A1 text-band/mesh reconstruction skips. **The
skip count is a health signal**: any other number means a battery has gone
dark rather than passed. Check the `-rs` reasons before trusting a count.

The end-to-end bake is the claim that this package is self-sufficient:
`tools/pv_navigator_export.py` consumed `tests/fixtures/phonopy/SnSe2_band.hdf5`
plus a matching mesh and wrote a valid 7.3 MB `json_v1` payload
(`['Sn','Se','Se']`, 9 bands, 6-gon BZ, 96×96 surface grid, 96.7% in-BZ
coverage) using only files inside this tree.

## Running it

```bash
# the viewer — no server, no build step
open web/index.html

# the full python battery (needs numpy, pyyaml, h5py, scipy, phonopy, jsonschema)
PYTHONPATH=$PWD/src python -m pytest -q -rs        # expect 384 passed, 4 skipped

# the JS battery — jsdom is deliberately NOT vendored
npm install jsdom --no-save
node --test tests-js/navigator_dom.test.mjs tests-js/kinematics.test.mjs \
            tests-js/launcher_dom.test.mjs tests-js/adp.test.mjs   # expect 163

# bake a new material
PYTHONPATH=$PWD/src python tools/pv_navigator_export.py BAND.hdf5 [MESH.hdf5] \
    -o navigator_data.js --group-a <species> --dim 2
```

The python battery takes ~3.5 minutes and several GB; run it detached rather
than under a short timeout.

## Public edition

This tree is the public edition of the package. Development datasets that are
not published (private research fixtures) and the neutron-weighted band
channel are not part of it; the fixture-guarded tests that depend on external
datasets skip cleanly when the data is absent, and each names its dataset in
its skip reason.

## Two known duplications, both checked

**The ADP module exists twice.** `src/phonon_core/adp.py` is canonical — a
wheel built from `src/` carries it, and `from phonon_core.adp import ...`
works without the repository layout. `python/phonon_adp.py` is the pre-merge
standalone copy, retained unmodified. Duplicated numerics drift silently, so
`tests/test_adp_parity.py` asserts the two agree **bitwise** across four
temperatures and three frequency floors, plus their constants, their derived
helpers and their error behaviour. When the standalone copy is removed, keep
the packaged one; the parity test is then deleted with it.

**The viewer's `index.html` differs from `phonon-core`'s.** This tree's is the
superset (4103 vs 3116 lines) and is the one to develop. The four vendored
libraries (`kinematics.js`, `three.min.js`, `plotly.min.js`,
`TrackballControls.js`) were verified byte-identical between the two trees
before the merge, so only the viewer diverges, and deliberately.

## Where to develop next

The items this merge unblocks:

- **Band-following** by maximum overlap, so branch indices stay meaningful
  across crossings. Currently the exported trajectory's `branch=` metadata
  inherits that ambiguity.
- **A convergence study of the Tier 1 displacement tensors** against mesh
  density, which would move them from documented-qualitative to quantitative.
  This needs meshes denser than the bundled payloads carry — and now that the
  exporter lives here, they can be baked here.
- **More input provenances on the common ground**: the `mp_import` layer
  already normalizes Materials Project pheasy and DFPT outputs; MLIP-derived
  force constants are the natural next source.
