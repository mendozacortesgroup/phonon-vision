# phonon-vision

Phonon analysis and visualization on a common ground.

Many packages produce phonons — phonopy, the Materials Project pipelines
(pheasy, DFPT), machine-learned interatomic potentials, Quantum ESPRESSO,
CRYSTAL — and each ships its own conventions, units and file formats.
phonon-vision normalizes those outputs through one validated data layer and
presents them through one interactive viewer, so results from different codes
can be compared side by side without re-deriving anyone's conventions.

## Quick look

Open `web/index.html` in any modern browser. No server, no build step; the
libraries and two example materials (SnSe2, graphene) are bundled, so it runs
offline from `file://`.

- **Dispersion panel** — click any point on a band and the crystal below
  animates that exact mode (mass-weighted displacements, Bloch phases across
  the supercell).
- **3D band surfaces** (2D materials) — ω(kx, ky) sheets over the whole
  Brillouin zone, colored by species character, with extended-zone tiling and
  fold-to-first-zone clicks.
- **Thermal ellipsoids** — the ADP tensor U_s(T) computed from the phonons
  over the q-mesh, with a temperature slider; ORTEP-style probability
  surfaces, zero-point motion included.
- **Coordination polyhedra, trajectory export, projected DOS** and more —
  every channel is presence-keyed, so payloads with more physics light up
  more panels.

## The data layer

`src/phonon_core/` is an installable package (`pip install .`):

| Source | Entry point |
|---|---|
| phonopy `band.yaml` / `band.hdf5` / `mesh.*` | `phonon_core.parsers` |
| Materials Project pheasy / DFPT | `phonon_core.mp_import` |
| MLIP Hessians (numpy arrays) | `phonon_core.mlip_import.convert_hessian` |
| Quantum ESPRESSO q2r `.fc` | `phonon_core.qe_import.convert_qe_fc` |
| Quantum ESPRESSO ph.x `.dyn` sets | `phonon_core.qe_import.convert_qe_dyn` |
| CRYSTAL fort.25 band structures | `phonon_core.crystal_import` |

plus a pure-numpy dynamical-matrix engine (`phonon_core.engine`) that
evaluates D(q) at arbitrary q from force constants — validated against
phonopy to ~1e-14 THz — and analysis layers: ADPs (`phonon_core.adp`),
band-following by eigenvector overlap (`phonon_core.band_connect`),
defect-supercell unfolding (`phonon_core.unfolding`).

Physics conventions are frozen and tested: signed frequencies (imaginary
modes are negative, and they are physics, not noise), mass-weighted
eigenvectors with Σ|e|² = 1, explicit unit constants, refuse-don't-guess
importers that raise with the reason instead of silently normalizing.

## Baking your own material

```
pip install ".[engine,export]"
python tools/pv_navigator_export.py band.hdf5 mesh.hdf5 \
    --engine phonopy.yaml -o navigator_data.js --also-html
```

`--engine` switches the surface and mesh-mode blocks to exact D(q) evaluation
(no interpolation); the payload records provenance (code versions, input
hashes, cross-check residuals) so every number in the viewer can be traced.

## Blender add-on

`blender/` contains a Blender 4.5/5.2 add-on that renders the same payloads
as publication-quality 3D scenes — atoms, bonds, mode animation, live D(q)
scrubbing along the band path — with the same engine vendored as a wheel so
the physics in Blender is byte-identical to the data layer.

## Tests

```
pip install ".[dev]"
pytest -q -rs          # python battery
npm install jsdom --no-save
node --test tests-js/navigator_dom.test.mjs tests-js/kinematics.test.mjs \
            tests-js/launcher_dom.test.mjs tests-js/adp.test.mjs
```

Tests that need large external datasets (real DFT outputs) skip cleanly and
name their dataset in the skip reason; everything else runs from committed
fixtures.

## License

MIT — see `LICENSE`. Bundled third-party libraries (plotly.js, three.js)
keep their own licenses.
