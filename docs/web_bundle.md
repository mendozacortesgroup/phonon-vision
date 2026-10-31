# The web bundle

A *web bundle* is the single, self-describing artifact a web viewer loads. It
pairs the JSON v1 data document with the golden kinematics reference for a few
modes, so a JavaScript engine can load the data, recompute displacements, and
prove it reproduces phonon_core's physics — the cross-language correctness
contract, with no Python in the loop at view time.

It is produced by `phonon_core.io.write_web_bundle` (also `tools/pv_export.py
--bundle DIR`) and consumes the frozen json_v1 layer, the conformance checker,
and the kinematics engine without modifying any of them.

## Layout

```
bundle_dir/
  document.json            JSON v1 document (schema-valid) — the data
  golden_kinematics.npz    displacement frames + eigenvectors for the bundled modes
  golden_kinematics.json   conventions block + per-mode input spec + version
  bundle.json              manifest tying the three together
```

`bundle.json`:

```json
{
  "bundle_version": 1,
  "phonon_core_version": "0.3.0",
  "schema_version": "1.0.0",
  "document": "document.json",
  "document_sha256": "…",
  "golden_vectors": "golden_kinematics.npz",
  "golden_sidecar": "golden_kinematics.json",
  "golden_npz_sha256": "…",
  "n_bundled_modes": 5,
  "bundled_modes": ["gamma_acoustic", "gamma_optical", "m_point", "k_point", "generic"],
  "has_finite_q": true
}
```

The golden files are byte-for-byte the **0.2.0 golden-vector format** (see
`tools/gen_golden.py` and `docs/conformance.md`): the `.npz` carries the shared
material and slot arrays (`_material_*`, `_slot_j`, `_slot_n_int`) and, per mode,
`<name>__eigvec_raw`, `__eigvec_canon`, `__bloch`, `__disp_phi`, `__disp_traj`;
the `.json` sidecar carries the conventions block, the fixed kinematics inputs
(amplitude, phi, supercell, n_frames, mass-weighting), and per-mode metadata
(`block`, `q_index`, `band_index`, `q_point`, frequency, finite-q flag). Because
the format is identical, the same conformance tooling validates a bundle as
validates the 0.2.0 fixture.

The bundled modes deliberately span Γ **and** finite q — the finite-q modes are
the only ones that expose a per-atom phase error, so a contract without them
would be hollow.

## How a JS engine self-validates against this bundle

The contract is: *the document's eigenvectors, run through the documented
kinematics, must reproduce the bundled golden displacement frames, and must
match the bundled golden eigenvectors under the three-tier rule.* In Python this
is `phonon_core.io.verify_web_bundle(dir)`; a JS engine mirrors it:

1. **Load** `bundle.json`, then `document.json` and the golden files it names.
2. For each bundled mode, read its eigenvector from `document.json` at
   `(block, q_index, band_index)`. This is the candidate.
3. **Eigenvector conformance (three tiers).** Compare the candidate eigenvectors
   to the bundled `<name>__eigvec_raw` with the same three-tier logic as
   `conformance.check_against_reference`:
   - tier 1: frequencies match;
   - tier 2: per-atom `|e|` match;
   - tier 3: per-atom complex `e` match after aligning the global phase via the
     canonical rule — pivot derived from the reference (`argmax|ref|`) and
     applied to both, so it is tie-robust (see docs/conformance.md). Tier 3 at a
     finite-q mode is the load-bearing assertion.
4. **Displacement reproduction.** Using the conventions block
   (`u_{ls}(φ) = A·Re[e_s·exp(i(q·R_l+φ))]/√m_s`, mass-weighting, the canonical
   phase, the Bloch phase `2π q·n_int`, the phase grid) and the fixed inputs,
   recompute the displacement at `phi` and the full `[0,2π)` trajectory for each
   mode, and compare to the bundled `__disp_phi` / `__disp_traj`. Agreement at
   the float floor (≲1e-9) means the JS kinematics matches the reference engine.

If both checks pass, the JS engine is conformant: it animates the *same* physics
the Blender renderer does. If tier 3 fails at a finite-q mode while tier 2
passes, the JS engine has a per-atom phase bug — invisible to magnitude metrics
and at Γ, exactly the class the gate exists to catch.

## Producing a bundle

```
pv_export.py band.hdf5 mesh.yaml -o document.json --bundle ./mybundle
```

or programmatically:

```python
from phonon_core import load_band_hdf5
from phonon_core.io import PhononDocument, write_web_bundle, verify_web_bundle

bp = load_band_hdf5("band.hdf5")
doc = PhononDocument(material=bp.material, band=bp, source_version="2.x")
write_web_bundle(doc, "mybundle")
assert verify_web_bundle("mybundle").passed
```
