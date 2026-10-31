# Committed phonopy fixtures (parser-equivalence)

Used by `tests/test_parser_equivalence.py`, which runs by default off these
files (no `$PHONON_CORE_FIXTURES` needed). They are the one place real binary
fixtures earn their place in the repo: the test's whole point is to compare
phonopy's *own* YAML emitter against its *own* HDF5 emitter.

- `2d_WSe2_band.{yaml,hdf5}` — the real WSe2 band pair (Γ–M–K–Γ, 606 q-points,
  3 atoms), committed verbatim from phonopy. ~3.9 MB total.

- `vasp_SnSe2_mesh.{yaml,hdf5}` — the first **64** q-points of the real
  91×91×1 SnSe2 mesh (the full mesh.yaml is 41 MB, too large to vendor). Each
  format was sliced INDEPENDENTLY from phonopy's own output — the HDF5 by a
  bytewise dataset copy of `qpoint/weight/frequency/eigenvector[:64]`, the YAML
  by keeping the header through `phonon:` plus the first 64 top-level
  `- q-position:` blocks verbatim — so the pair still compares phonopy's two
  emitters against each other, just on 64 of 8281 q-points. The retained header
  carries the full material block; `mesh:` stays `[91,91,1]` (the parser treats
  the result as a symmetry-reduced grid, which is fine here). ~0.4 MB total.

To regenerate the mesh subset, re-slice the full `vasp_SnSe2_mesh.{yaml,hdf5}`
the same way (first 64 q-points from each format independently). Point
`$PHONON_CORE_FIXTURES` at a directory holding the full-size files to run the
same test against the complete grid.
