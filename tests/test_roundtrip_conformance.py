"""Round-trip-through-conformance tests (Deliverable 2).

The headline: a document with ONE atom's eigenvector rotated by exp(i*0.3) at a
finite-q mode round-trips LOSSLESSLY (the bytes are consistent — the existing
json_v1 round-trip would pass it) yet is CAUGHT at tier 3 by the conformance
gate when compared to the pristine truth. Lossless != physically correct.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from phonon_core import load_band_hdf5, load_mesh_yaml
from phonon_core.types import BandPath
from phonon_core.io import (
    PhononDocument, write_json_v1, read_json_v1, roundtrip_conformance,
    select_reference_modes,
)

_HERE = os.path.dirname(__file__)
_FIX = os.path.join(_HERE, "fixtures", "phonopy")
_BAND = os.path.join(_FIX, "SnSe2_band.hdf5")
_MESH = os.path.join(_FIX, "vasp_SnSe2_mesh.yaml")


def _band_doc():
    bp = load_band_hdf5(_BAND)
    return PhononDocument(material=bp.material, band=bp, source_version="2.x")


def _merged_doc():
    bp = load_band_hdf5(_BAND)
    mg = load_mesh_yaml(_MESH)
    mg.material = bp.material   # same crystal (verified); share the object
    return PhononDocument(material=bp.material, band=bp, mesh=mg)


def test_roundtrip_conformance_clean_band():
    res = roundtrip_conformance(_band_doc())
    assert res.passed, str(res)
    assert res.tier1_frequencies.passed
    assert res.tier2_magnitude.passed
    assert res.tier3_complex_aligned.passed
    assert res.has_finite_q                       # Γ AND finite q covered


def test_roundtrip_conformance_clean_merged():
    res = roundtrip_conformance(_merged_doc())
    assert res.passed, str(res)
    assert res.has_finite_q


def _finite_q_mode_in_selection(doc):
    """A (q_index, band_index) that select_reference_modes covers and is finite-q."""
    for m in select_reference_modes(doc):
        if m.block == "band" and m.is_finite_q:
            return m.q_index, m.band_index
    raise AssertionError("no finite-q band mode selected")


def test_phase_corruption_caught_at_tier3_but_lossless_roundtrip_passes(tmp_path):
    # pristine truth
    pristine = _band_doc()
    qi, bi = _finite_q_mode_in_selection(pristine)
    assert np.linalg.norm(pristine.band.q_points[qi]) > 1e-6   # finite q

    # corrupted copy: rotate ONE atom's eigenvector by exp(i*0.3) at (qi, bi)
    ev = pristine.band.eigenvectors.copy()
    atom = 1
    assert np.linalg.norm(ev[qi, bi, atom]) > 0.1, "corrupt atom barely moves"
    ev[qi, bi, atom] = ev[qi, bi, atom] * np.exp(1j * 0.3)
    corrupted_band = BandPath(
        material=pristine.material, q_points=pristine.band.q_points,
        distances=pristine.band.distances, frequencies=pristine.band.frequencies,
        eigenvectors=ev, segment_nqpoint=pristine.band.segment_nqpoint,
        labels=pristine.band.labels)
    corrupted = PhononDocument(material=pristine.material, band=corrupted_band)

    # (A) LOSSLESS byte round-trip PASSES the corrupted doc: write -> read ->
    #     eigenvectors come back bit-identical. The byte gate sees nothing wrong.
    p = str(tmp_path / "corrupted.json")
    write_json_v1(corrupted, p)
    reread = read_json_v1(p)
    assert np.array_equal(reread.band.eigenvectors, corrupted.band.eigenvectors), \
        "json_v1 round-trip should be lossless"

    # (B) CONFORMANCE gate vs the pristine truth CATCHES it at tier 3, while
    #     tier 2 (|e|, unchanged by a phase rotation) still passes.
    res = roundtrip_conformance(corrupted, reference=pristine)
    assert res.tier2_magnitude.passed, str(res)
    assert not res.tier3_complex_aligned.passed, \
        "tier 3 must catch the per-atom phase corruption"
    assert res.first_failed_tier.name.startswith("per-atom complex")
    assert res.tier3_complex_aligned.max_abs_dev > 0.05

    # headline, asserted together: bytes consistent, physics NOT.
    assert np.array_equal(reread.band.eigenvectors, corrupted.band.eigenvectors)
    assert not res.passed
