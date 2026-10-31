"""mp_import Track B — MP dfpt phononwebsite -> json_v1 band document (0.6.90).

dfpt ships bands + eigenvectors, NO force constants: a PRECOMPUTED PATH. The
battery proves the conversion (through the FROZEN json_v1 reader/writer,
e18613db byte-identical), the honest labeling (engine=None, arbitrary_q
False), and the TWO convention findings this arc surfaced:

- MP website ``vectors`` are sqrt(M_avg/m)-displacement-scaled, NOT raw
  mass-weighted (the 0.6.89 pin holds for pheasy, is FALSE for the dfpt
  website export) — silicon is structurally blind to it (equal masses), the
  polar pair is not, and the correction is proven load-bearing;
- the cm^-1 unit residual vs the sibling bandstructure document is purely
  MULTIPLICATIVE (~2e-8 relative — conversion-constant noise), pinned as a
  relative bound, not an absolute one.
"""
import json
import os
import shutil

import numpy as np
import pytest

from conftest import DUAL_TRACK, MP_FIX, fixture_dir, requires_fixtures
from engine_metric import NU_FLOOR

pytestmark = requires_fixtures

#: absolute gate (doctrine, |nu| > NU_FLOOR); measured <= 3.1e-07 THz
ABS_GATE = 1e-5
#: the MULTIPLICATIVE pin: |dnu/nu| above 1 THz; measured ~2e-8
REL_GATE = 1e-7


def _dfpt_bs(label):
    b = json.load(open(os.path.join(MP_FIX, label, "dfpt_bandstructure.json")))
    qs = np.asarray(b["qpoints"], dtype=np.float64)
    fr = np.sort(np.asarray(b["frequencies"], dtype=np.float64).T, axis=1)
    return qs, fr, b


@pytest.fixture(scope="module")
def converted_b(tmp_path_factory):
    from phonon_core.mp_import import convert_dfpt
    cache = {}

    def _get(label):
        if label not in cache:
            out = tmp_path_factory.mktemp(f"mpb_{label.replace('-', '_')}")
            cache[label] = convert_dfpt(fixture_dir(label),
                                        str(out / "phonon_document.json"))
        return cache[label]

    return _get


@pytest.mark.parametrize("label", DUAL_TRACK)
def test_convert_honest_roundtrip(label, converted_b):
    """The written document: schema 1.1.0, band block intact BITWISE through
    the frozen reader, ``extensions == {}`` on the wire (engine=None — parsed,
    not engine-evaluated: the schema's own contract), arbitrary_q False with
    the reason, MP identifiers carried, source_code from the MP document."""
    from phonon_core.io.json_v1 import read_json_v1
    res = converted_b(label)
    doc = res["document"]
    assert doc.engine is None
    assert doc.metadata["arbitrary_q"] is False
    assert "no force constants" in doc.metadata["arbitrary_q_note"]
    assert doc.metadata["mp"]["phonon_method"] == "dfpt"
    assert doc.metadata["mp"]["identifier"]
    raw = json.load(open(res["json_v1"]))
    assert raw["schema_version"] == "1.1.0"
    assert raw["extensions"] == {}                   # honestly unprovenance'd
    assert raw["metadata"]["arbitrary_q"] is False
    re_doc = read_json_v1(res["json_v1"])
    assert np.array_equal(re_doc.band.frequencies, doc.band.frequencies)
    assert np.array_equal(re_doc.band.eigenvectors, doc.band.eigenvectors)
    assert np.array_equal(re_doc.band.q_points, doc.band.q_points)
    print(f"\n[mp-b] {label}: {res['info']['n_qpoints']} q x "
          f"{res['info']['n_bands']} bands, identifier "
          f"{res['info']['identifier']}, source_code={doc.source_code!r}, "
          f"has_nac={res['info']['has_nac']} — round-trip BITWISE")


@pytest.mark.parametrize("label", DUAL_TRACK)
def test_frequencies_match_bandstructure_doc(label, converted_b):
    """The website doc (cm^-1, through the frozen THZ_TO_CM1) reproduces the
    SIBLING bandstructure doc (THz) — same task id, independent export. The
    residual must be MULTIPLICATIVE (a unit-constant mismatch at the 1e-8
    relative level), not additive: |dnu| < 1e-5 absolute AND |dnu/nu| < 1e-7
    above 1 THz. An additive/data error would break the relative pin first."""
    res = converted_b(label)
    fe = np.sort(res["document"].band.frequencies, axis=1)
    qs_bs, fr, _b = _dfpt_bs(label)
    assert np.allclose(qs_bs, res["document"].band.q_points, atol=1e-10)
    big = np.abs(fr) > NU_FLOOR
    d_abs = float(np.abs(fe - fr)[big].max())
    hi = np.abs(fr) > 1.0
    d_rel = float((np.abs(fe - fr)[hi] / np.abs(fr)[hi]).max())
    print(f"\n[mp-b] {label}: website-vs-bandstructure max|dnu| = {d_abs:.3e} "
          f"THz, max relative (|nu|>1 THz) = {d_rel:.3e} — unit-constant "
          f"noise, multiplicative as required")
    assert d_abs < ABS_GATE
    assert d_rel < REL_GATE


@pytest.mark.parametrize("label", DUAL_TRACK)
def test_eigenvector_convention_finding(label, converted_b):
    """THE 0.6.90 FINDING, made executable. RAW website vectors (through the
    frozen reader, no correction) have per-mode norms M_avg/m on pure-species
    modes — deviating up to 0.037 (GaAs) and 1.99 (GaN) from 1 — while
    silicon's equal masses make the scaling INVISIBLE (norms exactly 1: the
    reason a monoatomic check could never catch this). The converter's
    corrected output is unit-norm mass-weighted and orthonormal per q at file
    precision."""
    from phonon_core.io.json_v1 import from_phononwebsite_json
    raw_doc = from_phononwebsite_json(
        json.load(open(os.path.join(MP_FIX, label, "dfpt_phononwebsite.json"))))
    raw_norm = (np.abs(raw_doc.band.eigenvectors) ** 2).sum(axis=(2, 3))
    raw_dev = float(np.abs(raw_norm - 1).max())
    res = converted_b(label)
    e = res["document"].band.eigenvectors
    fix_dev = float(np.abs((np.abs(e) ** 2).sum(axis=(2, 3)) - 1).max())
    E = e.reshape(e.shape[0], e.shape[1], -1)
    gram_dev = max(float(np.abs(E[i].conj() @ E[i].T
                                - np.eye(E.shape[1])).max())
                   for i in range(0, E.shape[0], 13))
    print(f"\n[mp-b] {label}: RAW website norms deviate |n-1| up to "
          f"{raw_dev:.3e}; corrected {fix_dev:.1e}; orthonormality dev "
          f"{gram_dev:.2e}")
    if label == "silicon_mp-149":
        assert raw_dev < 1e-9          # equal masses: structurally blind
    else:
        assert raw_dev > 0.03          # the finding is non-vacuous here
    assert fix_dev < 1e-9
    assert gram_dev < 1e-5             # file-precision orthonormality


def test_negctl_unit_scale_bites(tmp_path, converted_b):
    """Negative control: a fixture whose eigenvalues are scaled by 1% must
    FAIL the frequency oracle — both the absolute and the relative pin."""
    from phonon_core.mp_import import convert_dfpt
    src = fixture_dir("silicon_mp-149")
    bad = tmp_path / "doctored"
    bad.mkdir()
    for fn in ("dfpt_phononwebsite.json", "dfpt_metadata.json",
               "dfpt_bandstructure.json"):
        shutil.copy(os.path.join(src, fn), bad / fn)
    w = json.load(open(bad / "dfpt_phononwebsite.json"))
    w["eigenvalues"] = (np.asarray(w["eigenvalues"], float) * 1.01).tolist()
    json.dump(w, open(bad / "dfpt_phononwebsite.json", "w"))
    res = convert_dfpt(str(bad), str(tmp_path / "doc.json"))
    fe = np.sort(res["document"].band.frequencies, axis=1)
    _qs, fr, _b = _dfpt_bs("silicon_mp-149")
    big = np.abs(fr) > NU_FLOOR
    d = float(np.abs(fe - fr)[big].max())
    print(f"\n[mp-b negctl] 1%-scaled eigenvalues: max|dnu| = {d:.3f} THz "
          f"-> the oracle BITES")
    assert d > 0.05
    with pytest.raises(AssertionError):              # fail-then-pass
        assert d < ABS_GATE


def test_fused_tick_label_recorded(converted_b):
    """The 0.8.4 class of plot furniture exists IN MP'S SOURCE DATA: silicon's
    highsym_qpts fuses the path discontinuity into a compound 'K|U' label.
    Recorded executable so the Track-B consumer story stays honest (labels
    are display strings; q/frequencies/eigenvectors are exact regardless)."""
    res = converted_b("silicon_mp-149")
    labels = [l for pair in res["document"].band.labels for l in pair]
    fused = [l for l in labels if "|" in l]
    print(f"\n[mp-b] silicon segment labels: {res['document'].band.labels}")
    print(f"[mp-b] fused discontinuity labels (0.8.4-class furniture in the "
          f"SOURCE): {fused}")
    assert fused == ["K|U", "K|U"]     # end of one segment, start of the next
    assert sum(res["document"].band.segment_nqpoint) == \
        res["info"]["n_qpoints"]


@pytest.mark.parametrize("label", DUAL_TRACK)
def test_cross_method_record(label, converted_b, converted, tmp_path):
    """RECORD (no gate): dfpt stored bands vs Track A's pheasy FC through A1
    at the same q — two INDEPENDENT MP calculations of the same material by
    different methods. The delta is physics + method + NAC-handling, not a
    correctness signal for either track; it is recorded because the dual-track
    materials exist precisely to make such comparisons possible."""
    from phonon_core.engine import load_engine
    from conftest import a1_frequencies
    res_b = converted_b(label)
    qs = res_b["document"].band.q_points
    fr = np.sort(res_b["document"].band.frequencies, axis=1)
    res_a = converted(label)
    eng = load_engine(res_a["phonopy_yaml"])         # bare pheasy FC (no NAC)
    fe = a1_frequencies(eng, qs)
    big = np.abs(fr) > NU_FLOOR
    d = np.abs(fe - fr)[big]
    print(f"\n[mp-b record] {label}: dfpt-stored vs pheasy-FC(A1, bare) over "
          f"{len(qs)} q: max = {d.max():.3f} THz, median = "
          f"{np.median(d):.3f} THz (method + NAC-handling delta — recorded, "
          f"not gated)")
    assert np.all(np.isfinite(fe))
