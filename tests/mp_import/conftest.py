"""Shared plumbing for the mp_import Track A battery (0.6.89).

- MP fixtures are a DEV ASSET, not shipped (bulky, public, regeneratable via
  the harvest script): every test here is fixture-guarded and SKIPS in a
  bare-zip audit — intended behaviour, not a gap. MP data is public, so no
  secret-material ship guard applies (unlike private research fixtures).
- The metric doctrine lives in tests/engine_metric.py (one spelling, 0.6.88);
  this conftest puts tests/ on sys.path so the subpackage can import it.
"""
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, os.pardir))          # -> engine_metric

#: The harvested fixtures root — a sibling of the source tree, like
#: Phonopy-examples (bulky public MP data; never inside the ship).
MP_FIX = os.path.normpath(os.path.join(
    _HERE, os.pardir, os.pardir, os.pardir,
    "materials_project_data", "harvest", "mp_fixtures"))

_HAVE_PHONOPY = True
try:                                                          # pragma: no cover
    import phonopy  # noqa: F401
except ImportError:                                           # pragma: no cover
    _HAVE_PHONOPY = False

requires_fixtures = pytest.mark.skipif(
    not (os.path.isdir(MP_FIX) and _HAVE_PHONOPY),
    reason="MP fixtures or phonopy not present (bare-zip audit) — mp_import "
           "battery skipped")

#: The three dual-track materials (pheasy FC + pheasy bands + dfpt docs).
DUAL_TRACK = ["silicon_mp-149", "gaas_mp-2534", "gan_mp-830"]
#: Every INTERNALLY CONSISTENT pheasy fixture (nacl_mp-1120767 is excluded:
#: its band document is malformed — 12 stored bands for a 12-site structure —
#: and serves as the converter-precondition NEGATIVE fixture instead).
CONSISTENT = ["silicon_mp-149", "gaas_mp-2534", "gan_mp-830",
              "cssni3_mp-27381", "mgo_mp-1192189"]
BROKEN_BAND_DOC = "nacl_mp-1120767"

#: The pinned NAC constant — single-sourced from the src module since 0.6.91
#: (one spelling, two consumers: the converter's BORN writer and this
#: battery). See mp_import/nac.py for the pin's provenance + derivation note.
from phonon_core.mp_import.nac import PHEASY_NAC_FACTOR  # noqa: E402,F401


def fixture_dir(label):
    return os.path.join(MP_FIX, label)


def load_meta(label):
    return json.load(open(os.path.join(MP_FIX, label, "pheasy_metadata.json")))


def load_band(label):
    """(qpoints (Nq,3), frequencies (Nq,Nb) ASCENDING per q, band dict)."""
    b = json.load(open(os.path.join(MP_FIX, label,
                                    "pheasy_bandstructure.json")))
    qs = np.asarray(b["qpoints"], dtype=np.float64)
    fr = np.sort(np.asarray(b["frequencies"], dtype=np.float64).T, axis=1)
    return qs, fr, b


@pytest.fixture(scope="session")
def converted(tmp_path_factory):
    """Convert-once cache: label -> convert_pheasy() result dict."""
    from phonon_core.mp_import import convert_pheasy
    cache = {}

    def _get(label):
        if label not in cache:
            out = tmp_path_factory.mktemp(f"mp_{label.replace('-', '_')}")
            cache[label] = convert_pheasy(fixture_dir(label), str(out))
        return cache[label]

    return _get


def a1_frequencies(eng, qs):
    """A1 frequencies at qs, ASCENDING per q — via the SAME evaluate_dq the
    whole toolchain uses."""
    from phonon_core.engine import evaluate_dq
    out = np.empty((len(qs), len(eng.primitive) * 3))
    for i, q in enumerate(qs):
        out[i] = evaluate_dq(eng, q)[0]
    return out


def write_born_file(path, label, eng):
    """Write a phonopy BORN file carrying MP's Born charges + eps_infinity
    with the PINNED pheasy factor — for A1's pre-existing BORN pass-through
    (``load_engine(..., born=path)``, unchanged since 0.6.78).

    TEST-ONLY: the converter itself never writes this (nac_applied = False);
    and its sidecar is deliberately not named BORN (the auto-discovery trap).
    phonopy's BORN format wants tensors for the symmetry-INDEPENDENT atoms of
    the primitive; MP's born array is per unit-cell site, which maps 1:1 onto
    primitive atoms here (all fixtures' primitive_matrix are det-1 unimodular
    basis changes — atom count preserved).
    """
    from phonon_core.mp_import.nac import write_phonopy_born
    m = load_meta(label)
    return write_phonopy_born(eng, m["born"], m["epsilon_static"], path)
