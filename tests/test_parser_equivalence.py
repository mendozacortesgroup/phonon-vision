"""Parser-equivalence regression (Deliverable 3): phonopy's YAML and HDF5
outputs must carry the SAME physics.

Promotes the prior one-time bytewise hand-check to standing CI. For both a band
pair (WSe2) and a mesh pair (SnSe2) it asserts q_points, frequencies, and RAW
COMPLEX eigenvectors agree within documented ceilings. A failure here is a real
convention bug between phonopy's yaml and hdf5 emitters — exactly what this test
exists to catch.

Fixtures (committed, run by default — NO external setup needed)
---------------------------------------------------------------
Both pairs live in tests/fixtures/phonopy/ and are committed so a clean
``pytest`` runs both halves with zero skips:

- 2d_WSe2_band.{yaml,hdf5}    — the real WSe2 band pair, verbatim phonopy output.
- vasp_SnSe2_mesh.{yaml,hdf5} — the first 64 q-points of the real SnSe2 mesh,
  sliced INDEPENDENTLY from phonopy's own yaml (text) and hdf5 (binary), so the
  pair still compares phonopy's two emitters against each other (just on 64 of
  the 8281 q-points). The full 8281-point mesh yaml is 41 MB — too large to
  vendor — so the subset is what earns its place here.

$PHONON_CORE_FIXTURES (optional) overrides the directory, e.g. to point the same
filenames at the full local mesh; if a file is missing there it falls back to the
committed copy, so the default always executes.

Tolerance ceilings (see docs/conformance.md for the full reasoning)
-------------------------------------------------------------------
- q_points    5e-8  : measured max ~4.97e-8 (band) / ~4.95e-8 (mesh). Measured
  and correct; left as-is.
- frequencies 5e-11 : measured max ~5.0e-11 (both). Measured and correct.
- eigenvectors 1e-12 : a principled float-floor gate. The yaml-text vs
  hdf5-binary round-trip noise lives at ~1e-14 (measured max ~7.05e-15; every
  component < 1e-13, mean ~3e-15), so a literal ~7e-15 ceiling is below what real
  data hits and would fail the day it is enforced. This gate exists to catch
  convention / sign / phase bugs — which are ORDER UNITY at finite q — not float
  noise; 1e-12 sits ~9 orders below a real bug and comfortably above the floor.
"""
import os

import numpy as np
import pytest

# h5py honors this only if set before the file is opened.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from phonon_core import (
    load_band_yaml, load_band_hdf5, load_mesh_yaml, load_mesh_hdf5,
)

CEIL_Q = 5e-8       # measured ~4.97e-8 / ~4.95e-8
CEIL_FREQ = 5e-11   # measured ~5.0e-11
# eigenvector gate: catch order-unity convention/sign/phase bugs, NOT the
# ~1e-14 text/binary float-representation floor (measured max ~7.05e-15).
CEIL_EIG = 1e-12

# Committed fixtures run by default; $PHONON_CORE_FIXTURES is an optional
# override dir (searched first, with fallback to the committed copy).
_COMMITTED = os.path.join(os.path.dirname(__file__), "fixtures", "phonopy")
_OVERRIDE = os.environ.get("PHONON_CORE_FIXTURES")


def _resolve(name):
    """First existing path among [override dir, committed dir]."""
    if _OVERRIDE:
        p = os.path.join(_OVERRIDE, name)
        if os.path.isfile(p):
            return p
    p = os.path.join(_COMMITTED, name)
    if os.path.isfile(p):
        return p
    pytest.skip(f"fixture {name!r} not found in committed dir {_COMMITTED} "
                f"nor $PHONON_CORE_FIXTURES")


def _report(kind, dq, df, de):
    assert dq < CEIL_Q, f"{kind} q_points max|Δ|={dq:.3e} exceeds {CEIL_Q:.0e}"
    assert df < CEIL_FREQ, f"{kind} frequencies max|Δ|={df:.3e} exceeds {CEIL_FREQ:.0e}"
    assert de < CEIL_EIG, f"{kind} eigenvectors max|Δ|={de:.3e} exceeds {CEIL_EIG:.0e}"


def test_band_yaml_hdf5_equivalence():
    by = load_band_yaml(_resolve("2d_WSe2_band.yaml"))
    bh = load_band_hdf5(_resolve("2d_WSe2_band.hdf5"))
    dq = float(np.max(np.abs(by.q_points - bh.q_points)))
    df = float(np.max(np.abs(by.frequencies - bh.frequencies)))
    de = float(np.max(np.abs(by.eigenvectors - bh.eigenvectors)))
    _report("band", dq, df, de)


def test_mesh_yaml_hdf5_equivalence():
    my = load_mesh_yaml(_resolve("vasp_SnSe2_mesh.yaml"))
    # mesh.hdf5 carries no material; supply it from the yaml.
    mh = load_mesh_hdf5(_resolve("vasp_SnSe2_mesh.hdf5"), material=my.material)
    dq = float(np.max(np.abs(my.q_points - mh.q_points)))
    df = float(np.max(np.abs(my.frequencies - mh.frequencies)))
    assert my.eigenvectors is not None and mh.eigenvectors is not None, \
        "both mesh sources must carry eigenvectors"
    de = float(np.max(np.abs(my.eigenvectors - mh.eigenvectors)))
    _report("mesh", dq, df, de)
