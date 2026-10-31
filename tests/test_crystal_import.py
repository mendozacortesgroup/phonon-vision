"""Tests for crystal_import (CRYSTAL23 fort.25 + .out).

The oracle is cross-FILE: the fort.25 Γ row must equal the frequencies the
.out itself prints in its DISPERSION K POINT 1 block — two independent
outputs of the same run, parsed by two different code paths here.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phonon_core.crystal_import import (  # noqa: E402
    convert_crystal_bands, read_crystal_out_structure, read_fort25,
)
from phonon_core.io.json_v1 import THZ_TO_CM1  # noqa: E402

BASE = os.environ.get("PV_CRYSTAL23_EXAMPLES",
                      os.path.join(os.path.dirname(__file__), "..", "..",
                                   "crystal23-examples", "r2SCANh-noD3"))
F25 = os.path.join(BASE, "fort.25")
OUT = os.path.join(BASE, "1T-SnSe2_SLAB_1L_r2SCANh-noD3_phonon_7x7_HUGEGRID.out")

pytestmark = pytest.mark.skipif(not os.path.exists(F25),
                                reason="CRYSTAL23 examples not present")


def test_fort25_structure():
    segs = read_fort25(F25)
    assert [(s["n_bands"], s["n_points"]) for s in segs] == [(9, 1000), (9, 999),
                                                             (9, 999)]
    assert segs[0]["from_int"] == (0, 0, 0) and segs[0]["to_int"] == (3, 0, 0)
    assert segs[2]["to_int"] == (0, 0, 0)
    for s in segs:
        assert np.isfinite(s["freq_thz"]).all()


def test_gamma_matches_the_out_dispersion_block():
    """Cross-file oracle: fort.25's first point vs the .out's own K1 list."""
    seg0 = read_fort25(F25)[0]
    got = np.sort(seg0["freq_thz"][0] * THZ_TO_CM1)
    want = np.sort([-0.2375, -0.2375, -0.0446, 115.9186, 115.9186,
                    166.8479, 166.8479, 194.7562, 281.0732])
    assert np.abs(got - want).max() < 5e-4, got


def test_unit_constant_matches_the_groups_script():
    """PhononBands.py hard-codes 0.029979; ours is the frozen 1/THZ_TO_CM1."""
    assert abs(1.0 / THZ_TO_CM1 - 0.029979) < 1e-6


def test_out_structure_is_the_1t_monolayer():
    st = read_crystal_out_structure(OUT)
    assert sorted(st["symbols"]) == ["Se", "Se", "Sn"]
    assert st["dimensionality"] == 2
    assert st["masses"][st["symbols"].index("Sn")] == pytest.approx(119.9022)
    a = np.linalg.norm(st["lattice"][0])
    b = np.linalg.norm(st["lattice"][1])
    assert a == pytest.approx(b, rel=1e-9)          # hexagonal
    assert 3.6 < a < 3.9                             # 1T-SnSe2 at r2SCANh
    assert st["lattice"][2][2] == pytest.approx(500.0)   # slab vacuum
    cos = st["lattice"][0] @ st["lattice"][1] / (a * b)
    assert cos == pytest.approx(-0.5, abs=1e-9)      # 120 degrees
    assert st["path"] == [((0, 0, 0), (0.5, 0, 0)),
                          ((0.5, 0, 0), (1 / 3, 1 / 3, 0)),
                          ((1 / 3, 1 / 3, 0), (0, 0, 0))]


def test_payload_is_honest(tmp_path):
    out = str(tmp_path / "p.js")
    payload, info = convert_crystal_bands(F25, OUT, output=out)
    assert info["path_labels"] == ["Γ", "M", "K", "Γ"]
    assert info["n_q"] == 2998 and info["n_bands"] == 9
    band = payload["band"]
    d = np.array(band["distance"])
    assert np.all(np.diff(d) >= 0) and d[0] == 0.0
    # continuity within the fine path (no scrambled segment stitching)
    f = np.array(band["frequency"])
    assert np.abs(np.diff(f, axis=0)).max() < 0.2, "band discontinuity"
    # the honesty contract: eigenvectors and character are exactly zero, and
    # the payload SAYS why
    assert not np.any(np.array(band["eigenvectors_real"]))
    assert not np.any(np.array(band["character"]))
    assert "frequencies only" in payload["meta"]["source"]["arbitrary_q_reason"]
    assert "ZERO" in payload["meta"]["conventions"]["eigenvectors"]
    # endpoint closure: last q returns to Gamma
    assert np.allclose(band["q_frac"][-1], [0, 0, 0], atol=1e-12)
    text = open(out).read()
    assert text.startswith("window.NAV_DATA = ")
    json.loads(text[len("window.NAV_DATA = "):].rstrip().rstrip(";"))


def test_mismatched_inputs_are_loud(tmp_path):
    # a fort.25 whose band count cannot belong to the .out's primitive cell
    segs_text = ("-%-1BAND    6  4 0.0E+00 1.0E-01 0.0E+00\n"
                 " 0.0E+00 2.5E+03\n    0    0    0    3    0    0\n"
                 + (" 1.00000E+02" * 6 + "\n") * 4)
    bad = tmp_path / "bad.f25"
    bad.write_text(segs_text)
    with pytest.raises(ValueError, match="segments|bands"):
        convert_crystal_bands(str(bad), OUT)
    # truncated frequency stream
    t25 = tmp_path / "trunc.f25"
    t25.write_text(segs_text.rsplit("\n", 2)[0] + "\n")
    with pytest.raises(ValueError, match="expected"):
        read_fort25(str(t25))
