"""First Brillouin zone helper (phonon_core.bz)."""
import numpy as np
import pytest

pytest.importorskip("scipy")
from phonon_core import bz


def test_hexagonal_bz_is_regular_hexagon():
    # exactly-regular hexagonal reciprocal lattice (|b1|=|b2|=0.3, 60° apart);
    # matches the SnSe2 fixture to full precision (0.3*sqrt(3)/2, 0.15)
    a = 0.3
    recip = np.array([[a * np.sqrt(3) / 2, a / 2, 0.0],
                      [0.0, a, 0.0], [0.0, 0.0, 0.04]])
    v = bz.first_brillouin_zone_2d(recip)
    assert len(v) == 6
    r = np.linalg.norm(v, axis=1)
    assert np.allclose(r, r[0], rtol=1e-6)                 # regular
    ang = np.sort(np.degrees(np.arctan2(v[:, 1], v[:, 0])))
    spacing = np.diff(np.r_[ang, ang[0] + 360])
    assert np.allclose(spacing, 60.0, atol=1e-6)           # 6-fold symmetric


def test_square_bz_is_square():
    v = bz.first_brillouin_zone_2d(np.diag([1.0, 1.0, 1.0]))
    assert len(v) == 4
    assert np.allclose(np.linalg.norm(v, axis=1), np.sqrt(0.5), atol=1e-9)


def test_rectangular_bz_is_rectangle():
    v = bz.first_brillouin_zone_2d(np.diag([1.0, 2.0, 1.0]))
    assert len(v) == 4
    assert np.allclose(np.unique(np.round(np.abs(v[:, 0]), 9)), 0.5)
    assert np.allclose(np.unique(np.round(np.abs(v[:, 1]), 9)), 1.0)


def test_accepts_2x2_input():
    v = bz.first_brillouin_zone_2d(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert len(v) == 4


def test_vertices_ordered_ccw():
    a = 0.3
    recip = np.array([[a * np.sqrt(3) / 2, a / 2, 0.0],
                      [0.0, a, 0.0], [0.0, 0.0, 0.04]])
    v = bz.first_brillouin_zone_2d(recip)
    ang = np.arctan2(v[:, 1], v[:, 0])
    assert np.all(np.diff(ang) > 0)                        # strictly increasing


def test_bz_raises_clear_error_when_scipy_missing(monkeypatch):
    """Simulate scipy absent: first_brillouin_zone_2d must raise an ImportError whose
    message names the [export] extra (mirrors unfolding's guard test)."""
    import sys, numpy as np, pytest
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    with pytest.raises(ImportError) as ei:        # catch ImportError, assert on the MESSAGE
        bz.first_brillouin_zone_2d(np.eye(2))
    assert "phonon_core[export]" in str(ei.value)
    assert "scipy" in str(ei.value)
