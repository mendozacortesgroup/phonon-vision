"""Physics tests for phonon_adp.

The suite is built so that every claim is checked against something
INDEPENDENT of the implementation: a closed-form limit, an analytic
single-oscillator answer, or a symmetry the tensor must obey. A test that
only re-runs the same arithmetic proves nothing.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from phonon_adp import (  # noqa: E402
    AMU, HBAR, KB, THZ_TO_RAD_S,
    adp_tensors, ellipsoid_scale, principal_axes, u_isotropic,
)

WEB = os.path.join(os.path.dirname(__file__), "..", "web")


def load_payload(slug):
    path = os.path.join(WEB, f"navigator_data_{slug}.js")
    if not os.path.exists(path):
        pytest.skip(f"payload {slug} not present")
    with open(path) as fh:
        return json.loads(fh.read().split("=", 1)[1].strip().rstrip(";"))


def mesh_of(d):
    mm = d["mesh_modes"]
    return (np.array(mm["frequency"]), np.array(mm["eigenvectors_real"]),
            np.array(mm["eigenvectors_imag"]),
            np.array(d["meta"]["material"]["masses"]))


# ---------------------------------------------------------------- analytic

def single_mode_inputs(freq_thz, mass_amu, direction=(1.0, 0.0, 0.0)):
    """One atom, one mode, one q-point: U has a closed form."""
    e = np.zeros((1, 1, 1, 3))
    e[0, 0, 0, :] = np.asarray(direction, dtype=float)
    e /= np.linalg.norm(e)
    f = np.array([[float(freq_thz)]])
    return f, e, np.zeros_like(e), np.array([float(mass_amu)])


def test_single_oscillator_matches_closed_form():
    """U = hbar/(2 m w) coth(hbar w / 2 kB T) for one mode — no sum, no mesh."""
    nu, m, T = 4.0, 50.0, 300.0
    f, er, ei, masses = single_mode_inputs(nu, m)
    U, _ = adp_tensors(f, er, ei, masses, T, freq_floor_thz=0.0)

    w = THZ_TO_RAD_S * nu
    expect = HBAR / (2.0 * m * AMU * w) / np.tanh(HBAR * w / (2.0 * KB * T)) * 1e20
    assert U[0][0, 0] == pytest.approx(expect, rel=1e-12)
    # motion is purely along x: every other component is exactly zero
    assert U[0][1, 1] == 0.0 and U[0][2, 2] == 0.0


def test_zero_point_motion_survives_at_low_temperature():
    """T -> 0 leaves hbar/(2 m w), NOT zero. A classical ADP would vanish."""
    nu, m = 6.0, 30.0
    f, er, ei, masses = single_mode_inputs(nu, m)
    w = THZ_TO_RAD_S * nu
    zero_point = HBAR / (2.0 * m * AMU * w) * 1e20

    U_cold, _ = adp_tensors(f, er, ei, masses, 1e-3, freq_floor_thz=0.0)
    assert U_cold[0][0, 0] == pytest.approx(zero_point, rel=1e-9)
    assert U_cold[0][0, 0] > 0.0

    # The two limits are distinguished by their SHAPE in T, not by a magic
    # threshold: the quantum ADP saturates at the zero-point value while the
    # classical one stays strictly proportional to T all the way down.
    def q(T):
        return adp_tensors(f, er, ei, masses, T, freq_floor_thz=0.0)[0][0][0, 0]

    def c(T):
        return adp_tensors(f, er, ei, masses, T, freq_floor_thz=0.0,
                           classical=True)[0][0][0, 0]

    assert q(1e-4) == pytest.approx(q(1e-3), rel=1e-9)       # saturated
    assert c(1e-3) / c(1e-4) == pytest.approx(10.0, rel=1e-9)  # linear in T
    assert c(1e-3) < 1e-4 * zero_point                       # and negligible


def test_high_temperature_reaches_equipartition():
    """coth(x) -> 1/x, so the quantum U must converge on k_B T / (m w^2)."""
    d = load_payload("snse2")
    f, er, ei, masses = mesh_of(d)
    for T, tol in ((300.0, 2e-1), (2000.0, 2e-3), (20000.0, 2e-5)):
        Uq, _ = adp_tensors(f, er, ei, masses, T)
        Uc, _ = adp_tensors(f, er, ei, masses, T, classical=True)
        rel = abs(u_isotropic(Uq)[0] / u_isotropic(Uc)[0] - 1.0)
        assert rel < tol, f"T={T}: quantum/classical off by {rel:.2e}"


def test_scales_linearly_with_temperature_when_classical():
    d = load_payload("snse2")
    f, er, ei, masses = mesh_of(d)
    U1, _ = adp_tensors(f, er, ei, masses, 100.0, classical=True)
    U2, _ = adp_tensors(f, er, ei, masses, 400.0, classical=True)
    assert u_isotropic(U2)[0] / u_isotropic(U1)[0] == pytest.approx(4.0, rel=1e-12)


def test_inverse_mass_scaling():
    """U ~ 1/m at fixed frequency: a heavy atom vibrates less."""
    nu = 5.0
    out = []
    for m in (12.0, 120.0):
        f, er, ei, masses = single_mode_inputs(nu, m)
        U, _ = adp_tensors(f, er, ei, masses, 300.0, freq_floor_thz=0.0)
        out.append(U[0][0, 0])
    assert out[0] / out[1] == pytest.approx(10.0, rel=1e-12)


# ------------------------------------------------------------- tensor form

@pytest.mark.parametrize("slug", ["snse2", "graphene"])
def test_tensor_is_symmetric_and_positive_definite(slug):
    d = load_payload(slug)
    f, er, ei, masses = mesh_of(d)
    U, _ = adp_tensors(f, er, ei, masses, 300.0)
    for s in range(U.shape[0]):
        assert np.allclose(U[s], U[s].T, atol=1e-18)
        vals, _ = principal_axes(U[s])
        assert vals.min() > 0.0, f"atom {s}: non-positive eigenvalue {vals}"


@pytest.mark.parametrize("slug", ["snse2", "graphene"])
def test_magnitudes_are_physically_plausible(slug):
    """U_iso at 300 K must land in the range real crystals occupy."""
    d = load_payload(slug)
    f, er, ei, masses = mesh_of(d)
    U, _ = adp_tensors(f, er, ei, masses, 300.0)
    for iso in u_isotropic(U):
        assert 1e-3 < iso < 0.5, f"{slug}: U_iso {iso} A^2 is off the physical scale"


def test_layered_material_is_out_of_plane_anisotropic():
    """SnSe2 is a van der Waals layered crystal: the weak interlayer restoring
    force makes out-of-plane motion much larger than in-plane. If this ever
    inverts, the eigenvector axis convention has been broken."""
    d = load_payload("snse2")
    f, er, ei, masses = mesh_of(d)
    U, _ = adp_tensors(f, er, ei, masses, 300.0)
    for s in range(U.shape[0]):
        in_plane = 0.5 * (U[s][0, 0] + U[s][1, 1])
        assert U[s][2, 2] > 2.0 * in_plane, f"atom {s}: U_zz {U[s][2,2]} vs in-plane {in_plane}"


def test_monotonic_in_temperature():
    d = load_payload("snse2")
    f, er, ei, masses = mesh_of(d)
    prev = None
    for T in (10.0, 50.0, 150.0, 300.0, 600.0):
        iso = u_isotropic(adp_tensors(f, er, ei, masses, T)[0])[0]
        if prev is not None:
            assert iso > prev, f"U_iso must increase with T (T={T})"
        prev = iso


# ------------------------------------------------------------- book-keeping

def test_frequency_floor_is_reported_not_silent():
    d = load_payload("snse2")
    f, er, ei, masses = mesh_of(d)
    _, info = adp_tensors(f, er, ei, masses, 300.0, freq_floor_thz=0.5)
    assert info["n_modes_dropped"] == int((np.abs(f) < 0.5).sum())
    assert info["n_modes_dropped"] > 0
    assert info["f_min_kept_thz"] >= 0.5
    assert info["n_q"] == f.shape[0]


def test_raising_the_floor_lowers_u():
    """Dropping soft modes removes the largest 1/w contributions."""
    d = load_payload("snse2")
    f, er, ei, masses = mesh_of(d)
    lo = u_isotropic(adp_tensors(f, er, ei, masses, 300.0, freq_floor_thz=0.05)[0])[0]
    hi = u_isotropic(adp_tensors(f, er, ei, masses, 300.0, freq_floor_thz=1.0)[0])[0]
    assert hi < lo


def test_rejects_malformed_input():
    f, er, ei, masses = single_mode_inputs(4.0, 50.0)
    with pytest.raises(ValueError):
        adp_tensors(f, er, ei, np.array([1.0, 2.0]), 300.0)          # atom mismatch
    with pytest.raises(ValueError):
        adp_tensors(f, er, ei, masses, 0.0)                          # T = 0
    with pytest.raises(ValueError):
        adp_tensors(f[0], er, ei, masses, 300.0)                     # wrong rank


# -------------------------------------------------------------- ellipsoids

def test_ortep_scale_factors_match_the_chi_squared_definition():
    """c must solve P(chi^2_3 <= c^2) = p. Checked against the closed-form
    CDF, so the table cannot drift from its own definition."""
    from math import erf, exp, pi, sqrt
    for pct in (50, 90, 99):
        c = ellipsoid_scale(pct)
        x = c * c
        cdf = erf(sqrt(x / 2.0)) - sqrt(2.0 * x / pi) * exp(-x / 2.0)
        assert cdf == pytest.approx(pct / 100.0, abs=1e-6)


def test_ellipsoid_scale_rejects_unknown_level():
    with pytest.raises(ValueError):
        ellipsoid_scale(42)


def test_principal_axes_reconstruct_the_tensor():
    d = load_payload("snse2")
    f, er, ei, masses = mesh_of(d)
    U, _ = adp_tensors(f, er, ei, masses, 300.0)
    vals, vecs = principal_axes(U[0])
    assert np.allclose(vecs @ np.diag(vals) @ vecs.T, U[0], atol=1e-15)
    assert np.allclose(vecs.T @ vecs, np.eye(3), atol=1e-12)
