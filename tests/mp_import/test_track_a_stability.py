"""mp_import Track A — stability RECORDED from the data, not asserted from
memory (0.6.89).

Polymorph identities are not pinned: the harvest resolver chose
most-stable-with-phonons, so which fixtures are unstable was unknown a
priori. Measured (this battery):

- mgo_mp-1192189 is GENUINELY unstable (MP stored min -1.4831 THz) and the
  instability lives in the BARE force constants (reproduced without NAC at
  3.6e-11 THz) — the true-positive case for the 0.6.77 honesty badge.
- cssni3_mp-27381 is a STABLE polymorph — the confirmed stable case: the
  badge must NOT fire. A valid, needed negative case, not a coverage gap.
- nacl_mp-1120767 CANNOT serve as a stability oracle (its band document is
  malformed — see test_track_a_convert's precondition test; its earlier
  "missing instability" reading was that same 12-vs-36-band mismatch, not a
  physics result). Deliberately absent here, per the Option-A ruling.

AMENDMENT 2: for the anisotropic-Gamma materials (cssni3, mgo) the NAC
reproduction gate applies OFF-Gamma; the Gamma-row residual under the pinned
isotropic convention is PRINTED as the 0.6.91 acceptance target, not gated.
"""
import numpy as np
import pytest

from conftest import (
    CONSISTENT,
    a1_frequencies,
    load_band,
    load_meta,
    requires_fixtures,
    write_born_file,
)
from engine_metric import NU_FLOOR

pytestmark = requires_fixtures

FREQ_GATE = 1e-2       # THz
BADGE_FLOOR = 0.01     # THz — the 0.6.77 imaginary-badge display floor


def _lam(x):
    return np.sign(x) * x ** 2


@pytest.mark.parametrize("label", CONSISTENT)
def test_stability_verdict_reproduced(label, converted):
    """The converter reproduces MP's OWN stability verdict: the reconstructed
    (bare-FC) min frequency matches MP's stored-band min to < 1e-2 THz —
    compared via |dlambda| when the min sits below NU_FLOOR (a stable
    material's min is acoustic dust; dnu = dlambda/2nu is noise there).
    THEN the honesty badge fires iff min nu < the 0.6.77 floor, and
    sum_rules_breaking is available from the metadata to drive it."""
    from phonon_core.engine import load_engine
    res = converted(label)
    eng = load_engine(res["phonopy_yaml"])
    qs, fr, _b = load_band(label)
    fe = a1_frequencies(eng, qs)
    mn_r, mn_e = float(fr.min()), float(fe.min())
    if abs(mn_r) > NU_FLOOR:
        d = abs(mn_e - mn_r)
        metric = f"|dnu| = {d:.3e} THz"
        ok = d < FREQ_GATE
    else:
        d = abs(_lam(mn_e) - _lam(mn_r))
        metric = f"|dlambda| = {d:.3e} THz^2 (min below NU_FLOOR)"
        ok = d < FREQ_GATE ** 2
    badge = mn_e < -BADGE_FLOOR
    srb = load_meta(label).get("sum_rules_breaking")
    print(f"\n[mp-stability] {label}: MP min = {mn_r:+.4f} THz, "
          f"reconstructed min = {mn_e:+.4f} THz, {metric}; "
          f"badge fires: {badge}; sum_rules_breaking = {srb}")
    assert ok, metric
    # the badge tracks the DATA: fires iff MP's own verdict is unstable
    assert badge == (mn_r < -BADGE_FLOOR)
    # ... and the metadata carries the field the badge machinery may consult
    assert "sum_rules_breaking" in load_meta(label)


@pytest.mark.parametrize("label", ["cssni3_mp-27381", "mgo_mp-1192189"])
def test_anisotropic_nac_reproduction(label, converted, tmp_path):
    """The two anisotropic-Gamma pheasy-only materials: the full-band
    NAC-applied reproduction passes OFF-Gamma (< 1e-2 THz; measured cssni3
    5.0e-11, mgo 1.2e-05) — proving FC ingestion + atom order for the
    LARGE-cell fixtures (20 and 24 atoms) too. Gamma rows: AMENDMENT 2,
    printed as the 0.6.91 target."""
    from phonon_core.engine import load_engine
    res = converted(label)
    qs, fr, _b = load_band(label)
    on_g = np.all(np.abs(qs) < 1e-12, axis=1)
    eng_bare = load_engine(res["phonopy_yaml"])
    born_path = write_born_file(str(tmp_path / f"BORN_{label}"), label, eng_bare)
    eng = load_engine(res["phonopy_yaml"], born=born_path)
    fe = a1_frequencies(eng, qs)
    big = np.abs(fr) > NU_FLOOR
    d_off = float(np.abs(fe - fr)[~on_g[:, None] & big].max())
    d_gam = float(np.abs(fe - fr)[on_g[:, None] & big].max()) if on_g.any() else 0.0
    print(f"\n[mp-stability] {label}: NAC-applied OFF-Gamma max|dnu| = "
          f"{d_off:.3e} THz (gate {FREQ_GATE}, {int((~on_g).sum())} q)")
    print(f"[mp-stability] {label}: 0.6.91 target: needs anisotropic "
          f"nac_q_direction at Gamma, current residual = {d_gam:.6e} THz "
          f"({int(on_g.sum())} Gamma rows, isotropic pinned convention)")
    assert d_off < FREQ_GATE
