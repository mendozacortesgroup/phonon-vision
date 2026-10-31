"""mp_import NAC/LO-TO activation (0.6.91) — closes the 0.6.89 arc.

The four Γ-row acceptance targets printed by the 0.6.89 battery (isotropic
pinned convention): GaAs 5.694316e-01, GaN 5.708192e+00, CsSnI₃ 8.173086e-01,
MgO 4.252673e+00 THz. This battery CLOSES them: with the pinned pheasy NAC
convention (factor 15.633302300230191 + eps_infinity) AND the adjacent-
segment ``nac_q_direction`` rule at exact-Γ rows, all four collapse to
~5e-11 THz.

Activation path: ``convert_pheasy(nac=True)`` writes a phonopy ``BORN`` file
next to phonopy.yaml — A1's pre-existing auto-discovery (the 0.6.89 filename
trap, now the deliberate switch) applies the correction; A2 REFUSES the same
directory loudly (its 0.6.86 BORN refusal — no silent NAC-less numbers from
the vendored engine).
"""
import json
import os

import numpy as np
import pytest

from conftest import (
    CONSISTENT,
    a1_frequencies,
    fixture_dir,
    load_band,
    requires_fixtures,
)
from engine_metric import NU_FLOOR

pytestmark = requires_fixtures

FREQ_GATE = 1e-2       # THz — the standing oracle gate (off-Γ)
GAMMA_GATE = 1e-6      # THz — the Γ-closure gate (measured ~5e-11)

#: The 0.6.89 targets (printed by that battery, isotropic convention) —
#: asserted still-open before closure, so the closure is fail-then-pass.
TARGETS = {"gaas_mp-2534": 5.694316e-01, "gan_mp-830": 5.708192e+00,
           "cssni3_mp-27381": 8.173086e-01, "mgo_mp-1192189": 4.252673e+00}
POLAR = list(TARGETS)


@pytest.fixture(scope="module")
def converted_nac(tmp_path_factory):
    """Convert-once cache with nac=True."""
    from phonon_core.mp_import import convert_pheasy
    cache = {}

    def _get(label):
        if label not in cache:
            out = tmp_path_factory.mktemp(f"mpn_{label.replace('-', '_')}")
            cache[label] = convert_pheasy(fixture_dir(label), str(out),
                                          nac=True)
        return cache[label]

    return _get


# ---------------------------------------------------------------------------
# the activation switch itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label", CONSISTENT)
def test_nac_true_writes_born_and_honest_sidecar(label, converted_nac):
    """nac=True: a BORN file appears (first line = the PINNED factor), the
    sidecar says nac_applied true, and the default (nac=False) output is
    UNCHANGED — no BORN, nac_applied false (Track A behaviour preserved)."""
    from phonon_core.mp_import import PHEASY_NAC_FACTOR, convert_pheasy
    res = converted_nac(label)
    assert res["born"] is not None and os.path.basename(res["born"]) == "BORN"
    first = open(res["born"]).readline().strip()
    assert abs(float(first) - PHEASY_NAC_FACTOR) < 1e-12
    assert json.load(open(res["nac_sidecar"]))["nac_applied"] is True
    # default path regression: no BORN, nac_applied false
    import tempfile
    out = tempfile.mkdtemp()
    res0 = convert_pheasy(fixture_dir(label), out)
    assert res0["born"] is None
    assert not os.path.exists(os.path.join(out, "BORN"))
    assert json.load(open(res0["nac_sidecar"]))["nac_applied"] is False


def test_nac_requires_born_data(tmp_path):
    """A fixture without Born charges refuses nac=True loudly."""
    import shutil
    from phonon_core.mp_import import MPImportError, convert_pheasy
    src = fixture_dir("silicon_mp-149")
    bad = tmp_path / "no_born"
    bad.mkdir()
    for fn in ("pheasy_metadata.json", "pheasy_forceconstants.json"):
        shutil.copy(os.path.join(src, fn), bad / fn)
    m = json.load(open(bad / "pheasy_metadata.json"))
    m["born"] = None
    json.dump(m, open(bad / "pheasy_metadata.json", "w"))
    with pytest.raises(MPImportError, match="nac=True requires Born"):
        convert_pheasy(str(bad), str(tmp_path), nac=True)


# ---------------------------------------------------------------------------
# THE ARC-CLOSING ORACLE: converter BORN -> A1 auto-discovery -> MP's bands
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label", POLAR)
def test_nac_oracle_closes_targets(label, converted_nac):
    """End-to-end through the converter's OWN BORN file and A1's
    auto-discovery (load_engine with NO born argument):

    - off-Γ: the full-band reproduction passes the standing 1e-2 gate
      (measured 5.0e-11 .. 1.2e-05 THz);
    - exact-Γ rows: WITHOUT a direction the residual still equals the 0.6.89
      target (fail-then-pass: the closure is non-vacuous), and WITH the
      pinned adjacent-segment ``nac_q_direction`` rule it collapses below
      1e-6 THz — the 0.6.91 acceptance targets, CLOSED."""
    import phonopy as _ph_mod
    from phonon_core.engine import load_engine
    from phonon_core.mp_import import gamma_nac_directions
    res = converted_nac(label)
    eng = load_engine(res["phonopy_yaml"])          # BORN AUTO-DISCOVERED
    assert eng.nac_params is not None               # the switch really flipped
    qs, fr, _b = load_band(label)
    on_g = np.all(np.abs(qs) < 1e-12, axis=1)
    big = np.abs(fr) > NU_FLOOR

    fe = a1_frequencies(eng, qs)
    d_off = float(np.abs(fe - fr)[~on_g[:, None] & big].max())
    d_g_open = float(np.abs(fe - fr)[on_g[:, None] & big].max())

    # the Γ closure, per the pinned adjacent-segment direction rule
    d_g_closed = 0.0
    for g, direction in gamma_nac_directions(qs):
        eng.run_qpoints([qs[g]], nac_q_direction=direction)
        feg = np.sort(np.asarray(
            eng.get_qpoints_dict()["frequencies"], float), axis=1)[0]
        bigg = np.abs(fr[g]) > NU_FLOOR
        d_g_closed = max(d_g_closed, float(np.abs(feg - fr[g])[bigg].max()))

    print(f"\n[mp-nac] {label}: off-Γ max|dν| = {d_off:.3e} THz (gate "
          f"{FREQ_GATE}); Γ rows WITHOUT direction = {d_g_open:.6e} "
          f"(0.6.89 target {TARGETS[label]:.6e}) -> WITH adjacent-segment "
          f"direction = {d_g_closed:.3e} THz — TARGET CLOSED")
    assert d_off < FREQ_GATE
    # fail-then-pass: the target really was open at the printed magnitude...
    assert abs(d_g_open - TARGETS[label]) < 5e-3 * max(1.0, TARGETS[label])
    with pytest.raises(AssertionError):
        assert d_g_open < GAMMA_GATE
    # ...and the direction rule closes it
    assert d_g_closed < GAMMA_GATE


def test_negctl_phonopy_default_factor_bites(converted_nac, tmp_path):
    """The PINNED constant is load-bearing: a BORN file with phonopy's own
    VASP-default NAC constant (14.399652) FAILS the oracle on both cubic
    polar materials (measured 4.4e-02 / 3.9e-01 THz — above the 1e-2 gate),
    while the pinned constant passes at machine precision."""
    from phonon_core.engine import load_engine
    from phonon_core.mp_import import write_phonopy_born
    from conftest import load_meta
    for label, expect in (("gaas_mp-2534", 0.04), ("gan_mp-830", 0.3)):
        res = converted_nac(label)
        eng0 = load_engine(res["phonopy_yaml"])      # symmetry source only
        m = load_meta(label)
        bad_born = str(tmp_path / f"BORN_default_{label}")
        write_phonopy_born(eng0, m["born"], m["epsilon_static"], bad_born,
                           factor=14.399652)
        eng_bad = load_engine(res["phonopy_yaml"], born=bad_born)
        qs, fr, _b = load_band(label)
        on_g = np.all(np.abs(qs) < 1e-12, axis=1)
        big = np.abs(fr) > NU_FLOOR
        fe = a1_frequencies(eng_bad, qs)
        d = float(np.abs(fe - fr)[~on_g[:, None] & big].max())
        print(f"\n[mp-nac negctl] {label}: phonopy-default factor 14.399652 "
              f"-> off-Γ max|dν| = {d:.3e} THz (BITES; pinned constant gives "
              f"< 1.2e-05)")
        assert d > expect                            # the wrong constant is LOUD
        with pytest.raises(AssertionError):          # fail-then-pass
            assert d < FREQ_GATE


def test_a2_refuses_nac_output(converted_nac):
    """The vendored A2 engine REFUSES a converted-with-NAC directory (its
    0.6.86 BORN refusal, now exercised by a REAL polar material): the sibling
    BORN that switches A1's correction ON makes A2 fail LOUDLY rather than
    silently produce NAC-less numbers that disagree with A1 by ~0.5 THz."""
    from phonon_core.engine.dq_numpy import load_engine_np
    res = converted_nac("gaas_mp-2534")
    with pytest.raises(RuntimeError, match="(?i)born|non-analytic"):
        load_engine_np(res["phonopy_yaml"])
