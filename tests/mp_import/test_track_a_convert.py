"""mp_import Track A — the CONVERTER round-trip battery (0.6.89).

The first time any MP force-constant set drives the A1 engine. The oracle is
self-contained: MP ships BOTH the force constants AND its own band
frequencies; the converter is proven by reproducing those frequencies from
those force constants.

GATES, per the Option-A ruling (architect, 0.6.89):
- silicon (non-polar): BARE-FC reproduction < 1e-2 THz (measured 5.1e-11);
- GaAs/GaN (polar): MP's stored bands CONTAIN the non-analytic correction
  (LO-TO split at Gamma) despite ``has_nac: false`` — the bare-FC
  reproduction must FAIL > 0.1 THz (AMENDMENT 1, the NAC gate's own negative
  control), and the NAC-applied reproduction (A1's pre-existing BORN
  pass-through + the PINNED pheasy convention) must pass < 1e-2 THz;
- GaN (anisotropy class, AMENDMENT 2): the gate applies OFF-Gamma; the
  Gamma-row residual is PRINTED as the 0.6.91 acceptance target, not gated.
"""
import os
import re

import numpy as np
import pytest

from conftest import (
    CONSISTENT,
    DUAL_TRACK,
    a1_frequencies,
    fixture_dir,
    load_band,
    load_meta,
    requires_fixtures,
    write_born_file,
)
from engine_metric import NU_FLOOR

pytestmark = requires_fixtures

FREQ_GATE = 1e-2       # THz — task-pinned oracle gate (|nu| > NU_FLOOR)
BARE_FAIL = 0.1        # THz — AMENDMENT 1: bare FC must miss polar bands by this
HERM_TOL = 1e-10


def _doctrine_max(f_ref, f_eng):
    """max|dnu| over |nu| > NU_FLOOR (the near-zero half is checked via the
    stability battery's |dlambda| comparisons)."""
    big = np.abs(f_ref) > NU_FLOOR
    return float(np.abs(f_eng - f_ref)[big].max())


# ---------------------------------------------------------------------------
# purity: the src module must never grow MP-env dependencies
# ---------------------------------------------------------------------------
def test_purity_no_forbidden_imports():
    """mp_import imports numpy + phonopy + stdlib json/os ONLY — no pymatgen,
    no mp-api, no emmet, no requests (the harvest script owns those, in the
    user's MP environment). Grep the SOURCE TEXT, comments included."""
    import phonon_core.mp_import as mod
    pkg_dir = os.path.dirname(mod.__file__)
    forbidden = re.compile(r"^\s*(import|from)\s+(pymatgen|mp_api|emmet|requests)",
                           re.M)
    for fn in sorted(os.listdir(pkg_dir)):
        if fn.endswith(".py"):
            src = open(os.path.join(pkg_dir, fn)).read()
            hits = forbidden.findall(src)
            assert not hits, f"{fn}: forbidden import(s) {hits}"


# ---------------------------------------------------------------------------
# conversion + preconditions + round-trip through THE loader A1 uses
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label", CONSISTENT)
def test_convert_roundtrip_and_atom_identity(label, converted):
    """The triple parses through the SAME loader the engine and exporter call
    (engine.load_engine -> phonopy.load), the atom-count identity holds, and
    the NAC sidecar is wired OFF and not named BORN."""
    from phonon_core.engine import load_engine
    res = converted(label)
    info = res["info"]
    assert info["n_supercell_atoms"] == (info["n_unitcell_atoms"]
                                         * info["det_supercell_matrix"])
    eng = load_engine(res["phonopy_yaml"])           # A1's own loader
    assert len(eng.supercell) == info["n_supercell_atoms"]
    assert len(eng.primitive) == info["n_primitive_atoms"]
    assert os.path.basename(res["nac_sidecar"]) != "BORN"    # the trap
    import json
    side = json.load(open(res["nac_sidecar"]))
    assert side["nac_applied"] is False
    assert side["born"] is not None                  # carried, not applied
    if res["band_conf"] is not None:
        conf = open(res["band_conf"]).read()
        assert "BAND =" in conf and "FORCE_CONSTANTS = READ" in conf
    print(f"\n[mp-convert] {label}: {info['n_unitcell_atoms']} unit atoms x "
          f"det {info['det_supercell_matrix']} = {info['n_supercell_atoms']} "
          f"supercell atoms; {info['n_primitive_atoms']} primitive; "
          f"round-trip OK; MP has_nac flag = {info['mp_has_nac_flag']}")


def test_nacl_band_count_precondition_bites(tmp_path):
    """AMENDMENT 4 — nacl_mp-1120767 is the REAL malformed-input case: its MP
    band document stores 12 bands for a 12-site structure (36 required), same
    task id in both documents. The band-count precondition must refuse it
    LOUDLY, by name, with the actual counts."""
    from phonon_core.mp_import import MPImportError, convert_pheasy
    with pytest.raises(MPImportError, match=r"band-count identity FAILED"):
        convert_pheasy(fixture_dir("nacl_mp-1120767"), str(tmp_path))
    try:
        convert_pheasy(fixture_dir("nacl_mp-1120767"), str(tmp_path))
    except MPImportError as exc:
        msg = str(exc)
        assert "12" in msg and "36" in msg, msg      # the actual counts
        print(f"\n[mp-precondition] nacl_mp-1120767 refused: {msg}")


# ---------------------------------------------------------------------------
# THE ORACLE — reproduce MP's own stored bands from MP's own force constants
# ---------------------------------------------------------------------------
def test_oracle_bare_silicon(converted):
    """Silicon (non-polar, |Z*| = 0.000): the bare-FC reproduction — FC
    ingestion + atom order + supercell map + D(q), validated at once."""
    from phonon_core.engine import load_engine
    res = converted("silicon_mp-149")
    eng = load_engine(res["phonopy_yaml"])
    qs, fr, _b = load_band("silicon_mp-149")
    fe = a1_frequencies(eng, qs)
    dmax = _doctrine_max(fr, fe)
    print(f"\n[mp-oracle] silicon_mp-149 BARE FC over {len(qs)} q x "
          f"{fr.shape[1]} bands: max|dnu| (|nu|>{NU_FLOOR}) = {dmax:.3e} THz "
          f"(gate {FREQ_GATE})")
    assert dmax < FREQ_GATE


@pytest.mark.parametrize("label", ["gaas_mp-2534", "gan_mp-830"])
def test_oracle_polar_nac_applied(label, converted, tmp_path):
    """GaAs/GaN: MP's stored bands CONTAIN NAC (has_nac lies).

    AMENDMENT 1 (negative control, wired first): the BARE-FC reproduction
    must FAIL by > 0.1 THz — a NAC gate that also passes on bare FC tests
    nothing (measured: GaAs 0.569, GaN 5.708).
    THEN: with the PINNED pheasy convention (BORN pass-through file carrying
    MP born + eps_infinity + factor 15.633302300230191), the reproduction
    passes < 1e-2 THz OFF-Gamma; exact-Gamma rows are REPORTED (AMENDMENT 2),
    not gated — they need the anisotropic nac_q_direction, a 0.6.91 concern.
    """
    from phonon_core.engine import load_engine
    res = converted(label)
    qs, fr, _b = load_band(label)
    on_g = np.all(np.abs(qs) < 1e-12, axis=1)

    # (a) AMENDMENT 1: bare FC must demonstrably MISS the stored bands
    eng_bare = load_engine(res["phonopy_yaml"])
    d_bare = _doctrine_max(fr, a1_frequencies(eng_bare, qs))
    assert d_bare > BARE_FAIL, (
        f"bare-FC reproduction unexpectedly PASSES ({d_bare:.3e} THz) — "
        f"the NAC gate below would then test nothing")

    # (b) NAC via A1's PRE-EXISTING BORN pass-through (load_engine born=...,
    #     unchanged since 0.6.78) with the pinned convention
    born_path = write_born_file(str(tmp_path / f"BORN_{label}"), label, eng_bare)
    eng_nac = load_engine(res["phonopy_yaml"], born=born_path)
    fe = a1_frequencies(eng_nac, qs)
    big = np.abs(fr) > NU_FLOOR
    d_off = float(np.abs(fe - fr)[~on_g[:, None] & big].max())
    d_gam = float(np.abs(fe - fr)[on_g[:, None] & big].max()) if on_g.any() else 0.0
    print(f"\n[mp-oracle] {label}: BARE max|dnu| = {d_bare:.3f} THz (must fail "
          f"> {BARE_FAIL}: BITES); NAC-applied OFF-Gamma = {d_off:.3e} THz "
          f"(gate {FREQ_GATE})")
    print(f"[mp-oracle] {label}: 0.6.91 target: needs anisotropic "
          f"nac_q_direction at Gamma, current residual = {d_gam:.6e} THz "
          f"({int(on_g.sum())} Gamma rows, isotropic pinned convention)")
    assert d_off < FREQ_GATE


# ---------------------------------------------------------------------------
# D(q) sanity on the converted material
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label", DUAL_TRACK)
def test_hermiticity_and_asr(label, converted):
    """D(q) Hermitian at a few generic q; ASR at Gamma at the fixture's own
    level (sum_rules_breaking is None on every fixture — MP reports no
    residual; the measured Gamma dust is the honest number, bounded loosely
    at NU_FLOOR rather than asserted cleaner than MP's own data)."""
    from phonon_core.engine import dynamical_matrix_at, evaluate_dq, load_engine
    res = converted(label)
    eng = load_engine(res["phonopy_yaml"])
    for q in ((0.123, 0.267, 0.051), (0.4, 0.1, 0.3), (0.5, 0.0, 0.5)):
        D = dynamical_matrix_at(eng, q)
        herm = float(np.abs(D - D.conj().T).max())
        assert herm < HERM_TOL, f"q={q}: ||D - D^H|| = {herm:.2e}"
    f_g, _ = evaluate_dq(eng, (0.0, 0.0, 0.0))
    dust = np.sort(np.abs(f_g))[:3]
    srb = load_meta(label).get("sum_rules_breaking")
    print(f"\n[mp-sanity] {label}: Gamma acoustic |nu| = {dust} THz "
          f"(MP sum_rules_breaking = {srb})")
    assert dust.max() < NU_FLOOR
