"""mp_import Track A — the ATOM-ORDER negative control (0.6.89).

MP's FC is on MP/pheasy's supercell ordering; phonopy builds its OWN
supercell ordering from unitcell + supercell_matrix. If they differed, the
written FORCE_CONSTANTS would be in the wrong order and D(q) silently wrong.

CONVENTION FINDING (recorded in CLAUDE.md): they DO NOT differ — MP/pheasy's
supercell atom order IS phonopy's own (the identity permutation). The
converter applies no coord-matching permutation, and the machine-precision
oracle reproductions (silicon 5.1e-11 THz) prove none is needed. This test
proves the oracle would CATCH a mismatch: a single swapped supercell-atom
pair moves the reproduction by ~0.5 THz. A reproduction that survives a
permuted FC would be testing nothing.
"""
import numpy as np
import pytest

from conftest import a1_frequencies, load_band, requires_fixtures
from engine_metric import NU_FLOOR

pytestmark = requires_fixtures

FREQ_GATE = 1e-2


def test_permuted_fc_bites(converted, tmp_path):
    """Silicon (the bare-FC-passing material): re-write FORCE_CONSTANTS with
    two supercell atoms swapped -> the oracle reproduction must FAIL loudly,
    while the un-permuted file passes — fail-then-pass in one test."""
    from phonopy.file_IO import parse_FORCE_CONSTANTS, write_FORCE_CONSTANTS
    from phonon_core.engine import load_engine

    res = converted("silicon_mp-149")
    qs, fr, _b = load_band("silicon_mp-149")
    qs, fr = qs[::10], fr[::10]                      # 101 q suffice; teeth, not survey
    big = np.abs(fr) > NU_FLOOR

    # (a) the un-permuted triple passes (identity permutation is correct)
    eng = load_engine(res["phonopy_yaml"])
    d_ok = float(np.abs(a1_frequencies(eng, qs) - fr)[big].max())
    assert d_ok < FREQ_GATE

    # (b) swap two images of atom 0 in the supercell index order -> BITES.
    #     (A permutation WITHIN one unit atom's images is the subtlest case:
    #     invisible at Gamma — every phase there is 1 — and structurally the
    #     same class of error as the 0.6.87 tie bug's q-dependent signature.)
    fc = parse_FORCE_CONSTANTS(res["force_constants"])
    assert fc.shape[0] == fc.shape[1] == 128         # full-format guard
    perm = np.arange(fc.shape[0])
    perm[[1, 5]] = perm[[5, 1]]
    bad_path = str(tmp_path / "FORCE_CONSTANTS_permuted")
    write_FORCE_CONSTANTS(fc[np.ix_(perm, perm)], filename=bad_path)
    eng_bad = load_engine(res["phonopy_yaml"], force_constants=bad_path)
    fe_bad = a1_frequencies(eng_bad, qs)
    d_bad = float(np.abs(fe_bad - fr)[big].max())
    # ... and the Gamma invisibility that makes ordering dangerous is pinned:
    g = np.where(np.all(np.abs(qs) < 1e-12, axis=1))[0]
    d_gam = float(np.abs(fe_bad[g] - fr[g])[np.abs(fr[g]) > NU_FLOOR].max()) \
        if len(g) else None
    print(f"\n[mp-negctl] silicon: un-permuted max|dnu| = {d_ok:.3e} THz "
          f"(passes); one swapped pair = {d_bad:.3f} THz (BITES)"
          + (f"; at Gamma the permutation is INVISIBLE ({d_gam:.3e} THz)"
             if d_gam is not None else ""))
    assert d_bad > 0.1                               # >> the 1e-2 gate
    with pytest.raises(AssertionError):              # fail-then-pass, in-suite
        assert d_bad < FREQ_GATE
    if d_gam is not None:
        assert d_gam < 1e-6                          # the trap that hides it
