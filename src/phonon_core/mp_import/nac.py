"""NAC/LO-TO activation for MP-imported materials (0.6.91).

Closes the arc opened at 0.6.89: MP's stored pheasy bands for polar materials
CONTAIN the non-analytic correction (while ``has_nac: false`` lies), computed
with a convention this module pins:

- ``dielectric`` = the metadata's ``epsilon_static`` — which actually holds
  the ION-CLAMPED high-frequency dielectric (eps_infinity);
- ``born`` = the metadata's per-site Born effective charges;
- ``factor`` = :data:`PHEASY_NAC_FACTOR` = **15.633302300230191** — the
  VaspToTHz constant, NOT phonopy's VASP-default NAC constant 14.399652.

DERIVATION STATUS (owed at 0.6.91, settled as far as the sandbox allows):
phonopy's API carries TWO parameters both named ``factor`` — the FREQUENCY
conversion (``Phonopy(factor=VaspToTHz)``) and the NAC unit constant
(``nac_params["factor"]``, Hartree*Bohr = 14.399652 for VASP). A pipeline
that passes the frequency factor into ``nac_params`` reproduces our
measurement EXACTLY. That naming collision is the likely origin (hypothesis —
the pheasy/atomate2 source is not in this sandbox to confirm); the empirical
pin is now DOUBLE-validated: off-Gamma reproduction at 5.0e-11..1.2e-05 THz
on four polar materials (0.6.89) and the directional Gamma closure below
(0.6.91, ~5e-11 THz on all 12 Gamma rows). The negative control proves the
constant is load-bearing: phonopy's own default fails the oracle by
4.4e-02..3.9e-01 THz.

GAMMA DIRECTION CONVENTION (pinned here, machine precision on all 12 Gamma
rows of all four polar fixtures): at exactly q = Gamma the non-analytic term
needs a q -> 0 DIRECTION; MP's stored Gamma rows were evaluated along the
ADJACENT PATH-SEGMENT direction — toward the NEXT path point where Gamma
opens a segment, from the PREVIOUS point where it closes one. Isotropic for
cubic cells (GaAs/GaN: any direction reproduces), decisive for anisotropic
ones (CsSnI3 8.2e-01 -> 4.8e-11; MgO 4.3e+00 -> 5.0e-11 THz).

PURITY: numpy + (lazy) phonopy + stdlib — same contract as pheasy.py.
"""
import numpy as np

#: MP/pheasy's NAC unit constant — the VaspToTHz value, EMPIRICALLY PINNED
#: (0.6.89) and double-validated (0.6.91). Do NOT substitute phonopy's
#: VASP-default 14.399652: the negative control proves it fails the oracle.
PHEASY_NAC_FACTOR = 15.633302300230191


def write_phonopy_born(ph, born, dielectric, path,
                       factor=PHEASY_NAC_FACTOR):
    """Write a phonopy ``BORN`` file activating NAC for ``ph``'s material.

    ``ph`` is the (already-built) ``Phonopy`` object of the converted
    material; ``born`` is the MP per-unit-cell-site array ``(N, 3, 3)``;
    ``dielectric`` the 3x3 eps_infinity. phonopy's BORN format wants tensors
    for the symmetry-INDEPENDENT atoms of the primitive only (its parser
    site-symmetrizes and expands); MP's per-site array maps 1:1 onto
    primitive atoms because every fixture's ``primitive_matrix`` is a det-1
    unimodular basis change — asserted, not assumed.

    NOTE this file is the DELIBERATE activation switch: A1's ``load_engine``
    auto-discovers a sibling named ``BORN`` (the 0.6.89 filename-trap ledger
    entry, now used on purpose), and A2's ``load_engine_np`` REFUSES the same
    sibling loudly — the vendored engine cannot silently produce NAC-less
    numbers where A1 would produce corrected ones.
    """
    from phonon_core.mp_import.pheasy import MPImportError

    born = np.asarray(born, dtype=np.float64)
    dielectric = np.asarray(dielectric, dtype=np.float64)
    n_prim = len(ph.primitive)
    if born.shape != (n_prim, 3, 3):
        raise MPImportError(
            f"BORN writer: born array has shape {born.shape}, expected "
            f"({n_prim}, 3, 3) — the 1:1 site->primitive mapping requires a "
            f"det-1 primitive_matrix (n_prim == n_sites)")
    if dielectric.shape != (3, 3):
        raise MPImportError(
            f"BORN writer: dielectric has shape {dielectric.shape}, "
            f"expected (3, 3)")
    indep = ph.primitive_symmetry.get_independent_atoms()
    with open(path, "w") as fh:
        fh.write(f"{factor:.15f}\n")
        fh.write(" ".join(f"{x:.12f}" for x in dielectric.reshape(-1)) + "\n")
        for a in indep:
            fh.write(" ".join(f"{x:.12f}" for x in born[a].reshape(-1)) + "\n")
    return path


def gamma_nac_directions(qpoints, atol=1e-12):
    """Per-Gamma-row q->0 directions under the PINNED adjacent-segment rule.

    Returns ``[(row_index, direction (3,)), ...]`` for every row of
    ``qpoints`` that is exactly Gamma: the direction toward the NEXT path
    point when it is non-Gamma (Gamma opens a segment), else from the
    PREVIOUS point (Gamma closes one). Evaluate each row separately with
    ``Phonopy.run_qpoints([q_gamma], nac_q_direction=direction)`` — phonopy
    applies the direction only in the q -> 0 limit.
    """
    qs = np.asarray(qpoints, dtype=np.float64)
    on_g = np.all(np.abs(qs) < atol, axis=1)
    out = []
    for g in np.where(on_g)[0]:
        if g + 1 < len(qs) and not on_g[g + 1]:
            d = qs[g + 1] - qs[g]
        elif g - 1 >= 0 and not on_g[g - 1]:
            d = qs[g] - qs[g - 1]
        else:
            d = np.array([1.0, 0.0, 0.0])       # isolated Gamma: no adjacency
        out.append((int(g), d))
    return out
