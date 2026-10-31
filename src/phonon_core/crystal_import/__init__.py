"""CRYSTAL23 provenance: phonon band structures from fort.25 + the .out.

What the format carries — the finding this module is built on
-------------------------------------------------------------
CRYSTAL's ``fort.25`` (``.f25``) BAND blocks carry path FREQUENCIES ONLY:
no eigenvectors, no dynamical matrix. The ``.out`` of a SCELPHONO/FREQCALC
dispersion run adds the geometry, the path definition (integer segment
endpoints over a denominator, in PRIMITIVE fractional units), the isotopic
masses actually used, and per-k frequency lists — but still no off-Γ
eigenvectors. ``FREQINFO.DAT`` is a restart holding symmetry-IRREDUCIBLE
gradient sets, not a directly parseable Hessian.

Consequently this importer produces HONEST UNWEIGHTED bands: the payload's
eigenvector arrays are zero and ``meta.conventions`` says so. Atom-weighted
bands, mode animation and 3D surfaces need the supercell Hessian — forward
work, reachable either by decoding FREQINFO's symmetry expansion or by
re-running CRYSTAL with a Hessian-writing option; once a Hessian exists,
``mlip_import.convert_hessian`` already provides the rest of the pipeline.

The zero-eigenvector fallback is deliberate viewer behaviour: the 2D blend
falls back to a uniform species mix (``tot > 0`` guard in the viewer), and
the animation shows zero displacement rather than invented motion.
"""

from phonon_core.crystal_import.fort25 import (  # noqa: F401
    convert_crystal_bands, read_crystal_out_structure, read_fort25,
)

__all__ = ["read_fort25", "read_crystal_out_structure", "convert_crystal_bands"]
