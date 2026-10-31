"""MLIP provenance: bring machine-learned-potential Hessians onto the
common ground.

The typical MLIP phonon workflow already ends in phonopy format (phonopy
displacements → MLIP forces → FORCE_SETS), and needs nothing from us. This
package covers the OTHER route: methods that produce the full supercell
Hessian directly (autograd / Hessian-vector products / finite differences
over an ASE calculator), which is not a phonopy file. `convert_hessian`
turns that Hessian into the phonopy triple the rest of phonon-vision
consumes.
"""

from phonon_core.mlip_import.hessian import convert_hessian  # noqa: F401

__all__ = ["convert_hessian"]
