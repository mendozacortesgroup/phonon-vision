"""Quantum ESPRESSO provenance.

Two routes, with different guarantees:

* :func:`read_qe_dyn` / :func:`read_qe_dyn_set` — ph.x ``.dyn`` dynamical
  matrices: EXACT at the computed q, no interpolation, and valid for polar
  materials today. Each file self-validates (it carries both D(q) and the
  frequencies ph.x derived from it).
* :func:`convert_qe_fc` — the q2r ``.fc`` real-space force constants, which
  interpolate to arbitrary q but are REFUSED for polar materials until the
  long-range dipole reconstruction is pinned. See ``fc_file``.
"""

from phonon_core.qe_import.fc_file import convert_qe_fc, read_qe_fc  # noqa: F401
from phonon_core.qe_import.dyn_file import (  # noqa: F401
    RY_THZ, QEDynamical, dyn_frequencies, dyn_modes, read_qe_dyn,
    read_qe_dyn_set,
)
from phonon_core.qe_import.dyn_payload import convert_qe_dyn  # noqa: F401

__all__ = ["convert_qe_fc", "read_qe_fc", "read_qe_dyn", "read_qe_dyn_set",
           "dyn_frequencies", "dyn_modes", "convert_qe_dyn", "QEDynamical",
           "RY_THZ"]
