"""phonon_core.engine — compute engines (optional; Level-2 arc).

Stage 1 (0.6.78): the A1 dynamical-matrix engine via phonopy's API
(:mod:`phonon_core.engine.dq_phonopy`) — evaluates D(q) at arbitrary q and
returns frequencies + eigenvectors in the tool's parser layout. Requires the
``engine`` optional extra (``pip install phonon-core[engine]``); importing
this package works without it (phonopy loads lazily at call time).

Stage 4 (0.6.86): the A2 pure-numpy engine
(:mod:`phonon_core.engine.dq_numpy`) — the same contract plus a batched API,
ZERO phonopy dependency (the module the Blender add-on will vendor). A1 is
its ORACLE: A2 ships only because it reproduces A1 everywhere tested
(tests/test_engine_dq_numpy.py). No NAC: a sibling BORN file is a hard
refusal — use A1 for polar materials.

The exporter still drives A1 exclusively (an engine switch is a separate
concern, if ever); json_v1, the viewer and kinematics.js are untouched.
"""
from phonon_core.engine.dq_phonopy import (
    dynamical_matrix_at,
    evaluate_dq,
    load_engine,
)
from phonon_core.engine.dq_numpy import (
    NumpyEngine,
    dynamical_matrix_at_np,
    evaluate_dq_np,
    evaluate_dq_np_batch,
    load_engine_np,
)

__all__ = [
    "load_engine", "evaluate_dq", "dynamical_matrix_at",
    "load_engine_np", "evaluate_dq_np", "evaluate_dq_np_batch",
    "dynamical_matrix_at_np", "NumpyEngine",
]
