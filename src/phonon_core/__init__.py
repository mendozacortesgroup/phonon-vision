"""phonon-core: physics-accurate phonon data layer.

A small, well-tested Python library that parses phonopy outputs (band.yaml,
mesh.yaml, mesh.hdf5) into typed dataclasses and provides the kinematics for
animating modes correctly. Designed as the shared foundation for both a
Blender renderer and a web visualization tool — same math, two front-ends.

Quick start
-----------
>>> from phonon_core import load_band_yaml
>>> band = load_band_yaml("band.yaml")
>>> mode = band.get_mode(q_index=100, band_index=3)
>>> from phonon_core.kinematics import compute_displacements_at_phase
>>> u = compute_displacements_at_phase(mode, band.material, amplitude=2.0, phase=0.0)
"""

__version__ = "3.6.0"

from phonon_core.types import (
    Material,
    PhononMode,
    BandPath,
    MeshGrid,
)
from phonon_core.parsers import (
    load_band_yaml,
    load_mesh_yaml,
    load_band_hdf5,
    load_mesh_hdf5,
    parse_phonopy_yaml_material,
)
from phonon_core import (kinematics, animation, species, projection, units,
                          radii, expand, io, conformance, unfolding, bz)
from phonon_core.expand import expand_cell, ExpandedAtoms

__all__ = [
    "__version__",
    "Material",
    "PhononMode",
    "BandPath",
    "MeshGrid",
    "load_band_yaml",
    "load_mesh_yaml",
    "load_band_hdf5",
    "load_mesh_hdf5",
    "parse_phonopy_yaml_material",
    "kinematics",
    "animation",
    "species",
    "projection",
    "units",
    "radii",
    "expand",
    "io",
    "conformance",
    "expand_cell",
    "ExpandedAtoms",
]
