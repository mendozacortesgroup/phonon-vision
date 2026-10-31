"""The vendored phonon_core wheel subset — ONE list, TWO consumers.

`tools/build_vendored_wheel.py` ASSEMBLES the wheel from exactly this list;
`tools/check_vendored_parity.py` VERIFIES the shipped wheel against it (every
module byte-identical to the core src, nothing extra, nothing missing). One
constant, no drift — the 0.7.14 fix for the pre-0.2.0 situation where the
subset existed only implicitly, inside a hand-built wheel.

STUB_PINS: the two files that legitimately DIVERGE from the core src — the
packaging stubs, written from templates in build_vendored_wheel.py and pinned
here by md5. When a stub template changes (new wheel version, subset change),
re-run the build and update the pins IN THE SAME COMMIT; the parity guard
fails otherwise, by design. History: the single 0.1.4 stub pin 07478e09 is
RETIRED (one stub then; two now that the wheel carries the A2 engine
subpackage).
"""

WHEEL_VERSION = "0.2.1"

#: wheel-relative module paths under phonon_core/, byte-compared against
#: --core-src/src/phonon_core/<same path> by the parity guard.
SUBSET_MODULES = [
    "animation.py",
    "expand.py",
    "kinematics.py",
    "projection.py",
    "radii.py",
    "species.py",
    "types.py",
    "units.py",
    "parsers/__init__.py",
    "parsers/_material.py",
    "parsers/_phonopy_yaml.py",
    "parsers/_yaml_loader.py",
    "parsers/band_hdf5.py",
    "parsers/band_yaml.py",
    "parsers/mesh_hdf5.py",
    "parsers/mesh_yaml.py",
    "engine/dq_numpy.py",          # A2 pure-numpy D(q) engine (core 0.6.91: absolute-symprec tie fix)
]

#: the templated packaging stubs (md5 of the bytes build_vendored_wheel.py
#: writes; printed by every build run). The parity guard consumes them.
STUB_PINS = {
    "__init__.py": "35f9ac1b76e1c81fd6cb2b0815cf24f8",
    "engine/__init__.py": "4bdbbc5662bb74a326c1951e430e505f",
}
