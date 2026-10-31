#!/usr/bin/env python3
"""Build the vendored phonon_core wheel — the REPRODUCIBLE recipe (0.7.14).

Until 0.2.0 the vendored wheel was built BY HAND (0.1.4 had no committed
recipe). This script is the recipe: it assembles the wheel from the SHARED
subset list (tools/_subset.py — the same constant the parity guard verifies),
writes the two packaging stubs from templates BELOW, builds, prints the
inner-file md5 manifest, and SMOKES the built wheel in a subprocess whose
sys.path front-loads a shadow `phonopy` module that raises ImportError —
proving the A2 engine works with phonopy strictly absent (the whole point of
vendoring it).

    python tools/build_vendored_wheel.py --core-src ../phonon-core [--out wheels/]
    python tools/build_vendored_wheel.py --smoke wheels/phonon_core-0.2.0-py3-none-any.whl

The inner-file md5 manifest is the reproducibility contract: the wheel ZIP's
own md5 varies with build timestamps; the CONTENTS must not. Every subset
module is a byte-copy of --core-src/src/phonon_core/<path>, so the parity
guard (17 modules + 2 pinned stubs) passes on the result by construction.

Stub policy: the two __init__.py files below are the ONLY bytes not copied
from the core src. The top-level stub carries the wheel's own __version__
(the public wheel line) and the subset imports; the engine stub exposes the
A2 names ONLY (the phonopy-backed A1 engine is deliberately absent from the
wheel — a vendored A1 could never import inside Blender, and a present-but-
broken module is worse than an absent one). When a template changes, update
STUB_PINS in tools/_subset.py IN THE SAME COMMIT (the parity guard fails
otherwise, by design).

Build backend: tries `python -m build --wheel` (PEP 517), falls back to
`pip wheel . --no-deps` — both produce identical package contents (the
manifest print is the proof that matters).
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import zipfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from _subset import SUBSET_MODULES, WHEEL_VERSION   # noqa: E402  (shared list)

# --------------------------------------------------------------------------
# The two packaging-stub templates (the only non-core bytes in the wheel)
# --------------------------------------------------------------------------
TOP_STUB = '''"""phonon-core: physics-accurate phonon data layer.

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

__version__ = "{version}"

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
from phonon_core import kinematics, animation, species, projection, units, radii, expand
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
    "expand_cell",
    "ExpandedAtoms",
]
'''.format(version=WHEEL_VERSION)

ENGINE_STUB = '''"""phonon_core.engine — vendored A2 subset (pure-numpy D(q)).

The phonopy-backed A1 engine (`dq_phonopy`) is DELIBERATELY ABSENT from this
wheel: phonopy cannot be a dependency inside Blender, and a module that can
never import is worse than one that does not exist. Use `load_engine_np` —
the A2 engine, oracle-validated against A1 at build time in the core tree
(phonon-core 0.6.86, tests/test_engine_dq_numpy.py).
"""
from .dq_numpy import load_engine_np, evaluate_dq_np, evaluate_dq_np_batch

__all__ = ["load_engine_np", "evaluate_dq_np", "evaluate_dq_np_batch"]
'''

PYPROJECT = f'''[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "phonon_core"
version = "{WHEEL_VERSION}"
description = "Vendored phonon-core subset for the Phonon Vision Blender add-on"
dependencies = ["numpy>=1.20"]

[tool.setuptools.packages.find]
include = ["phonon_core*"]
'''


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _stage(core_src, stage):
    """Copy the subset + write the stubs; return the package dir."""
    core_pkg = os.path.join(core_src, "src", "phonon_core")
    if not os.path.isdir(core_pkg):
        sys.exit(f"[build] not a core src tree (no src/phonon_core): {core_src}")
    pkg = os.path.join(stage, "phonon_core")
    for rel in SUBSET_MODULES:
        src = os.path.join(core_pkg, rel)
        if not os.path.isfile(src):
            sys.exit(f"[build] subset module missing in core src: {rel}")
        dst = os.path.join(pkg, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    with open(os.path.join(pkg, "__init__.py"), "w", encoding="utf-8") as f:
        f.write(TOP_STUB)
    with open(os.path.join(pkg, "engine", "__init__.py"), "w",
              encoding="utf-8") as f:
        f.write(ENGINE_STUB)
    with open(os.path.join(stage, "pyproject.toml"), "w",
              encoding="utf-8") as f:
        f.write(PYPROJECT)
    return pkg


def _build(stage, out_dir):
    """PEP-517 build with a pip-wheel fallback; return the wheel path."""
    env = dict(os.environ)
    try:
        subprocess.run([sys.executable, "-m", "build", "--wheel",
                        "--outdir", out_dir], cwd=stage, check=True, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[build] `python -m build` unavailable/failed — falling back "
              "to `pip wheel` (identical package contents)")
        subprocess.run([sys.executable, "-m", "pip", "wheel", ".",
                        "--no-deps", "-w", out_dir], cwd=stage, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    whl = os.path.join(out_dir, f"phonon_core-{WHEEL_VERSION}-py3-none-any.whl")
    if not os.path.isfile(whl):
        sys.exit(f"[build] expected wheel not produced: {whl}")
    return whl


def _manifest(whl):
    """Print the inner-file md5 manifest (the reproducibility contract)."""
    print(f"[build] wheel: {whl}")
    print("[build] inner-file md5 manifest (the reproducibility contract):")
    with zipfile.ZipFile(whl) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith(".py"):
                h = hashlib.md5(zf.read(name)).hexdigest()
                print(f"    {h[:8]}  {name}")
    print("[build] STUB md5s for tools/_subset.py STUB_PINS "
          "(update IN THE SAME COMMIT if changed):")
    with zipfile.ZipFile(whl) as zf:
        for rel in ("phonon_core/__init__.py", "phonon_core/engine/__init__.py"):
            h = hashlib.md5(zf.read(rel)).hexdigest()
            print(f"    \"{rel[len('phonon_core/'):]}\": \"{h}\",")


_SMOKE_DRIVER = r'''
import sys, os
import numpy as np
wheel, workdir = sys.argv[1], sys.argv[2]
sys.path.insert(0, workdir)     # the phonopy SHADOW sits here
sys.path.insert(0, wheel)       # zipimport the wheel under test
try:
    import phonopy              # must hit the shadow
    raise SystemExit("shadow phonopy did not fire")
except ImportError:
    pass
import phonon_core
from phonon_core.engine import load_engine_np, evaluate_dq_np
eng = load_engine_np(os.path.join(workdir, "phonopy.yaml"))
for q in ([0.0, 0.0, 0.0], [0.25, 0.1, 0.05], [0.5, 0.5, 0.0]):
    f, v = evaluate_dq_np(eng, q)
    assert np.all(np.isfinite(f)), q
    norms = np.linalg.norm(v.reshape(len(f), -1), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10), q
fg, _ = evaluate_dq_np(eng, [0.0, 0.0, 0.0])
assert np.max(np.abs(fg[:3])) < 1e-4, fg     # ASR toy: ~0 acoustic at Gamma
assert fg[3] > 0.1, fg                        # real optical gap
print(f"[smoke] A2-in-wheel OK phonon_core {phonon_core.__version__}: "
      f"phonopy ABSENT, engine evaluated 3 q, Gamma acoustic "
      f"{np.round(fg[:3], 6).tolist()} THz, first optical {fg[3]:.3f} THz")
'''


def _write_smoke_fixture(workdir):
    """A tiny 2-atom cubic toy: hand-written phonopy.yaml + FORCE_CONSTANTS
    (translationally consistent isotropic-spring FC with exact ASR — the
    Tier-1 pattern from core's tests/test_engine_dq_numpy.py)."""
    import numpy as np
    a = 3.1
    plat = a * np.eye(3)
    tau = np.array([[0.0, 0.0, 0.0], [0.4, 0.55, 0.35]])
    masses = [10.5, 27.3]
    n_rep = 2
    slat = n_rep * plat
    cells = np.array([[i, j, k] for i in range(n_rep)
                      for j in range(n_rep) for k in range(n_rep)], float)
    sfrac = np.concatenate([(tau[at] + cells) / n_rep for at in range(2)])
    n_cell = len(cells)
    n_s = 2 * n_cell
    r_s = sfrac @ slat

    # RIDER FIX (0.8.0, audit defect 0.7.14): numpy scalars interpolated
    # with !r emit "np.float64(3.1)" under numpy>=2 and poison the YAML —
    # EVERY interpolated scalar is converted via float() first, and the
    # generated text is asserted np.float64-free below.
    def _f(x):
        return repr(float(x))

    y = ["primitive_cell:", "  lattice:"]
    for row in plat:
        y.append(f"  - [{_f(row[0])}, {_f(row[1])}, {_f(row[2])}]")
    y.append("  points:")
    for k, sym in enumerate(("X", "Y")):
        y.append(f"  - symbol: {sym}")
        y.append(f"    coordinates: [{_f(tau[k][0])}, {_f(tau[k][1])}, {_f(tau[k][2])}]")
        y.append(f"    mass: {_f(masses[k])}")
    y.append("supercell:")
    y.append("  lattice:")
    for row in slat:
        y.append(f"  - [{_f(row[0])}, {_f(row[1])}, {_f(row[2])}]")
    y.append("  points:")
    for at in range(2):
        for c in range(n_cell):
            i = at * n_cell + c
            y.append(f"  - symbol: {'X' if at == 0 else 'Y'}")
            y.append(f"    coordinates: [{_f(sfrac[i][0])}, {_f(sfrac[i][1])}, "
                     f"{_f(sfrac[i][2])}]")
            y.append(f"    mass: {_f(masses[at])}")
            y.append(f"    reduced_to: {at * n_cell + 1}")
    yaml_text = "\n".join(y) + "\n"
    assert "np.float64" not in yaml_text, (
        "numpy scalar repr leaked into the generated YAML (the numpy>=2 "
        "!r trap) — the float() conversions above must cover every scalar")
    with open(os.path.join(workdir, "phonopy.yaml"), "w") as f:
        f.write(yaml_text)

    # minimum-image, tie-averaged isotropic-spring FC + exact ASR
    rng3 = np.arange(-1, 2)
    L = np.stack(np.meshgrid(rng3, rng3, rng3, indexing="ij"),
                 axis=-1).reshape(-1, 3) @ slat
    d = r_s[None, :, None, :] - r_s[:, None, None, :] + L[None, None, :, :]
    dist = np.linalg.norm(d, axis=-1)
    tie = dist <= dist.min(axis=-1)[..., None] * (1 + 1e-5) + 1e-12

    def g(v):
        r = np.linalg.norm(v)
        if r < 1e-12:
            return 1.7 * np.eye(3)
        vh = v / r
        return -(np.exp(-r / 2.0) * np.eye(3)
                 + 0.7 * np.exp(-r / 3.0) * np.outer(vh, vh))

    fc = np.zeros((n_s, n_s, 3, 3))
    for i in range(n_s):
        for j in range(n_s):
            fc[i, j] = np.mean([g(v) for v in d[i, j][tie[i, j]]], axis=0)
    for i in range(n_s):
        fc[i, i] -= fc[i].sum(axis=0)
    with open(os.path.join(workdir, "FORCE_CONSTANTS"), "w") as f:
        f.write(f"{n_s} {n_s}\n")
        for i in range(n_s):
            for j in range(n_s):
                f.write(f"{i + 1} {j + 1}\n")
                for row in fc[i, j]:
                    f.write("  ".join(f"{x:.15f}" for x in row) + "\n")


def smoke(wheel):
    """Run the phonopy-absent A2 smoke against `wheel` (exit 0 = pass)."""
    wheel = os.path.abspath(wheel)
    with tempfile.TemporaryDirectory() as workdir:
        with open(os.path.join(workdir, "phonopy.py"), "w") as f:
            f.write("raise ImportError('phonopy deliberately absent "
                    "(vendored-wheel smoke)')\n")
        _write_smoke_fixture(workdir)
        driver = os.path.join(workdir, "_driver.py")
        with open(driver, "w") as f:
            f.write(_SMOKE_DRIVER)
        r = subprocess.run([sys.executable, driver, wheel, workdir],
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        if r.returncode != 0:
            sys.exit(f"[smoke] FAILED against {os.path.basename(wheel)} "
                     f"(exit {r.returncode})")
        print(f"[smoke] PASS against {os.path.basename(wheel)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--core-src", default=None,
                    help="phonon_core source tree root (contains src/phonon_core)")
    ap.add_argument("--out", default=os.path.join(_ROOT, "wheels"),
                    help="output directory (default: wheels/)")
    ap.add_argument("--smoke", default=None, metavar="WHEEL",
                    help="run ONLY the phonopy-absent smoke against WHEEL")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the post-build smoke")
    args = ap.parse_args()

    if args.smoke:
        smoke(args.smoke)
        return
    if not args.core_src:
        ap.error("--core-src is required to build (or use --smoke WHEEL)")

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as stage:
        _stage(os.path.abspath(args.core_src), stage)
        whl = _build(stage, out_dir)
    _manifest(whl)
    if not args.no_verify:
        smoke(whl)


if __name__ == "__main__":
    main()
