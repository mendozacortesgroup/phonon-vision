#!/usr/bin/env python3
"""pv_export — phonopy files -> validated, conformance-checked JSON v1 web bundle.

Pure Python; runs anywhere (no Blender, no network). Builds ON the frozen
json_v1 layer and the 0.2.0 conformance harness; it consumes them, never
modifies them.

    pv_export.py INPUT [INPUT2 ...] -o OUT.json [options]

A band file and a mesh file of the SAME material may be given together; they are
merged into ONE document with both a band block and a mesh block.

Options
    --kind band|mesh|auto        auto-detect from phonopy file shape (default auto)
    --source-code NAME           provenance -> conventions block (default phonopy)
    --source-version VER
    --indent N                   pretty-print the JSON (default compact)
    --validate / --no-validate   JSON-Schema-validate the output (default validate)
    --conformance / --no-conformance
                                 round-trip-through-conformance gate (default on)
    --bundle DIR                 also emit a web bundle (document + golden vectors)

Exit codes: 0 ok; 2 usage / schema-validation failure; 3 conformance failure;
1 other error. On failure the message says which input, which mode kind, and —
for conformance — which tier failed.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "src"))

import numpy as np

from phonon_core import (
    load_band_yaml, load_band_hdf5, load_mesh_yaml, load_mesh_hdf5,
)
from phonon_core.io import (
    PhononDocument, write_json_v1, read_json_v1, roundtrip_conformance,
    write_web_bundle,
)
from phonon_core.io.json_v1 import build_document_dict


def _die(msg, code=1):
    print(f"pv_export: error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _is_hdf5(path):
    return path.lower().endswith((".hdf5", ".h5"))


def detect_kind(path):
    """band|mesh from a phonopy file's actual shape."""
    if _is_hdf5(path):
        import h5py
        with h5py.File(path, "r") as f:
            keys = set(f.keys())
        if "segment_nqpoint" in keys or "path" in keys:
            return "band"
        if "mesh" in keys and "qpoint" in keys:
            return "mesh"
        _die(f"{path}: cannot auto-detect kind from HDF5 keys {sorted(keys)}; "
             f"pass --kind", 2)
    # YAML: scan the header (cheap even for large files)
    with open(path, "r", encoding="utf-8") as fh:
        for _ in range(500):
            line = fh.readline()
            if not line:
                break
            if line.startswith("mesh:"):
                return "mesh"
            if line.startswith("segment_nqpoint:") or line.startswith("npath:"):
                return "band"
    _die(f"{path}: cannot auto-detect kind from YAML header; pass --kind", 2)


def _load(path, kind, *, material=None):
    if kind == "band":
        return load_band_hdf5(path) if _is_hdf5(path) else load_band_yaml(path)
    if _is_hdf5(path):
        return load_mesh_hdf5(path, material=material)
    return load_mesh_yaml(path)


def _material_diffs(a, b):
    diffs = []
    if list(a.symbols) != list(b.symbols):
        diffs.append(f"symbols {a.symbols} != {b.symbols}")
    if not np.allclose(a.lattice, b.lattice):
        diffs.append("lattice")
    if not np.allclose(a.masses, b.masses):
        diffs.append("masses")
    if a.fractional_coords.shape != b.fractional_coords.shape or \
            not np.allclose(a.fractional_coords, b.fractional_coords):
        diffs.append("fractional_coords")
    return diffs


def build_document(inputs, kinds, *, source_code, source_version):
    """Load each input, classify, and merge into one PhononDocument."""
    classified = []
    for p in inputs:
        if not os.path.isfile(p):
            _die(f"input not found: {p}", 2)
        k = kinds if kinds != "auto" else detect_kind(p)
        classified.append((p, k))

    bands = [(p, k) for p, k in classified if k == "band"]
    meshes = [(p, k) for p, k in classified if k == "mesh"]
    if len(bands) > 1:
        _die(f"more than one band input: {[p for p,_ in bands]}", 2)
    if len(meshes) > 1:
        _die(f"more than one mesh input: {[p for p,_ in meshes]}", 2)
    if not bands and not meshes:
        _die("no inputs to export", 2)

    band = mesh = None
    if bands:
        bp, _ = bands[0]
        try:
            band = _load(bp, "band")
        except Exception as e:
            _die(f"failed to load band {bp}: {e}")
    # mesh.hdf5 carries no material; feed it the band's material when merging.
    mat_for_mesh = band.material if band is not None else None
    if meshes:
        mp, _ = meshes[0]
        try:
            mesh = _load(mp, "mesh", material=mat_for_mesh)
        except Exception as e:
            _die(f"failed to load mesh {mp}: {e}")

    # one consistent Material across both blocks
    if band is not None and mesh is not None:
        diffs = _material_diffs(band.material, mesh.material)
        if diffs:
            _die("band and mesh describe different materials; cannot merge "
                 f"({', '.join(diffs)})", 2)
    shared = band.material if band is not None else mesh.material
    if band is not None:
        band.material = shared
    if mesh is not None:
        mesh.material = shared

    return PhononDocument(material=shared, band=band, mesh=mesh,
                          source_code=source_code, source_version=source_version)


def validate_schema(path):
    try:
        import jsonschema
    except ImportError:
        _die("jsonschema not installed; install it or pass --no-validate", 2)
    import json
    schema_path = os.path.join(_HERE, os.pardir, "docs", "json_v1_schema.json")
    with open(schema_path) as f:
        schema = json.load(f)
    with open(path) as f:
        instance = json.load(f)
    try:
        jsonschema.validate(instance=instance, schema=schema)
    except jsonschema.ValidationError as e:
        _die(f"schema validation FAILED: {e.message} (at {list(e.absolute_path)})", 2)


def run_conformance(document):
    res = roundtrip_conformance(document)
    if not res.passed:
        tier = res.first_failed_tier
        _die(f"conformance gate FAILED at tier '{tier.name}' "
             f"(max|Δ|={tier.max_abs_dev:.3e}); "
             f"covered finite-q modes: {res.has_finite_q}", 3)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pv_export", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="phonopy band/mesh yaml or hdf5")
    ap.add_argument("-o", "--output", required=True, help="output JSON v1 path")
    ap.add_argument("--kind", choices=["band", "mesh", "auto"], default="auto")
    ap.add_argument("--source-code", default="phonopy")
    ap.add_argument("--source-version", default=None)
    ap.add_argument("--indent", type=int, default=None)
    ap.add_argument("--validate", dest="validate", action="store_true", default=True)
    ap.add_argument("--no-validate", dest="validate", action="store_false")
    ap.add_argument("--conformance", dest="conformance", action="store_true", default=True)
    ap.add_argument("--no-conformance", dest="conformance", action="store_false")
    ap.add_argument("--bundle", default=None, help="also emit a web bundle in DIR")
    args = ap.parse_args(argv)
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    doc = build_document(args.inputs, args.kind,
                         source_code=args.source_code,
                         source_version=args.source_version)
    blocks = [b for b, present in (("band", doc.band is not None),
                                   ("mesh", doc.mesh is not None)) if present]
    out_parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_parent, exist_ok=True)
    write_json_v1(doc, args.output, indent=args.indent)
    print(f"[pv_export] wrote {args.output}  ({' + '.join(blocks)} block"
          f"{'s' if len(blocks) > 1 else ''}; "
          f"{doc.material.num_atoms} atoms {''.join(doc.material.symbols)})")

    if args.validate:
        validate_schema(args.output)
        print("[pv_export] schema: OK")
    if args.conformance:
        res = run_conformance(doc)
        print(f"[pv_export] conformance: PASS (tier1 {res.tier1_frequencies.passed}, "
              f"tier2 {res.tier2_magnitude.passed}, tier3 "
              f"{res.tier3_complex_aligned.passed}; finite-q covered "
              f"{res.has_finite_q})")
    if args.bundle:
        manifest = write_web_bundle(doc, args.bundle, indent=args.indent)
        print(f"[pv_export] bundle -> {args.bundle}  "
              f"({manifest['n_bundled_modes']} golden modes, finite-q "
              f"{manifest['has_finite_q']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
