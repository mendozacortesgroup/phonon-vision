#!/usr/bin/env python3
"""Vendored-wheel parity guard (0.7.8; SHARED-SUBSET contract since 0.7.14).

The add-on ships a phonon_core wheel cut from the PRIVATE phonon-core line.
The architect audit (Report IX era) found the vendored modules byte-identical
to private 0.6.75 — true that day, guarded by nothing. This tool makes it a
check that runs before EVERY add-on release:

    python tools/check_vendored_parity.py --core-src ../phonon-core

It unzips the wheel named in blender_manifest.toml (or --wheel PATH) and
verifies THE SHARED SUBSET CONTRACT (tools/_subset.py — the same constant
tools/build_vendored_wheel.py assembles the wheel from; one list, two
consumers, no drift):

  1. every SUBSET_MODULES entry is present in the wheel AND byte-identical
     (md5) to src/phonon_core/<same relative path> in --core-src;
  2. the TWO packaging stubs match their PINNED md5s (STUB_PINS);
  3. the wheel contains NOTHING ELSE — an extra module is a parity break
     exactly like a missing one.

Exit codes:
    0  17 modules byte-identical + 2 pinned stubs, nothing extra/missing —
       prints the module list and the core version read from the src pyproject
    1  content divergence (module differs, or a stub hash moved off its pin)
    2  set mismatch — a wheel module with no counterpart in the SUBSET list,
       or a subset/stub entry missing from the wheel; takes precedence

THE SANCTIONED DIVERGENCES — the packaging stubs. The wheel's root
phonon_core/__init__.py carries its own __version__ (the public wheel line:
0.2.0 since 0.7.14) and imports only the vendored subset; since 0.2.0 the
wheel also carries phonon_core/engine/__init__.py exposing the A2 numpy
engine ONLY (the phonopy-backed A1 is deliberately absent). Both stubs are
written from templates in tools/build_vendored_wheel.py and compared against
the PINNED md5s in tools/_subset.py. When a stub is deliberately re-cut (new
wheel version, subset change), re-pin IN THE SAME COMMIT and record it in
CLAUDE.md's provenance line. If a stub hash moves without a re-pin, that is a
parity break (exit 1). HISTORY: the single 0.1.4-era stub pin
07478e09624c01a14ade8f11622efd72 is RETIRED (superseded by the two 0.2.0
pins in tools/_subset.py).
"""
import argparse
import difflib
import hashlib
import os
import re
import sys
import tempfile
import zipfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from _subset import STUB_PINS, SUBSET_MODULES   # noqa: E402  (shared contract)


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _wheel_from_manifest():
    manifest = os.path.join(_ROOT, "blender_manifest.toml")
    with open(manifest, encoding="utf-8") as f:
        m = re.search(r'"\./wheels/(phonon_core-[^"]+\.whl)"', f.read())
    if not m:
        sys.exit("[parity] no phonon_core wheel entry in blender_manifest.toml")
    return os.path.join(_ROOT, "wheels", m.group(1))


def _core_version(core_src):
    pyproject = os.path.join(core_src, "pyproject.toml")
    try:
        with open(pyproject, encoding="utf-8") as f:
            m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
        return m.group(1) if m else "?"
    except OSError:
        return "? (no pyproject.toml)"


def _diff_count(a_path, b_path):
    with open(a_path, encoding="utf-8", errors="replace") as f:
        a = f.readlines()
    with open(b_path, encoding="utf-8", errors="replace") as f:
        b = f.readlines()
    return sum(1 for line in difflib.unified_diff(a, b, lineterm="")
               if line[:1] in "+-" and line[:3] not in ("+++", "---"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--core-src", required=True,
                    help="phonon_core source tree root (contains src/phonon_core)")
    ap.add_argument("--wheel", default=None,
                    help="phonon_core wheel path (default: the manifest's entry)")
    args = ap.parse_args()

    wheel = args.wheel or _wheel_from_manifest()
    core_pkg = os.path.join(args.core_src, "src", "phonon_core")
    if not os.path.isdir(core_pkg):
        sys.exit(f"[parity] not a core src tree (no src/phonon_core): {args.core_src}")

    ok, stub_ok, diffs, set_breaks = [], [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(tmp)
        whl_pkg = os.path.join(tmp, "phonon_core")
        rels = sorted(
            os.path.relpath(os.path.join(dp, fn), whl_pkg).replace(os.sep, "/")
            for dp, _dns, fns in os.walk(whl_pkg)
            for fn in fns if fn.endswith(".py")
        )
        expected = set(SUBSET_MODULES) | set(STUB_PINS)
        for rel in sorted(expected - set(rels)):
            set_breaks.append(f"MISSING-FROM-WHEEL  phonon_core/{rel}")
        for rel in sorted(set(rels) - expected):
            set_breaks.append(f"NOT-IN-SUBSET-LIST  phonon_core/{rel}")
        for rel in rels:
            wpath = os.path.join(whl_pkg, rel)
            if rel in STUB_PINS:
                got = _md5(wpath)
                if got == STUB_PINS[rel]:
                    stub_ok.append(rel)
                else:
                    diffs.append((rel, f"stub md5 {got[:8]} != pinned "
                                       f"{STUB_PINS[rel][:8]} (re-pin deliberately)"))
                continue
            if rel not in expected:
                continue                       # already a set break above
            spath = os.path.join(core_pkg, rel)
            if not os.path.isfile(spath):
                set_breaks.append(f"NO-COUNTERPART      phonon_core/{rel}")
            elif _md5(wpath) == _md5(spath):
                ok.append(rel)
            else:
                diffs.append((rel, f"{_diff_count(wpath, spath)} diff lines"))

    ver = _core_version(args.core_src)
    print(f"[parity] wheel: {os.path.basename(wheel)}")
    print(f"[parity] core src: {args.core_src} (version {ver})")
    print(f"[parity] contract: {len(SUBSET_MODULES)} subset modules + "
          f"{len(STUB_PINS)} pinned stubs (tools/_subset.py)")
    if set_breaks:
        print(f"[parity] PARITY BREAK — {len(set_breaks)} set mismatch(es):")
        for line in set_breaks:
            print(f"    {line}")
    if diffs:
        print(f"[parity] PARITY BREAK — {len(diffs)} divergent file(s):")
        for rel, note in diffs:
            print(f"    DIFFER  phonon_core/{rel:<28s} {note}")
    if set_breaks:
        sys.exit(2)
    if diffs:
        sys.exit(1)
    for rel in ok:
        print(f"    MATCH   phonon_core/{rel}")
    for rel in sorted(stub_ok):
        print(f"    STUB-OK phonon_core/{rel} (pinned "
              f"{STUB_PINS[rel][:8]})")
    print(f"[parity] OK — {len(ok)} modules byte-identical to core {ver} "
          f"+ {len(stub_ok)} pinned packaging stubs")
    sys.exit(0)


if __name__ == "__main__":
    main()
