#!/usr/bin/env python3
"""Version-stamp sync guard (0.7.8).

The add-on carries THREE version stamps with no mechanical link:
    __init__.py            bl_info["version"] = (X, Y, Z)
    blender_manifest.toml  version = "X.Y.Z"
    CHANGELOG.md           newest "## X.Y.Z" heading

This tool parses all three and exits 0 iff they agree, else 1 naming every
mismatch. Runs before every release (and inside check-all chains).

    python tools/check_stamps.py [--root PATH]   # default: the add-on root
"""
import argparse
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.path.dirname(_HERE),
                    help="add-on root (default: this script's parent's parent)")
    args = ap.parse_args()
    root = args.root

    stamps = {}

    m = re.search(r'"version"\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)',
                  _read(os.path.join(root, "__init__.py")))
    stamps["bl_info (__init__.py)"] = ".".join(m.groups()) if m else "UNPARSEABLE"

    m = re.search(r'^version\s*=\s*"([^"]+)"',
                  _read(os.path.join(root, "blender_manifest.toml")), re.M)
    stamps["blender_manifest.toml"] = m.group(1) if m else "UNPARSEABLE"

    m = re.search(r'^##\s*([0-9]+\.[0-9]+\.[0-9]+)',
                  _read(os.path.join(root, "CHANGELOG.md")), re.M)
    stamps["CHANGELOG.md (newest heading)"] = m.group(1) if m else "UNPARSEABLE"

    values = set(stamps.values())
    for src, v in stamps.items():
        print(f"[stamps] {src:<32s} {v}")
    if len(values) == 1 and "UNPARSEABLE" not in values:
        print(f"[stamps] OK — all three stamps agree: {values.pop()}")
        sys.exit(0)
    print("[stamps] MISMATCH — the three stamps do not agree "
          f"({', '.join(sorted(values))})")
    sys.exit(1)


if __name__ == "__main__":
    main()
