#!/usr/bin/env bash
# Download the PyYAML + h5py wheels the extension bundles (all five Blender
# platforms). Run before building; blender_manifest.toml lists the results.
#
# ALL knobs live below: package versions are PINNED (0.7.7 shipped a split
# h5py set — 3.14.0 linux / 3.16.0 mac+win — precisely because the old script
# let pip resolve "latest per platform" at download time; never again), and
# PY_TAG pins the CPython ABI of every binary wheel.
#
# ── BLENDER 5.2 MIGRATION PROCEDURE (the cp311 landmine) ────────────────────
# Every binary wheel here is cp311 = Blender 4.5 LTS's bundled Python 3.11.
# If Blender 5.x bumps its bundled Python, these wheels silently stop
# importing. When Blender 5.2 ships:
#   1. Read its bundled Python version:
#        blender --background --python-expr 'import sys; print(sys.version)'
#   2. Re-run this script with the new tag, e.g.:  ./download_wheels.sh --py-tag cp313
#      (verify first that PINNED versions publish wheels for that tag on all
#      five platforms — pip index / pypi.org JSON; bump the pins if not)
#   3. Update the wheels = [...] list in blender_manifest.toml to the new
#      filenames and delete the old-tag wheels.
#   4. Run the headless smoke under 5.2 (tools/pv_headless_smoke.py).
#   5. ONLY after the smoke passes: raise blender_version_min.
# See CLAUDE.md "Blender 5.2 readiness".
#
# ── THE macOS x86_64 DEPLOYMENT-TARGET LANDMINE (found migrating to cp313) ──
# PyPI's minimum macOS x86_64 wheel tag for a given cp-tag is NOT stable
# across CPython releases: cp311 wheels publish macosx_10_9_x86_64, but
# cp313 wheels publish macosx_10_13_x86_64 (CPython 3.13 raised its own
# supported macOS baseline) — requesting 10_9 under --abi cp313 finds NO
# matching distribution and pip fails outright (not a "latest resolves
# differently" case; the 10_9-tagged wheel for cp313 does not exist upstream).
# MACOS_X86_64_TAG below is therefore itself a PINNED input, keyed by py-tag,
# verified against the live PyPI JSON index at migration time — the same
# reproducibility discipline as PYYAML_VERSION/H5PY_VERSION. If a future
# migration hits "No matching distribution" again for macosx_10_9/10_13/…,
# check pypi.org/pypi/<pkg>/<version>/json for the actual macosx_*_x86_64
# tag the new cp-tag publishes and add a case below — do not guess.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ---- knobs ------------------------------------------------------------------
PYYAML_VERSION="6.0.2"
H5PY_VERSION="3.14.0"   # newest with cp311 wheels for ALL five platforms (2026-07)
PY_TAG="cp311"          # Blender 4.5 LTS bundles Python 3.11
# -----------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --py-tag) PY_TAG="$2"; shift 2 ;;
        *) echo "usage: $0 [--py-tag cpXYZ]" >&2; exit 2 ;;
    esac
done
# cp311 -> 3.11, cp313 -> 3.13 (single-digit major assumed)
_digits="${PY_TAG#cp}"
PY_VERSION="${_digits:0:1}.${_digits:1}"

# macOS x86_64 minimum-deployment tag PyPI publishes, keyed by py-tag (see
# the landmine note above) — PINNED per cp-tag, not auto-detected.
case "${PY_TAG}" in
    cp311) MACOS_X86_64_TAG="macosx_10_9_x86_64" ;;
    cp313) MACOS_X86_64_TAG="macosx_10_13_x86_64" ;;
    *)
        echo "ERROR: no pinned macosx_*_x86_64 tag for ${PY_TAG} — check" \
             "pypi.org/pypi/<pkg>/<version>/json for the actual tag and add" \
             "a case above; do not guess." >&2
        exit 2
        ;;
esac

cd "$(dirname "$0")/wheels"
echo "Python tag: ${PY_TAG} (interpreter ${PY_VERSION}, macOS x86_64 tag ${MACOS_X86_64_TAG})"

# Linux x64 + arm64, macOS x64 + arm64, Windows x64
declare -a TARGETS=(
    "manylinux2014_x86_64"
    "manylinux2014_aarch64"
    "${MACOS_X86_64_TAG}"
    "macosx_11_0_arm64"
    "win_amd64"
)

fails=0
for spec in "pyyaml==${PYYAML_VERSION}" "h5py==${H5PY_VERSION}"; do
    for plat in "${TARGETS[@]}"; do
        echo "Downloading ${spec} for ${plat}..."
        pip download "${spec}" \
            --only-binary=:all: \
            --platform "${plat}" \
            --python-version "${PY_VERSION}" \
            --abi "${PY_TAG}" \
            --no-deps \
            -d . || { echo "  FAILED: ${spec} @ ${plat}"; fails=$((fails + 1)); }
    done
done

echo "Done. Wheels in: $(pwd)"
ls -la *.whl
if [[ ${fails} -gt 0 ]]; then
    # The manifest lists all five platforms — a missing wheel must be loud.
    echo "ERROR: ${fails} wheel download(s) failed; the bundle is incomplete." >&2
    exit 1
fi
