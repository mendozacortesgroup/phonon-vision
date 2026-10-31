# Phonon Vision — Blender addon

Visualize phonon modes from phonopy outputs with publication-quality renders.

Phonon-core is bundled inside the addon as a wheel — no separate install
needed. Works on Blender 4.5 LTS+ on Linux, macOS, and Windows.

## Install

1. Download or build `phonon_vision-0.2.0.zip` (see "Build" below).
2. In Blender: `Edit > Preferences > Add-ons > Install from Disk...`
3. Select the zip file. Enable the **Phonon Vision** add-on.
4. Open the 3D View sidebar (`N` key) — there's a new **Phonon** tab.

## Quick start

1. **Data panel**: browse to your `band.yaml`/`band.hdf5` or
   `mesh.yaml`/`mesh.hdf5`, click *Load File*. The 3D viewport switches
   to orthographic projection automatically (toggle with Numpad 5).
2. **Mode Navigator**: pick a (q, band) with the sliders. The viewport updates
   immediately. For band paths, the tick-buttons (Γ, M, K, Γ) jump to high-symmetry
   points. **Hover any button for a tooltip explaining what it does** — the tooltip
   is dynamic and warns about discontinuities and label artifacts (see below).
3. **Display**: bump *Amplitude* to ~10 to make the motion visible.
   Adjust *Supercell* (default 3×3×3) for 2D materials.
4. **Render**:
   - Pick a style (*Toon (outlined)* matches the classic publication look).
   - Click *Apply Render Preset*.
   - Click a camera button (*Side*, *Top*, *Iso*).
   - Adjust frames per cycle and FPS if you want.
   - Click *Render Animation*. The output goes into the chosen folder.

## Phonopy configuration

For best results, your phonopy run should include:

`band.conf`:
```
EIGENVECTORS = .TRUE.
BAND_CONNECTION = .TRUE.
BAND_LABELS = ...
HDF5 = .TRUE.
```

`mesh.conf` (only if using mesh.yaml):
```
EIGENVECTORS = .TRUE.
MESH_SYMMETRY = .FALSE.
HDF5 = .TRUE.
```

The `HDF5 = .TRUE.` line is optional but **strongly recommended** for any
non-trivial calculation: phonopy will then emit `band.hdf5` / `mesh.hdf5`
alongside the YAMLs, and the addon loads them roughly 100× faster (e.g.
33 s → 0.13 s on a 1414-q × 21-band path).

## Navigating discontinuous band paths

A band path is **continuous** when consecutive segments share a q-point
(e.g. Γ→M→K→Γ). Many real-world paths have **discontinuities** —
non-adjacent jumps in the BZ (e.g. Γ→X then Y→Γ, where X and Y are
distinct points). The high-symmetry buttons in the Mode Navigator handle
both cases, but the navigation semantics are worth understanding.

**Button labels with `|`** mark joins between segments. There are two
kinds:

* **Real discontinuities** like `X|Y` or `V₂|Γ`: clicking the button
  jumps to the X side (last q-point of the left segment). The very next
  q-point (q+1) is Y, on the right segment. Hovering the button shows a
  tooltip with the exact q-indices.

* **`Γ|Γ` (or any `X|X`)**: this is **not** a real discontinuity. It comes
  from phonopy receiving a trailing comma in your `BAND_LABELS` line —
  e.g. `BAND_LABELS = $\Gamma$ X, Y $\Gamma$, $\Gamma$ Z` (note the comma
  after the second `$\Gamma$`). Phonopy attaches the comma to the token,
  so the end-of-segment-2 label is `'$\Gamma$,'` while the
  start-of-segment-3 label is `'$\Gamma$'`. They differ as strings, so
  they get joined with `|`. The two sides are the same physical Γ point
  — the button works correctly, it just labels the join with a `|`.

  If you don't want to see the `Γ|Γ` markers, remove the commas from
  your `BAND_LABELS` line:

      BAND_LABELS = $\Gamma$ X Y $\Gamma$ $\Gamma$ Z ...   # no commas

  Then re-run phonopy. The labels will be clean.

**Band indices across discontinuities.** When you navigate from q=N
(left side of a discontinuity) to q=N+1 (right side) with the band
index fixed, the displayed mode can change abruptly — frequency may
flip sign, eigenvector may differ completely. This is correct physics:
phonopy sorts eigenvalues by frequency at each q-point independently,
and `BAND_CONNECTION = .TRUE.` only tracks continuity within a
single segment, not across the discontinuity. Band N at X and band N
at Y are not the same physical mode in any reproducible sense.

## Notes on the animation

The frame handler computes atom displacements using:

    u_{ℓs}(φ) = A · Re[ e_s(q) · exp(i (q·R_ℓ + φ)) ] / sqrt(m_s)

The phase advances each frame as ``φ = 2π · frame / frames_per_cycle``, so
the rendered loop is **mathematically seamless** — when the player wraps
from the last frame back to the first, the phase advances by exactly
``2π / frames_per_cycle``, identical to every other inter-frame step. No
stutter, no duplicate frame.

If a mode has an imaginary frequency (phonopy emits negative ν), it gets
visualized at ``|ν|`` with a small `imaginary` flag in the UI.

## GIF export

`Render Animation` with format `GIF` produces an animated GIF. Requires
`ffmpeg` on your `PATH` (used for two-pass palette generation). If ffmpeg
isn't available, the addon falls back to leaving an MP4 in the output
folder.

## Build

```bash
cd /path/to/blender-addon
./download_wheels.sh                   # one time — fetches PyYAML wheels
blender --command extension build      # produces phonon_vision-0.2.0.zip
```

## License

GPL-3.0-or-later.
