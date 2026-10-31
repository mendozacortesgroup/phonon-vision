# Phonon Vision — job file schema v1

A **job file** captures every `scene.phonon_vision` setting so a render can be
reproduced headless. Phase 1 serializes it as **JSON** (stdlib `json`, always
available under `blender --background`). A friendlier YAML authoring layer
(YAML→JSON) arrives with the Phase 2 outer driver.

The mapping is implemented once in `phonon_vision/job.py` (`_MAP` +
`apply_job_to_scene` / `scene_to_job`) and is **provably exhaustive**: a test
cross-checks `_MAP` against every property declared on `PhononVisionProps`
(54/54 covered, plus the species collection and the three Miller indices).

```json
{ "schema_version": 1, "input": {…}, "display": {…}, "appearance": {…},
  "camera": {…}, "render": {…}, "overlay": {…}, "loop": {…}, "output": {…} }
```

All keys are optional on input: `apply_job_to_scene` only writes the keys that
are present, leaving everything else at its current/default value. `scene_to_job`
always emits the full set.

## schema_version
`schema_version` (int) — currently `1`.

## input
| key | prop | type | notes |
|-----|------|------|-------|
| `file` | `yaml_path` | str | phonopy band/mesh `.yaml`/`.yml`/`.hdf5`/`.h5`. The inner script loads this first. |
| `data_kind` | `data_kind` | enum `NONE\|BAND\|MESH` | set automatically by the loader from the filename; included for round-trip. |

## display (mode + physics + geometry — all build/animation-time)
| key | prop | type | notes |
|-----|------|------|-------|
| `q_index` | `q_index` | int | q-point index. |
| `band_index` | `band_index` | int | band index. |
| `amplitude` | `amplitude` | float | displacement amplitude (Å). |
| `clamp_amplitude` | `clamp_amplitude` | bool | clamp the per-atom displacement. |
| `max_displacement` | `max_displacement` | float | clamp ceiling (Å). |
| `canonical_phase` | `use_canonical_phase` | bool | fix the global mode phase canonically. |
| `imaginary_threshold` | `imag_freq_threshold` | float | THz below which a mode is treated as imaginary/soft. |
| `supercell` | `supercell_n` | int[3] | supercell N×M×K (build-time). |
| `mesh_quality` | `render_quality` | enum `LOW\|MEDIUM\|HIGH` | sphere/arrow tessellation — **build-time geometry**. |
| `show_arrows` | `show_arrows` | bool | eigenvector arrows. |
| `arrow_scale` | `arrow_scale` | float | arrow length scale. |
| `show_unit_cell` | `show_unit_cell` | bool | draw the cell box. |
| `unit_cell_kind` | `unit_cell_kind` | enum `PRIMITIVE\|GRID\|SUPERCELL` | which cell to outline. |
| `cell_thickness` | `cell_thickness` | float | cell **edge thickness**. |
| `show_bonds` | `show_bonds` | bool | draw bonds. |
| `bonds_across_pbc` | `bond_periodic` | bool | **show atoms & bonds across periodic boundaries**. |
| `bond_tolerance` | `bond_tolerance` | float | bond detection tolerance. |
| `bond_thickness` | `bond_thickness` | float | bond cylinder radius. |
| `bond_two_color` | `bond_two_color` | bool | split-color bonds by the two species. |
| `bond_max_distance_factor` | `bond_break_max_factor` | float | **max bond distance** factor. |
| `bond_min_distance_factor` | `bond_break_min_factor` | float | **min bond distance** factor. |

## appearance (materials / colors)
| key | prop | type | notes |
|-----|------|------|-------|
| `vibrancy` | `species_vibrancy` | float | species emission strength. |
| `arrow_color` | `arrow_color` | float[3] | global arrow RGB. |
| `species` | `species_overrides` | object | `{ "<symbol>": { "color": [r,g,b], "radius": Å } }`. `radius` is **build-time geometry**; `color` is appearance. Applied to the entries that `load_file` populated, matched by symbol. |

## camera
| key | prop | type | notes |
|-----|------|------|-------|
| `mode` | `camera_view` | enum `SIDE\|TOP\|ISO\|SPHERICAL\|MILLER` | placement mode. |
| `auto_frame` | `camera_auto_frame` | bool | auto ortho-scale + distance. |
| `ortho_scale` | `camera_ortho_scale` | float | manual ortho frame width (zoom) when `auto_frame` is false. |
| `distance` | `camera_distance` | float | manual distance (positions only, in ortho). |
| `azimuth` | `camera_azimuth` | float | spherical azimuth (deg). |
| `polar` | `camera_polar` | float | spherical polar from +Z (deg, 0–180). |
| `radius` | `camera_radius` | float | spherical distance (Å). |
| `miller` | `camera_miller_h/k/l` | float[3] | Miller `[h,k,l]` plane normal (reciprocal direction). |
| `target_offset` | `camera_target_offset` | float[3] | look-at / orbit-center offset from bbox center (Å). |

## render
| key | prop | type | notes |
|-----|------|------|-------|
| `preset` | `render_preset` | enum `TOON\|EEVEE_PBR\|CYCLES_PBR` | render style (drives the engine/world/shaders via `apply_preset`). |
| `lighting` | `lighting_style` | enum `STUDIO\|SCIENTIFIC\|DRAMATIC` | light rig. |
| `eevee_samples` | `eevee_samples` | int | EEVEE samples. |
| `cycles_samples` | `cycles_samples` | int | Cycles samples. |
| `denoising` | `use_denoising` | bool | denoiser on/off. |
| `resolution_x` | `render_resolution_x` | int | px. |
| `resolution_y` | `render_resolution_y` | int | px. |
| `resolution_percentage` | `render_percentage` | int | % of resolution. |
| `transparent_background` | `transparent_background` | bool | film transparency (alpha). |
| `device` | *(headless-only)* | enum `CPU\|GPU\|AUTO` | render device for the headless/CLI path (default `CPU` = unchanged). `GPU` binds Cycles to the graphics device; `AUTO` uses the GPU if present, else CPU. Set ephemerally per headless process — never touches the GUI's Cycles prefs. |
| `gpu_backend` | *(headless-only)* | enum `AUTO\|OPTIX\|CUDA` | which Cycles GPU backend to try when `device` ≠ `CPU` (default `AUTO` = OPTIX then CUDA). |

## overlay
*(Phase 1 sets these props; burning the overlay into the rendered image is deferred — the compositor mechanism comes later.)*

| key | prop | type | notes |
|-----|------|------|-------|
| `show_overlay` | `show_overlay` | bool | master overlay toggle. |
| `show_mode_overlay` | `show_mode_overlay` | bool | mode-info text (q, band, frequency). |
| `text_size` | `overlay_text_size` | int | pt. |
| `text_color` | `overlay_text_color` | float[3] | RGB. |
| `background_color` | `overlay_background_color` | float[4] | RGBA. |

## loop
| key | prop | type | notes |
|-----|------|------|-------|
| `frames_per_cycle` | `frames_per_cycle` | int | frames in one phonon cycle. For `animation: true` this is the rendered length — exactly one seamless loop (the seam frame is not duplicated), the same range the GUI playback uses. |
| `fps` | `fps_render` | int | playback/render fps; also the encode frame rate for mp4/gif. |

## output
| key | prop / role | type | notes |
|-----|------|------|-------|
| `directory` | `output_directory` | str | GUI output dir. |
| `format` | `output_format` | enum `MP4\|GIF\|PNG` | container. For `animation: true`, the inner renders a **PNG frame sequence** (frames-first) and the launcher `pv_render` assembles the final file with ffmpeg: **MP4** = H.264/yuv420p/crf18; **GIF** = two-pass palettegen/paletteuse (larger/lower quality — for slides/web). **PNG** = a still (`animation: false`) or a kept frame sequence (`animation: true`). |
| `path` | *(job-only)* | str | resolved output path consumed directly by `pv_inner.py`. For PNG it is the frame prefix; for MP4 the `.mp4` file. |
| `animation` | *(job-only)* | bool | `true` → render the whole loop; `false` → a single still of the current frame. |

`path` and `animation` are **not** props — they are read straight from the job
by the inner script (`render_headless(..., animation=…, output_path=…,
file_format=…)`). `scene_to_job` synthesizes them so an exported job is runnable.

## Notes / gotchas
- **shading** (solid/material/rendered) is a *viewport* concept; the rendered
  look is driven by `render.preset` via `apply_preset`. Not a job field.
- **mesh_quality** and **species radii** affect geometry, so the inner script
  applies the job *before* `build_scene_from_data`.
- **transparent background / resolution**: the inner script sets
  `render.film_transparent` and `resolution_x/y/percentage` explicitly before
  rendering, in case `apply_preset` doesn't.

## schema_version 2 (0.7.9) — curated bonds

One ADDITIVE, presence-keyed change: an OPTIONAL top-level `bonds` key holding
a navigator bond side-car (`docs/bonds_sidecar.md`, format
`"phonon-vision-bonds"` v1) **embedded by value** — never a file path, so a
cluster job stays self-contained — plus a `source` provenance block
(`{"filename": …, "imported": "YYYY-MM-DD"}`) added at GUI import time.

```json
{
  "schema_version": 2,
  "...": "...",
  "bonds": {
    "format": "phonon-vision-bonds",
    "version": 1,
    "material": { "formula": "SnSeSe", "natoms": 3 },
    "units": "angstrom",
    "bonds": { "Se-Sn": { "max": 3.11, "radius_scale": 1.5 } },
    "source": { "filename": "snse2_bonds.json", "imported": "2026-07-03" }
  }
}
```

- While present, the pair table REPLACES the covalent `bond_tolerance`
  criterion entirely (pairs absent from the table form no bonds; `max` is an
  absolute cutoff in Å). `color` is stored but not applied at 0.7.9.
- The reader accepts `schema_version` 1 **and** 2; the writer emits 2.
- `bonds` is NOT partial-job keyed: a job WITHOUT the key CLEARS any loaded
  spec — a job is a complete render description, so v1 jobs render
  byte-identically to 0.7.8 regardless of prior GUI state.
