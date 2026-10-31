# Phonon Vision — headless CLI

Render a phonon mode to an image/movie without opening the GUI, driven by a job
file. Two layers:

- **`tools/pv_render.py`** — the launcher you run, in your **system/conda
  Python**. Reads YAML or JSON, converts to a temp JSON, finds Blender, and runs
  the inner script. Exits with Blender's return code.
- **`tools/pv_inner.py`** — the orchestrator that runs **inside Blender**
  (`--background`). It discovers the installed Phonon Vision extension and drives
  it: load → apply job → build → camera → render. You normally don't call this
  directly; `pv_render.py` does.

## Prerequisites

1. **PyYAML in the launcher's Python** (only needed for YAML jobs):
   ```
   pip install pyyaml        # or: conda install pyyaml
   ```
   JSON jobs need nothing beyond the standard library.
2. **The Phonon Vision extension installed + enabled in the target Blender.**
   The inner script discovers it at runtime — there is no second copy and no
   `--factory-startup`. Locally this is just the normal GUI install; the
   automated headless install on HPC is Phase 3.

## Usage

```
pv_render.py <job.(yaml|yml|json)> [--blender PATH] [-o OUTPUT]
             [--samples N] [--dry-run] [--keep-temp]
```

| flag | effect |
|------|--------|
| `--blender PATH` | Blender executable. Precedence: `--blender` > `$PV_BLENDER` > `blender` on `PATH`. |
| `-o, --output` | override `output.path` in the job |
| `--samples N` | override `render.cycles_samples` (quick quality knob) |
| `--device {CPU,GPU,AUTO}` | override `render.device` (headless render device; default CPU) |
| `--gpu-backend {AUTO,OPTIX,CUDA}` | override `render.gpu_backend` (default AUTO) |
| `--dry-run` | print the resolved command + the temp JSON, then exit 0 (no launch) |
| `--keep-temp` | keep the generated temp JSON instead of deleting it |
| `--clean-frames` | delete the PNG frame sequence after a successful mp4/gif encode (default: keep) |

The job schema is documented in `docs/job_schema_v1.md`. YAML is the friendly
authoring path (comments allowed); JSON is also accepted.

### Local

```
python tools/pv_render.py examples/job_snse2.yaml --blender /opt/blender-4.5/blender
```
(Edit `input.file` in the job to your real band file first.)

### ICER (interactive session)

```
module purge && module load Blender/4.5.9-GCCcore-13.3.0-linux-x86_64-CUDA-12.6.0
python tools/pv_render.py my_job.yaml          # 'blender' is now on PATH
python tools/pv_render.py my_job.yaml --device GPU   # bind CUDA/OptiX on a GPU node
```

## Animation output (mp4 / gif)

For a job with `output.animation: true` and `output.format` `MP4` or `GIF`, the
inner script renders a **PNG frame sequence** (one seamless loop of
`loop.frames_per_cycle` frames at `loop.fps`) into a `<output_stem>_frames/`
directory; then `pv_render` assembles the final file with **ffmpeg** (H.264 mp4
or two-pass-palette gif). The frames are the source of truth and are **kept by
default** (pass `--clean-frames` to delete them after a successful encode). The
final file is named exactly your `-o` path. **ffmpeg must be on PATH** (e.g.
`module load FFmpeg` on ICER); if it's missing the frames are preserved and the
launcher exits nonzero so you can re-encode without re-rendering.

## Behavior notes

- Validation happens **before** Blender launches: a missing `input.file`, a
  missing `output.path` (with no `-o`), or no Blender found all exit nonzero with
  a clear message — so you don't wait through a slow startup to learn of a typo.
- `schema_version` is checked; absent or ≠ 1 only warns.
- The exit code is **Blender's** (0 on success, nonzero on failure), so HPC job
  arrays can detect failures (Phase 3).
- `--dry-run` is the safe way to inspect exactly what will run.

## Out of scope (Phase 3)

SLURM job-array templates, multi-job sweeps, GPU device enabling inside Blender,
the automated headless extension install on ICER, and scratch/copy-back.
Phase 2 is the single-job launcher only.
