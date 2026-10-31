# Phonon Vision — mode sweeps (`pv_sweep`)

Render a whole set of phonon modes from ONE submission. `tools/pv_sweep.py` takes
one **base job** (the look/supercell/camera/render settings to hold fixed) and
sweeps `display.q_index` and/or `display.band_index`, stamping out:

```
<out-dir>/jobs/job_0000.<ext> …     one job per (q, band) task
<out-dir>/sweep_manifest.tsv        task_id  job_path  q  band  output_name   (no header)
<out-dir>/render_sweep.sb           a SLURM --array script that renders task i
```

It runs on a login/dev node in plain Python (no Blender, no compute) and is
instant. Submit the generated array with `sbatch`.

## Axes

Each axis is an explicit list or a range:

- list: `--q 400,403,410`
- range: `--q 0:909:50` → `start:stop[:step]`, **Python-range semantics, end
  EXCLUSIVE**. So `--band 0:9` expands to `0,1,2,3,4,5,6,7,8` (nine bands), and
  `--q 0:909:50` is `0,50,…,900`.

Pairing is `--mode product` (cartesian q×band, default) or `--mode zip` (paired;
the two axes must have equal length, else it errors).

`--format mp4|gif|png` sets `output.format` and `output.animation` (mp4/gif →
animation, png → a single still). `--device`/`--samples` optionally override
`render.device`/`render.cycles_samples` in every generated job.

Each generated job is byte-for-byte the base job except `display.q_index`,
`display.band_index`, and `output.*`. The job's `output.path` is a bare basename
(from `--name-template`, default `PV_q{q:04d}_b{b:02d}`); the array script sets
the real `-o` path in per-task scratch and copies the result to
`~/phonon-vision/renders/`.

## Worked examples

### 1. Every branch at one q-point (acoustic vs optical at a fixed wavevector)

```
python tools/pv_sweep.py base.yaml --q 403 --band 0:9 --format mp4 \
    --out-dir ~/sweeps/q403_allbands
sbatch ~/sweeps/q403_allbands/render_sweep.sb
```
Nine tasks (bands 0–8) at q-index 403, one mp4 each.

### 2. One branch swept along the BZ path

```
python tools/pv_sweep.py base.yaml --band 5 --q 0:909:30 --format mp4 \
    --out-dir ~/sweeps/band5_path
sbatch ~/sweeps/band5_path/render_sweep.sb
```
Band 5 sampled every 30 q-points from 0 to 900 (31 tasks).

### 3. A small product matrix

```
python tools/pv_sweep.py base.yaml --q 400,403,410 --band 0,1 --mode product \
    --format mp4 --device GPU --out-dir ~/sweeps/matrix
sbatch ~/sweeps/matrix/render_sweep.sb
```
Six tasks (3 q × 2 band).

## The generated array script

`render_sweep.sb` mirrors the hero render script's environment block —
`module purge`; `module load <blender-module>`; `module load Miniforge3`;
`source "$(conda info --base)/etc/profile.d/conda.sh"`; `conda activate <env>`
(which brings ffmpeg for mp4/gif assembly). It then:

- declares `#SBATCH --array=0-<N-1>%<concurrency>` (default concurrency 5);
- sanity-checks that the phonon_vision extension is installed/enabled (exit 2 if
  not);
- looks up this task's row with
  `awk -v id=$SLURM_ARRAY_TASK_ID -F'\t' '$1==id{print}' sweep_manifest.tsv`;
- renders into per-task scratch
  `/mnt/gs21/scratch/$USER/pv/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}`
  via `pv_render.py … --device GPU -o "$OUT"`;
- copies the result to `~/phonon-vision/renders/` and exits nonzero on failure.

Tune the SLURM directives with `--account`, `--walltime`, `--gpus`,
`--array-concurrency`, `--env`, and `--blender-module`.

## Out of scope

Contact-sheet/montage assembly; reading the file to expand `--band all`
(use explicit lists/ranges); sweeping non-mode axes (camera/amplitude) — this
sweeps `q_index`/`band_index` only.
