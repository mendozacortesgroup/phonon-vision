#!/usr/bin/env python3
"""pv_sweep — generate a sweep of Phonon Vision jobs + a SLURM array script.

Runs on a login/dev node in plain system/conda Python (NO bpy, NO phonon_core,
NO compute). Takes ONE base job and one or two variation axes (q_index and/or
band_index) and stamps out N job files that differ ONLY in
display.q_index / display.band_index / output, plus:

    <out-dir>/jobs/job_0000.<ext> …      one job per task
    <out-dir>/sweep_manifest.tsv         task_id  job_path  q  band  output_name
    <out-dir>/render_sweep.sb            a SLURM --array script that renders task i

Then:  sbatch <out-dir>/render_sweep.sb

Axes accept an explicit list ("400,403,410") or a range ("start:stop[:step]",
PYTHON-RANGE semantics, end-EXCLUSIVE — "0:9" is 0..8). Pairing is cartesian
product (default) or zip (paired; equal lengths required).
"""
import argparse
import copy
import os
import sys

_EXT = {"mp4": ("mp4", "MP4", True),
        "gif": ("gif", "GIF", True),
        "png": ("png", "PNG", False)}


def die(msg, code=2):
    print(f"[pv-sweep] ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def parse_axis(spec, name):
    """'400,403,410' -> [400,403,410]; '0:9' -> range(0,9); '0:909:50' ->
    range(0,909,50). Python-range semantics (end-exclusive)."""
    spec = spec.strip()
    try:
        if ":" in spec:
            parts = [int(x) for x in spec.split(":")]
            if len(parts) == 2:
                return list(range(parts[0], parts[1]))
            if len(parts) == 3:
                return list(range(parts[0], parts[1], parts[2]))
            die(f"--{name} range must be start:stop or start:stop:step, got {spec!r}")
        return [int(x) for x in spec.split(",") if x.strip() != ""]
    except ValueError:
        die(f"--{name} must be integers (list or range), got {spec!r}")


def load_base(path):
    if not os.path.isfile(path):
        die(f"base job not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            die("PyYAML required for a YAML base job: pip install pyyaml")
        with open(path) as fh:
            return yaml.safe_load(fh), ext
    if ext == ".json":
        import json
        with open(path) as fh:
            return json.load(fh), ext
    die(f"unrecognized base job extension {ext!r} (use .yaml/.yml/.json)")


def dump_job(job, path, ext):
    if ext in (".yaml", ".yml"):
        import yaml
        with open(path, "w") as fh:
            yaml.safe_dump(job, fh, sort_keys=False)
    else:
        import json
        with open(path, "w") as fh:
            json.dump(job, fh, indent=2)


def build_pairs(qs, bands, mode):
    if mode == "zip":
        if len(qs) != len(bands):
            die(f"--mode zip needs equal lengths, got {len(qs)} q and "
                f"{len(bands)} band")
        return list(zip(qs, bands))
    # product (default): q-major
    return [(q, b) for q in qs for b in bands]


SB_TEMPLATE = """#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --time={walltime}
#SBATCH --gpus={gpus}
#SBATCH --array=0-{last}%{concurrency}
#SBATCH --output={out_dir}/logs/task_%A_%a.out

# --- environment (mirrors render_hero.sb; conda brings ffmpeg) ---
module purge
module load {blender_module}
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {env}

set -u
SWEEP_DIR="{out_dir}"
PV_RENDER="{pv_render}"
cd "$SWEEP_DIR"

# --- extension sanity check (exit 2 if the add-on isn't installed/enabled) ---
if ! blender --background \\
     --python-expr "import sys; sys.exit(0 if any(m.split('.')[-1]=='phonon_vision' for m in sys.modules) else 2)" \\
     >/dev/null 2>&1; then
  echo "[sweep] phonon_vision extension not installed/enabled in this Blender" >&2
  exit 2
fi

# --- look up THIS task's row in the manifest ---
ROW=$(awk -v id="$SLURM_ARRAY_TASK_ID" -F'\\t' '$1==id{{print; exit}}' sweep_manifest.tsv)
if [ -z "$ROW" ]; then
  echo "[sweep] no manifest row for task $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi
JOB=$(printf '%s' "$ROW" | cut -f2)
OUT_NAME=$(printf '%s' "$ROW" | cut -f5)

# --- per-task scratch, render, copy back ---
SCRATCH="/mnt/gs21/scratch/$USER/pv/${{SLURM_JOB_NAME}}.${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
mkdir -p "$SCRATCH"
OUT="$SCRATCH/$OUT_NAME"

python "$PV_RENDER" "$JOB" --blender "$(which blender)" --device {device} -o "$OUT"
rc=$?
if [ $rc -ne 0 ]; then
  echo "[sweep] task $SLURM_ARRAY_TASK_ID failed (rc=$rc); scratch: $SCRATCH" >&2
  exit $rc
fi

DEST="$HOME/phonon-vision/renders"
mkdir -p "$DEST"
cp "$OUT" "$DEST/"
echo "[sweep] task $SLURM_ARRAY_TASK_ID OK -> $DEST/$OUT_NAME"
"""


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="pv_sweep.py",
        description="Generate N Phonon Vision jobs + a SLURM array script from "
                    "one base job swept over q_index/band_index.")
    p.add_argument("base_job", help="base job (.yaml/.yml/.json): the fixed "
                                    "look/supercell/camera/etc.")
    p.add_argument("--q", required=True, help='q_index axis: "400,403,410" or '
                                              '"start:stop[:step]" (range is end-exclusive)')
    p.add_argument("--band", required=True, help="band_index axis: same syntax")
    p.add_argument("--mode", choices=["product", "zip"], default="product",
                   help="product = cartesian q×band (default); zip = paired")
    p.add_argument("--format", choices=["mp4", "gif", "png"], default="mp4",
                   help="output.format; mp4/gif => animation, png => still")
    p.add_argument("--out-dir", required=True, help="output directory")
    p.add_argument("--name-template", default="PV_q{q:04d}_b{b:02d}",
                   help="per-task output basename (default PV_q{q:04d}_b{b:02d})")
    p.add_argument("--device", choices=["CPU", "GPU", "AUTO"],
                   help="override render.device in ALL jobs (and the .sb)")
    p.add_argument("--samples", type=int,
                   help="override render.cycles_samples in ALL jobs")
    p.add_argument("--array-concurrency", type=int, default=5,
                   help="the %%K cap on the array (default 5)")
    p.add_argument("--account", default="mendoza_q")
    p.add_argument("--walltime", default="02:00:00")
    p.add_argument("--gpus", default="v100:1")
    p.add_argument("--env", default="pv", help="conda env to activate (brings ffmpeg)")
    p.add_argument("--blender-module",
                   default="Blender/4.5.9-GCCcore-13.3.0-linux-x86_64-CUDA-12.6.0")
    args = p.parse_args(argv)

    base, base_ext = load_base(args.base_job)
    if not isinstance(base, dict):
        die("base job did not parse to a mapping/object")

    qs = parse_axis(args.q, "q")
    bands = parse_axis(args.band, "band")
    if not qs or not bands:
        die("both --q and --band must expand to at least one value")
    pairs = build_pairs(qs, bands, args.mode)
    n = len(pairs)

    ext, fmt, animation = _EXT[args.format]

    out_dir = os.path.abspath(args.out_dir)
    jobs_dir = os.path.join(out_dir, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    manifest_rows = []
    for i, (q, b) in enumerate(pairs):
        job = copy.deepcopy(base)
        disp = job.setdefault("display", {})
        disp["q_index"] = q
        disp["band_index"] = b
        out = job.setdefault("output", {})
        out["format"] = fmt
        out["animation"] = animation
        try:
            base_name = args.name_template.format(q=q, b=b)
        except (KeyError, IndexError, ValueError) as e:
            die(f"--name-template invalid (use {{q}} and {{b}}): {e}")
        output_name = f"{base_name}.{ext}"
        out["path"] = output_name   # bare basename; the .sb sets the real -o
        if args.device is not None:
            job.setdefault("render", {})["device"] = args.device
        if args.samples is not None:
            job.setdefault("render", {})["cycles_samples"] = args.samples

        job_path = os.path.join(jobs_dir, f"job_{i:04d}{base_ext}")
        dump_job(job, job_path, base_ext)
        # manifest job_path is relative to the sweep dir (the .sb cd's there)
        rel_job = os.path.relpath(job_path, out_dir)
        manifest_rows.append((i, rel_job, q, b, output_name))

    manifest = os.path.join(out_dir, "sweep_manifest.tsv")
    with open(manifest, "w") as fh:
        for row in manifest_rows:
            fh.write("\t".join(str(c) for c in row) + "\n")

    pv_render = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "pv_render.py")
    job_name = os.path.basename(out_dir.rstrip("/")) or "pv_sweep"
    sb_path = os.path.join(out_dir, "render_sweep.sb")
    with open(sb_path, "w") as fh:
        fh.write(SB_TEMPLATE.format(
            job_name=job_name,
            account=args.account,
            walltime=args.walltime,
            gpus=args.gpus,
            last=n - 1,
            concurrency=args.array_concurrency,
            out_dir=out_dir,
            blender_module=args.blender_module,
            env=args.env,
            pv_render=pv_render,
            device=(args.device or "GPU"),
        ))

    # summary
    print(f"[pv-sweep] {n} tasks "
          f"({len(qs)} q × {len(bands)} band, {args.mode})")
    print(f"  q:      {qs}")
    print(f"  band:   {bands}")
    print(f"  format: {args.format} (animation={animation})")
    print(f"  jobs:   {jobs_dir}/job_0000{base_ext} … job_{n-1:04d}{base_ext}")
    print(f"  manifest: {manifest}")
    print(f"  script:   {sb_path}")
    print(f"\n  submit:   sbatch {sb_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
