#!/usr/bin/env python3
"""pv_render — Phonon Vision single-job launcher (Phase 2, outer driver).

Runs in the SYSTEM / conda Python (NOT Blender's). Reads a friendly YAML (or
JSON) job, converts it to a temp JSON (Blender's bundled Python has no PyYAML),
locates Blender, and fires the exact Phase-1 command:

    blender --background --python pv_inner.py -- --job <temp.json>

Usage:
    pv_render.py <job.(yaml|yml|json)> [--blender PATH] [-o OUTPUT]
                 [--samples N] [--dry-run] [--keep-temp]

PREREQUISITES:
  - PyYAML in THIS Python (the launcher's env) to read YAML jobs:
      pip install pyyaml      (or: conda install pyyaml)
  - The phonon_vision extension must be INSTALLED + ENABLED in the target
    Blender — the inner script discovers it at runtime. Locally that is the
    normal GUI install; the automated HPC install is Phase 3.
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile


def die(msg, code=2):
    """Print an error to stderr and exit nonzero (pre-launch failures)."""
    print(f"[pv-render] ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def warn(msg):
    print(f"[pv-render] warning: {msg}", file=sys.stderr)


def load_job(path):
    ext = os.path.splitext(path)[1].lower()
    if not os.path.isfile(path):
        die(f"job file not found: {path}")
    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            die("PyYAML is required to read YAML jobs. Install it in this "
                "Python:  pip install pyyaml   (or: conda install pyyaml). "
                "Alternatively, pass a .json job.")
        with open(path) as fh:
            return yaml.safe_load(fh)
    if ext == ".json":
        with open(path) as fh:
            return json.load(fh)
    die(f"unrecognized job extension '{ext}'. Use .yaml/.yml or .json.")


def resolve_blender(arg_blender):
    """Precedence: --blender > $PV_BLENDER > PATH ('blender')."""
    cand = arg_blender or os.environ.get("PV_BLENDER")
    if cand:
        # Accept either an explicit path or a bare name resolvable on PATH.
        resolved = cand if os.path.isfile(cand) else shutil.which(cand)
        if not resolved:
            die(f"Blender not found at: {cand}")
        return resolved
    resolved = shutil.which("blender")
    if not resolved:
        die("No Blender found. Pass --blender /path/to/blender, set $PV_BLENDER, "
            "or 'module load Blender/4.5.9-...' so 'blender' is on PATH.")
    return resolved


VIDEO_EXTS = {".mp4": "MP4", ".gif": "GIF"}
_CONTAINER_EXT = {"MP4": ".mp4", "GIF": ".gif", "PNG": ".png"}


def _find_frame_sequence(frames_dir):
    """Return (ffmpeg_input_pattern, start_number, count) for frame_*.png in
    frames_dir, detecting the zero-pad width from the files themselves. None if
    no frames are present."""
    import glob
    import re
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not files:
        return None
    m = re.search(r"frame_(\d+)\.png$", os.path.basename(files[0]))
    if not m:
        return None
    width = len(m.group(1))
    start = int(m.group(1))
    pattern = os.path.join(frames_dir, f"frame_%0{width}d.png")
    return pattern, start, len(files)


def _assemble_mp4(ffmpeg, pattern, start, fps, out_path):
    cmd = [ffmpeg, "-y", "-framerate", str(fps), "-start_number", str(start),
           "-i", pattern,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-r", str(fps), out_path]
    print("[pv-render] encoding mp4: " + " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd).returncode


def _assemble_gif(ffmpeg, pattern, start, fps, out_path):
    # Two-pass palettegen/paletteuse for decent color.
    palette = out_path + ".palette.png"
    p1 = [ffmpeg, "-y", "-framerate", str(fps), "-start_number", str(start),
          "-i", pattern, "-vf", f"fps={fps},palettegen", palette]
    print("[pv-render] gif palette: " + " ".join(shlex.quote(c) for c in p1))
    rc = subprocess.run(p1).returncode
    if rc != 0:
        return rc
    p2 = [ffmpeg, "-y", "-framerate", str(fps), "-start_number", str(start),
          "-i", pattern, "-i", palette,
          "-lavfi", f"fps={fps}[x];[x][1:v]paletteuse", out_path]
    print("[pv-render] gif encode: " + " ".join(shlex.quote(c) for c in p2))
    rc = subprocess.run(p2).returncode
    try:
        os.remove(palette)
    except OSError:
        pass
    return rc


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="pv_render.py",
        description="Render one Phonon Vision job headless via the installed "
                    "Blender extension.")
    p.add_argument("job", help="job file (.yaml/.yml/.json)")
    p.add_argument("--blender", help="path to the Blender executable "
                                     "(overrides $PV_BLENDER and PATH)")
    p.add_argument("-o", "--output", help="override output.path")
    p.add_argument("--samples", type=int, help="override render.cycles_samples")
    p.add_argument("--device", choices=["CPU", "GPU", "AUTO"],
                   help="override render.device (headless render device)")
    p.add_argument("--gpu-backend", choices=["AUTO", "OPTIX", "CUDA"],
                   help="override render.gpu_backend")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved command + temp job, then exit "
                        "without launching Blender")
    p.add_argument("--keep-temp", action="store_true",
                   help="keep the generated temp JSON instead of deleting it")
    p.add_argument("--clean-frames", action="store_true",
                   help="delete the PNG frame sequence after a SUCCESSFUL "
                        "mp4/gif encode (default: keep frames as a safety net)")
    args = p.parse_args(argv)

    # 1. load
    job = load_job(args.job)
    if not isinstance(job, dict):
        die("job did not parse to a mapping/object")

    # 2. validate (fast, before the slow Blender startup)
    sv = job.get("schema_version")
    if sv is None:
        warn("schema_version missing; assuming 1")
    elif sv not in (1, 2):
        warn(f"schema_version is {sv!r}, expected 1 or 2; proceeding anyway")

    inp = (job.get("input") or {}).get("file")
    if not inp:
        die("job is missing input.file")
    if not os.path.isfile(inp):
        die(f"input file not found: {inp}")

    # 3. overrides
    out = job.setdefault("output", {})
    if args.output:
        out["path"] = args.output
    if args.samples is not None:
        job.setdefault("render", {})["cycles_samples"] = args.samples
    if args.device is not None:
        job.setdefault("render", {})["device"] = args.device
    if args.gpu_backend is not None:
        job.setdefault("render", {})["gpu_backend"] = args.gpu_backend

    if not out.get("path"):
        die("job is missing output.path (and no -o/--output given)")

    # Reconcile the requested container: a .mp4/.gif -o extension wins; otherwise
    # use output.format. (A .png path or PNG format = still or PNG sequence.)
    animation = bool(out.get("animation", False))
    opath = out["path"]
    ext = os.path.splitext(opath)[1].lower()
    if ext in VIDEO_EXTS:
        container = VIDEO_EXTS[ext]
        final_video = opath
    elif ext == ".png":
        container = "PNG"
        final_video = opath
    else:
        container = (out.get("format") or "PNG").upper()
        if container in ("MP4", "GIF"):
            final_video = os.path.splitext(opath)[0] + _CONTAINER_EXT[container]
        else:
            final_video = opath

    fps = int((job.get("loop") or {}).get("fps", 30))
    need_assembly = animation and container in ("MP4", "GIF")
    frames_dir = None

    if need_assembly:
        # Frames-first: tell the inner to render a clean PNG sequence into
        # <video_stem>_frames/, then we assemble <final_video> with ffmpeg.
        vid_dir = os.path.dirname(os.path.abspath(final_video))
        stem = os.path.splitext(os.path.basename(final_video))[0]
        frames_dir = os.path.join(vid_dir, stem + "_frames")
        out["path"] = os.path.join(frames_dir, "frame_")   # sequence prefix
        out["format"] = "PNG"
        out["animation"] = True

    # 5. resolve Blender, 6. resolve pv_inner.py (sibling)
    blender = resolve_blender(args.blender)
    here = os.path.dirname(os.path.abspath(__file__))
    pv_inner = os.path.join(here, "pv_inner.py")
    if not os.path.isfile(pv_inner):
        die(f"pv_inner.py not found next to pv_render.py (expected {pv_inner})")

    # 4. write temp JSON (Blender's Python has no PyYAML)
    fd, temp_json = tempfile.mkstemp(suffix=".json", prefix="pvjob_")
    with os.fdopen(fd, "w") as fh:
        json.dump(job, fh, indent=2)

    # 7. exact Phase-1 command — NO --factory-startup (relies on the installed
    #    extension, which the inner script discovers).
    cmd = [blender, "--background", "--python", pv_inner, "--", "--job", temp_json]

    try:
        if args.dry_run:
            print("[pv-render] dry-run — would launch:")
            print("  " + " ".join(shlex.quote(c) for c in cmd))
            if need_assembly:
                print(f"[pv-render] then assemble {container} -> {final_video} "
                      f"from {frames_dir} @ {fps}fps")
            print(f"[pv-render] temp job written to: {temp_json}")
            if not args.keep_temp:
                print("  (deleted on exit; pass --keep-temp to retain)")
            print(json.dumps(job, indent=2))
            return 0

        # 8. launch, inheriting stdout/stderr so [pv] lines + render progress
        #    stream live.
        print(f"[pv-render] launching: {blender}", file=sys.stderr)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            return rc   # Blender failed — propagate; don't try to assemble.

        if not need_assembly:
            return 0    # still PNG, or PNG sequence — nothing to assemble.

        # 2 (outer). Assemble the frame sequence into the final video.
        seq = _find_frame_sequence(frames_dir)
        if seq is None:
            die(f"no frames found in {frames_dir} to assemble", code=4)
        pattern, start, count = seq

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            # Frames survive — the user re-encodes without re-rendering.
            die(f"frames rendered to {frames_dir}; ffmpeg not found to assemble "
                f"the video — `module load FFmpeg` on the cluster, or install "
                f"ffmpeg locally, then re-run.", code=3)

        print(f"[pv-render] assembling {count} frames -> {final_video}")
        if container == "MP4":
            arc = _assemble_mp4(ffmpeg, pattern, start, fps, final_video)
        else:
            arc = _assemble_gif(ffmpeg, pattern, start, fps, final_video)
        if arc != 0:
            die(f"ffmpeg failed (exit {arc}); frames preserved in {frames_dir}",
                code=4)

        print(f"[pv-render] OK wrote: {final_video}")
        if args.clean_frames:
            import shutil as _sh
            try:
                _sh.rmtree(frames_dir)
                print(f"[pv-render] removed frames dir: {frames_dir}")
            except OSError as e:
                warn(f"could not remove frames dir {frames_dir}: {e}")
        else:
            print(f"[pv-render] frames kept in: {frames_dir} "
                  f"(use --clean-frames to delete)")
        return 0
    finally:
        # 9. clean up the temp JSON unless asked to keep it.
        if not args.keep_temp:
            try:
                os.remove(temp_json)
            except OSError:
                pass


if __name__ == "__main__":
    # Exit with Blender's return code so failures propagate (HPC job arrays).
    sys.exit(main())
