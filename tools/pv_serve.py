#!/usr/bin/env python3
"""pv-serve (0.6.58) — a localhost dev server for building phonon-vision navigator pages.

A small, dependency-free backbone: a POST /build endpoint accepts a multipart upload of
phonopy outputs (band.hdf5 [+ optional mesh.hdf5 / DOS files / phonopy.yaml +
FORCE_CONSTANTS for ENGINE mode, 0.6.83]) plus a few text options, stages them, and runs
the EXISTING exporter (tools/pv_navigator_export.py) with --also-html to produce a
self-contained page under <root>/pv_serve_out/<session>/index.html. Static files
(including the built pages) are served by SimpleHTTPRequestHandler. When phonopy.yaml is
staged the exporter runs with --engine (exact D(q) payload: provenance + mesh_modes);
the exporter remains the single validator — its strict fail modes (exit 2 no phonopy /
exit 3 FC<->band cross-check) flow back to the client verbatim via captured stderr.

Design constraints (deliberate):
  * Stdlib only — no Flask/werkzeug/multipart, and NO cgi/cgitb (removed in 3.13). The
    multipart payload is parsed at the BYTE level so binary HDF5 is never text-mangled.
  * The exporter is IMPORTED as the single source of build logic; this module contains
    zero copied export code — it only assembles argv and calls pne.main().
  * Binds 127.0.0.1 only. This is a localhost developer tool, not a public server.

CLI:  python3 tools/pv_serve.py [--port 8000] [--root .]
"""
import contextlib
import http.server
import io
import socketserver
import json
import re
import shutil
import sys
import uuid
import pathlib

# CHANGE A — import the exporter as the single source of build logic (never reimplement).
_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import pv_navigator_export as pne  # noqa: E402

# Resolved at server start from --root; tests pass serve_root explicitly to handle_build().
SERVE_ROOT = pathlib.Path(".").resolve()

# Bound pv_serve_out growth: keep the N newest builds, delete older. <=0 disables.
DEFAULT_MAX_SESSIONS = 20
MAX_SESSIONS = DEFAULT_MAX_SESSIONS   # set in main() from --max-sessions; read by do_POST

# Fixed on-disk slot names for each recognized upload (the actual client filename is
# irrelevant — only the form field NAME selects the slot).
_FILE_SLOTS = {
    "band": "band.hdf5",
    "mesh": "mesh.hdf5",
    "total_dos": "total_dos.dat",
    "proj_dos": "proj_dos.dat",
    "pdos_conf": "pdos.conf",
    # 0.6.83 engine mode: canonical names are exactly what the 0.6.80 engine's
    # sibling discovery accepts (load_engine looks for FORCE_CONSTANTS /
    # force_constants.hdf5 NEXT TO phonopy.yaml — the session dir provides
    # "next to" for free). force_constants has ONE save-name exception below:
    # an .hdf5 upload is saved as force_constants.hdf5, because hdf5 bytes
    # under the text name would poison the parse.
    "engine_yaml": "phonopy.yaml",
    "force_constants": "FORCE_CONSTANTS",
}
_TEXT_DEFAULTS = {"group_a": "Sn", "dim": "2", "grid": "96"}


def _unquote(v):
    v = v.strip()
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        return v[1:-1]
    return v


def _parse_disposition(header_blob):
    """From a decoded (latin-1) header blob, return (name, filename) for
    Content-Disposition. A `filename=` parameter (even empty) marks the part as
    a FILE (filename is a str, possibly ""); filename None means a TEXT field.
    Uploads still land in fixed slots — the client filename's ONLY use (0.6.83)
    is the force_constants .hdf5-vs-text save-name rule in handle_build."""
    name, filename = None, None
    for line in header_blob.split("\r\n"):
        if line[:20].lower() == "content-disposition:":
            for seg in line.split(";"):
                seg = seg.strip()
                low = seg.lower()
                if low.startswith("name="):
                    name = _unquote(seg[5:])
                elif low.startswith("filename="):
                    filename = _unquote(seg[9:])
            break
    return name, filename


def _parse_multipart(content_type, body):
    """Byte-level multipart/form-data parse (NO cgi, NO text-decoding of the payload).
    Returns (files: name->bytes, fields: name->str, fnames: name->client filename str)."""
    files, fields, fnames = {}, {}, {}
    boundary = None
    for seg in content_type.split(";"):
        seg = seg.strip()
        if seg.lower().startswith("boundary="):
            boundary = _unquote(seg[len("boundary="):])
            break
    if not boundary:
        return files, fields, fnames
    delim = b"--" + boundary.encode("latin-1")
    chunks = body.split(delim)
    # Drop the preamble (chunks[0]) and the closing epilogue (chunks[-1] == b"--\r\n").
    for chunk in chunks[1:-1]:
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]                       # strip one leading CRLF
        raw_headers, sep, content = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        if content.endswith(b"\r\n"):
            content = content[:-2]                  # strip the trailing CRLF
        name, filename = _parse_disposition(raw_headers.decode("latin-1"))
        if name is None:
            continue
        if filename is not None:
            files[name] = content                  # raw bytes, untouched
            fnames[name] = filename
        else:
            fields[name] = content.decode("utf-8")
    return files, fields, fnames


def _prune_sessions(serve_root, max_sessions):
    """Keep only the `max_sessions` newest uuid4().hex session dirs under
    serve_root/pv_serve_out; delete older ones. <=0 disables. ONLY ever removes
    32-hex-named direct children of pv_serve_out (a non-session file/dir is never touched),
    and since the just-built session is the newest it is always retained when max >= 1."""
    if max_sessions <= 0:
        return
    base = pathlib.Path(serve_root) / "pv_serve_out"
    if not base.is_dir():
        return
    sessions = [d for d in base.iterdir()
                if d.is_dir() and re.fullmatch(r"[0-9a-f]{32}", d.name)]
    sessions.sort(key=lambda d: d.stat().st_mtime, reverse=True)   # newest first
    for d in sessions[max_sessions:]:
        shutil.rmtree(d, ignore_errors=True)


def handle_build(content_type, body, serve_root, max_sessions=DEFAULT_MAX_SESSIONS):
    """Pure, testable build: parse the multipart payload, stage present files under fixed
    slots, assemble the exporter argv (optional inputs included ONLY when present), and run
    pne.main with --also-html. Returns (http_status:int, resp:dict)."""
    serve_root = pathlib.Path(serve_root)
    files, fields, fnames = _parse_multipart(content_type, body)

    if "band" not in files:
        return (400, {"ok": False, "error": "band hdf5 is required"})
    # 0.6.83: an FC file with nothing to attach it to is a user mistake — name it.
    # (engine_yaml WITHOUT force_constants is VALID: phonopy.yaml may embed the
    # force constants; the exporter is the validator.)
    if "force_constants" in files and "engine_yaml" not in files:
        return (400, {"ok": False,
                      "error": "FORCE_CONSTANTS requires phonopy.yaml (engine mode)"})

    session = uuid.uuid4().hex
    out_dir = serve_root / "pv_serve_out" / session
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage every PRESENT recognized file under its fixed slot name (raw bytes).
    slot_path = {}
    for field, slot in _FILE_SLOTS.items():
        if field in files:
            # force_constants ONLY: an .hdf5 upload keeps the hdf5 spelling —
            # both names are in load_engine's sibling discovery, and hdf5 bytes
            # under the text name would poison the parse.
            if field == "force_constants" and \
                    fnames.get(field, "").lower().endswith(".hdf5"):
                slot = "force_constants.hdf5"
            p = out_dir / slot
            p.write_bytes(files[field])
            slot_path[field] = p
    out_json = out_dir / "navigator_data.json"

    group_a = fields.get("group_a", _TEXT_DEFAULTS["group_a"])
    dim = fields.get("dim", _TEXT_DEFAULTS["dim"])
    grid = fields.get("grid", _TEXT_DEFAULTS["grid"])

    # Assemble argv in the exporter's exact order; missing optional inputs are omitted.
    argv = [str(slot_path["band"])]
    if "mesh" in slot_path:
        argv += [str(slot_path["mesh"])]
    argv += ["-o", str(out_json),
             "--group-a", group_a, "--grid", grid, "--dim", dim]
    if "total_dos" in slot_path:
        argv += ["--total-dos", str(slot_path["total_dos"])]
    if "proj_dos" in slot_path:
        argv += ["--proj-dos", str(slot_path["proj_dos"])]
    if "pdos_conf" in slot_path:
        argv += ["--pdos-conf", str(slot_path["pdos_conf"])]
    # 0.6.83 engine mode: staged phonopy.yaml -> the exporter runs with --engine
    # (FORCE_CONSTANTS discovered as its session-dir sibling by load_engine).
    if "engine_yaml" in slot_path:
        argv += ["--engine", str(slot_path["engine_yaml"])]
    argv += ["--also-html"]

    # Capture the exporter's stderr so its strict fail modes reach the CLIENT:
    # _die prints the real message (e.g. the cross-check max|dω| line) to stderr
    # and raises SystemExit(code) — str(SystemExit(3)) is just "3", useless to a
    # user. The captured text is re-emitted to the server console either way.
    err_buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(err_buf):
            pne.main(argv)
    except (Exception, SystemExit) as exc:
        captured = err_buf.getvalue()
        if captured:
            sys.stderr.write(captured)
        msg = captured.strip() or str(exc) or "build produced no index.html"
        return (500, {"ok": False, "error": msg})
    captured = err_buf.getvalue()
    if captured:
        sys.stderr.write(captured)      # informational lines still reach the console
    if not (out_dir / "index.html").exists():
        return (500, {"ok": False, "error": "build produced no index.html"})
    _prune_sessions(serve_root, max_sessions)   # bound pv_serve_out growth (newest kept)
    return (200, {"ok": True, "url": "/pv_serve_out/%s/index.html" % session})


class PVHandler(http.server.SimpleHTTPRequestHandler):
    """Static file server under SERVE_ROOT (incl. built pv_serve_out/<id>/ pages, with the
    parent's MIME handling + path-traversal protection) plus a POST /build endpoint."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SERVE_ROOT), **kwargs)

    def _send_json(self, status, obj):
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_bytes(self, status, data, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path == "/build":
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            status, resp = handle_build(self.headers.get("Content-Type", ""), body, SERVE_ROOT, MAX_SESSIONS)
            self._send_json(status, resp)
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path in ("/", "/pv_launcher.html"):
            launcher = _HERE.parent / "web" / "pv_launcher.html"   # bundled, independent of --root
            if launcher.exists():
                self._send_bytes(200, launcher.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send_bytes(200, b"pv-serve running; launcher not installed (0.6.58)",
                                 "text/plain; charset=utf-8")
        else:
            super().do_GET()


class _Server(socketserver.TCPServer):
    allow_reuse_address = True


def _parse_cli(argv):
    # --max-sessions N: keep the N newest builds under pv_serve_out, delete older; <=0 disables
    port, root, max_sessions = 8000, ".", DEFAULT_MAX_SESSIONS
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1]); i += 2
        elif a == "--root" and i + 1 < len(argv):
            root = argv[i + 1]; i += 2
        elif a == "--max-sessions" and i + 1 < len(argv):
            max_sessions = int(argv[i + 1]); i += 2
        else:
            i += 1
    return port, root, max_sessions


def main(argv=None):
    global SERVE_ROOT, MAX_SESSIONS
    if argv is None:
        argv = sys.argv[1:]
    port, root, max_sessions = _parse_cli(argv)
    SERVE_ROOT = pathlib.Path(root).resolve()
    MAX_SESSIONS = max_sessions
    httpd = _Server(("127.0.0.1", port), PVHandler)
    print("pv-serve on http://127.0.0.1:%d/  (root=%s, max_sessions=%d)" % (port, SERVE_ROOT, MAX_SESSIONS))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
