"""Tests for tools/pv_serve.py — the multipart /build backbone.

No real HDF5 and no live socket: pne.main is monkeypatched with a recorder that captures
the assembled argv and writes a placeholder index.html into dirname(-o), so a successful
build is detectable. handle_build() is called directly with synthetic multipart bodies."""
import sys
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pv_serve            # noqa: E402
import pv_navigator_export as pne  # noqa: E402


def _multipart(parts):
    """parts: list of (name, filename_or_None, content[str|bytes]). filename=None => text
    field, else file. Returns (content_type, body_bytes)."""
    boundary = "----pvtestboundaryZZ42"
    delim = ("--" + boundary).encode("latin-1")
    out = b""
    for name, filename, content in parts:
        body = content if isinstance(content, bytes) else content.encode("utf-8")
        out += delim + b"\r\n"
        if filename is not None:
            disp = 'Content-Disposition: form-data; name="%s"; filename="%s"' % (name, filename)
            out += disp.encode("latin-1") + b"\r\n"
            out += b"Content-Type: application/octet-stream\r\n\r\n"
            out += body + b"\r\n"
        else:
            disp = 'Content-Disposition: form-data; name="%s"' % name
            out += disp.encode("latin-1") + b"\r\n\r\n" + body + b"\r\n"
    out += delim + b"--\r\n"
    return "multipart/form-data; boundary=" + boundary, out


def _session_of(resp):
    # url == "/pv_serve_out/<session>/index.html"
    return resp["url"].split("/")[2]


@pytest.fixture
def recorder(monkeypatch):
    """Replace pne.main with a recorder: capture argv + write a placeholder index.html
    into dirname(-o) so handle_build's success check passes."""
    calls = []

    def fake_main(argv):
        calls.append(list(argv))
        out_json = argv[argv.index("-o") + 1]
        out_dir = pathlib.Path(out_json).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "index.html").write_text("<!-- built placeholder -->")

    monkeypatch.setattr(pne, "main", fake_main)
    return calls


def test_T1_multipart_binary_roundtrip(tmp_path, recorder):
    # raw bytes include a CRLF and a 0xFF byte — must survive byte-for-byte (no text mangle)
    raw = b"\x89HDF\r\n\x1a\n\x00\xff payload \r\n end"
    ct, body = _multipart([("band", "band.hdf5", raw)])
    status, resp = pv_serve.handle_build(ct, body, tmp_path)
    assert status == 200 and resp["ok"] is True
    band_path = tmp_path / "pv_serve_out" / _session_of(resp) / "band.hdf5"
    assert band_path.read_bytes() == raw


def test_T2_band_only_argv(tmp_path, recorder):
    ct, body = _multipart([("band", "band.hdf5", b"BANDBYTES")])
    status, resp = pv_serve.handle_build(ct, body, tmp_path)
    assert status == 200 and resp["ok"] is True
    session = _session_of(resp)
    base = tmp_path / "pv_serve_out" / session
    assert recorder[0] == [
        str(base / "band.hdf5"),
        "-o", str(base / "navigator_data.json"),
        "--group-a", "Sn", "--grid", "96", "--dim", "2",
        "--also-html",
    ]
    assert resp["url"] == "/pv_serve_out/%s/index.html" % session


def test_T3_band_mesh_dim3(tmp_path, recorder):
    ct, body = _multipart([
        ("band", "band.hdf5", b"b"),
        ("mesh", "mesh.hdf5", b"m"),
        ("dim", None, "3"),
    ])
    status, resp = pv_serve.handle_build(ct, body, tmp_path)
    assert status == 200 and resp["ok"] is True
    base = tmp_path / "pv_serve_out" / _session_of(resp)
    argv = recorder[0]
    # mesh is the 2nd positional (right after band, before -o)
    assert argv[0] == str(base / "band.hdf5")
    assert argv[1] == str(base / "mesh.hdf5")
    assert argv[2] == "-o"
    assert argv[argv.index("--dim") + 1] == "3", "--dim 3 expected"


def test_T4_all_dos_and_text_fields(tmp_path, recorder):
    ct, body = _multipart([
        ("band", "band.hdf5", b"b"),
        ("total_dos", "total_dos.dat", b"t"),
        ("proj_dos", "proj_dos.dat", b"p"),
        ("pdos_conf", "pdos.conf", b"c"),
        ("group_a", None, "Mo"),
        ("grid", None, "48"),
    ])
    status, resp = pv_serve.handle_build(ct, body, tmp_path)
    assert status == 200 and resp["ok"] is True
    base = tmp_path / "pv_serve_out" / _session_of(resp)
    argv = recorder[0]
    assert argv[argv.index("--total-dos") + 1] == str(base / "total_dos.dat")
    assert argv[argv.index("--proj-dos") + 1] == str(base / "proj_dos.dat")
    assert argv[argv.index("--pdos-conf") + 1] == str(base / "pdos.conf")
    assert argv[argv.index("--group-a") + 1] == "Mo"
    assert argv[argv.index("--grid") + 1] == "48"
    assert argv[argv.index("--dim") + 1] == "2"   # default honored when field absent


def test_T5_no_band_is_400_and_no_build(tmp_path, recorder):
    ct, body = _multipart([("dim", None, "2")])   # band missing
    status, resp = pv_serve.handle_build(ct, body, tmp_path)
    assert status == 400 and resp["ok"] is False
    assert "band" in resp["error"]
    assert recorder == [], "pne.main must NOT be called when band is missing"


def test_T6_build_raises_is_500(tmp_path, monkeypatch):
    def boom(argv):
        raise RuntimeError("synthetic build failure")
    monkeypatch.setattr(pne, "main", boom)
    ct, body = _multipart([("band", "band.hdf5", b"x")])
    status, resp = pv_serve.handle_build(ct, body, tmp_path)
    assert status == 500 and resp["ok"] is False
    assert "synthetic build failure" in resp["error"]


# ---- 0.6.59: do_GET serves the BUNDLED launcher (web/pv_launcher.html), not SERVE_ROOT's ----

def _drive_get(path):
    """Drive PVHandler.do_GET without a socket: build a header-free instance, capture what
    _send_bytes would write."""
    h = pv_serve.PVHandler.__new__(pv_serve.PVHandler)   # skip __init__ (no request processing)
    h.path = path
    cap = {}
    h._send_bytes = lambda status, data, ctype: cap.update(status=status, data=data, ctype=ctype)
    h.do_GET()
    return cap


def test_T7_serves_bundled_launcher_not_serve_root(tmp_path, monkeypatch):
    # a decoy launcher in SERVE_ROOT must NOT be served — do_GET serves the bundled web/ one
    (tmp_path / "pv_launcher.html").write_text("DECOY-SHOULD-NOT-BE-SERVED")
    monkeypatch.setattr(pv_serve, "SERVE_ROOT", tmp_path)
    cap = _drive_get("/")
    assert cap["status"] == 200
    html = cap["data"].decode("utf-8")
    assert "DECOY-SHOULD-NOT-BE-SERVED" not in html, "SERVE_ROOT must NOT be consulted for the launcher"
    for sid in ('id="pv-band"', 'id="pv-mesh"', 'id="pv-total-dos"', 'id="pv-proj-dos"',
                'id="pv-pdos-conf"', 'id="pv-group-a"', 'id="pv-dim"', 'id="pv-grid"', 'id="pv-go"'):
        assert sid in html, "launcher HTML must contain " + sid
    # /pv_launcher.html resolves to the same bundled page
    cap2 = _drive_get("/pv_launcher.html")
    assert cap2["status"] == 200 and 'id="pv-go"' in cap2["data"].decode("utf-8")


def test_T8_missing_bundled_launcher_falls_back_to_stub(tmp_path, monkeypatch):
    # NEG CONTROL: point the resolver (_HERE) where _HERE.parent/web/pv_launcher.html is absent
    monkeypatch.setattr(pv_serve, "_HERE", tmp_path / "tools_decoy")  # parent == tmp_path; no web/ launcher
    cap = _drive_get("/")
    assert cap["status"] == 200 and b"launcher not installed" in cap["data"]
    # restore the real _HERE -> the bundled launcher returns again
    monkeypatch.undo()
    cap2 = _drive_get("/")
    assert b"launcher not installed" not in cap2["data"] and b'id="pv-go"' in cap2["data"]


# ---- 0.6.62: pv_serve_out session pruning (--max-sessions) ----
import os
import time


def _build_once(tmp_path, recorder, max_sessions):
    ct, body = _multipart([("band", "band.hdf5", b"x")])
    status, resp = pv_serve.handle_build(ct, body, tmp_path, max_sessions=max_sessions)
    assert status == 200 and resp["ok"] is True
    return tmp_path / "pv_serve_out" / _session_of(resp)


def test_T9_prune_by_count(tmp_path, recorder):
    d1 = _build_once(tmp_path, recorder, 2)
    d2 = _build_once(tmp_path, recorder, 2)
    # pin deterministic ordering: d1 oldest, d2 middle; d3 (next build) is newest by real mtime
    now = time.time()
    os.utime(d1, (now - 100, now - 100))
    os.utime(d2, (now - 50, now - 50))
    d3 = _build_once(tmp_path, recorder, 2)   # prune runs inside this build
    base = tmp_path / "pv_serve_out"
    remaining = sorted(p.name for p in base.iterdir() if p.is_dir())
    assert len(remaining) == 2, "exactly 2 sessions remain with max_sessions=2"
    assert d3.name in remaining, "the newest/just-built session survives"
    assert d2.name in remaining
    assert d1.name not in remaining, "the oldest session was pruned"


def test_T10_prune_disabled(tmp_path, recorder):
    dirs = [_build_once(tmp_path, recorder, 0) for _ in range(3)]
    base = tmp_path / "pv_serve_out"
    remaining = {p.name for p in base.iterdir() if p.is_dir()}
    assert len(remaining) == 3, "max_sessions=0 disables pruning; all three remain"
    for d in dirs:
        assert d.name in remaining


def test_T11_prune_safety_nonsession(tmp_path, recorder):
    base = tmp_path / "pv_serve_out"
    base.mkdir(parents=True, exist_ok=True)
    keep_dir = base / "keepme"; keep_dir.mkdir()                 # non-session dir
    loose = base / "loose.txt"; loose.write_text("hi")           # loose file
    old_session = base / ("a" * 32); old_session.mkdir()         # a real (32-hex) session, OLD
    (old_session / "index.html").write_text("old")
    os.utime(old_session, (time.time() - 100, time.time() - 100))

    new_session = _build_once(tmp_path, recorder, 1)             # max=1 -> keep newest only
    assert keep_dir.is_dir(), "non-session dir 'keepme' must survive"
    assert loose.is_file(), "loose file must survive"
    assert new_session.is_dir(), "just-built session retained (newest, max>=1)"
    assert not old_session.exists(), "surplus 32-hex session dir removed"


# ---- 0.6.83: the launcher reaches engine mode (the /build <-> exporter argv seam) ----

def test_Teng1_engine_slots_staged_and_argv(tmp_path, recorder):
    fc_raw = b"1 1\r\n  0.1 0.0 0.0\r\n"          # binary-safe roundtrip check
    ct, body = _multipart([
        ("band", "band.hdf5", b"b"),
        ("mesh", "mesh.hdf5", b"m"),
        ("engine_yaml", "phonopy.yaml", b"phonopy:\n  version: x\n"),
        ("force_constants", "FORCE_CONSTANTS", fc_raw),
    ])
    status, resp = pv_serve.handle_build(ct, body, tmp_path)
    assert status == 200 and resp["ok"] is True
    base = tmp_path / "pv_serve_out" / _session_of(resp)
    argv = recorder[0]
    i = argv.index("--engine")
    assert argv[i + 1] == str(base / "phonopy.yaml")
    # canonical save names (load_engine's sibling-discovery contract) + bytes
    assert (base / "phonopy.yaml").read_bytes() == b"phonopy:\n  version: x\n"
    assert (base / "FORCE_CONSTANTS").read_bytes() == fc_raw


def test_Teng2_fc_hdf5_spelling_preserved(tmp_path, recorder):
    ct, body = _multipart([
        ("band", "band.hdf5", b"b"),
        ("engine_yaml", "phonopy.yaml", b"y"),
        ("force_constants", "force_constants.HDF5", b"\x89HDF\x00\xff"),
    ])
    status, resp = pv_serve.handle_build(ct, body, tmp_path)
    assert status == 200
    base = tmp_path / "pv_serve_out" / _session_of(resp)
    assert (base / "force_constants.hdf5").read_bytes() == b"\x89HDF\x00\xff", \
        "hdf5 upload saved under the hdf5 spelling (text name would poison the parse)"
    assert not (base / "FORCE_CONSTANTS").exists()


def test_Teng3_engine_yaml_alone_is_valid(tmp_path, recorder):
    # embedded-FC phonopy.yaml: no FC file needed; the exporter is the validator
    ct, body = _multipart([
        ("band", "band.hdf5", b"b"),
        ("engine_yaml", "phonopy.yaml", b"y"),
    ])
    status, resp = pv_serve.handle_build(ct, body, tmp_path)
    assert status == 200 and resp["ok"] is True
    assert "--engine" in recorder[0]


def test_Teng4_fc_without_yaml_is_400(tmp_path, recorder):
    ct, body = _multipart([
        ("band", "band.hdf5", b"b"),
        ("force_constants", "FORCE_CONSTANTS", b"fc"),
    ])
    status, resp = pv_serve.handle_build(ct, body, tmp_path)
    assert status == 400 and resp["ok"] is False
    assert resp["error"] == "FORCE_CONSTANTS requires phonopy.yaml (engine mode)"
    assert recorder == [], "pne.main must NOT run on the named user mistake"


def test_Teng5_die_stderr_reaches_client(tmp_path, monkeypatch):
    # emulate the exporter's strict fail mode: _die prints the REAL message to
    # stderr then raises SystemExit(3) — str(SystemExit(3)) is just "3", so the
    # relay must forward the captured stderr, not the exception repr.
    def dying_main(argv):
        print("pv_navigator_export: error: engine/band cross-check FAILED — "
              "max|dω| = 8.445e-02 THz (tol 1e-06)", file=sys.stderr)
        raise SystemExit(3)
    monkeypatch.setattr(pne, "main", dying_main)
    ct, body = _multipart([
        ("band", "band.hdf5", b"b"),
        ("engine_yaml", "phonopy.yaml", b"y"),
    ])
    status, resp = pv_serve.handle_build(ct, body, tmp_path)
    assert status == 500 and resp["ok"] is False
    assert "cross-check FAILED" in resp["error"]
    assert "8.445e-02" in resp["error"], \
        "the client sees the real cross-check line, not the exit code"
