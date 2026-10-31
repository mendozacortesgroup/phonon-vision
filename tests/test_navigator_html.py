"""Navigator HTML integrity (regression guard for the 0.6.2 truncation bug).

The 0.6.2 template ended at the viewer IIFE "})();" with no </script></body>
</html>; browsers won't execute an unclosed inline <script>, so the viewer never
ran and the page hung on "Loading…". This structural check — in the spirit of
the project's LaTeX-validation discipline — asserts the template (and a
generated sample) are well-formed, and is verified to FAIL on a truncated file.
"""
import importlib.util
import os
import re

import pytest

_HERE = os.path.dirname(__file__)
_TPL = os.path.join(_HERE, os.pardir, "web", "index.html")


def _check_html_wellformed(text):
    # ignore HTML comments (a comment legitimately mentions "<script src>") and
    # any escaped "<\/script>" inside JS strings (e.g. a document.write payload).
    body = re.sub(r"<!--.*?-->", "", text, flags=re.S).replace("<\\/script>", "")
    opens = len(re.findall(r"<script\b", body))
    closes = len(re.findall(r"</script>", body))
    assert opens == closes, f"unbalanced <script>: {opens} open vs {closes} close"
    assert "</body>" in text, "missing </body>"
    assert "</html>" in text, "missing </html>"
    assert text.rstrip().endswith("</html>"), "file does not end with </html>"


def test_template_html_is_wellformed():
    _check_html_wellformed(open(_TPL, encoding="utf-8").read())


def test_truncated_template_is_rejected():
    """The exact 0.6.2 failure mode (file cut off at the IIFE close) must fail."""
    full = open(_TPL, encoding="utf-8").read()
    truncated = full[:full.rindex("})();") + len("})();")] + "\n"
    with pytest.raises(AssertionError):
        _check_html_wellformed(truncated)


def test_generated_html_is_wellformed_and_self_contained(tmp_path):
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    spec = importlib.util.spec_from_file_location(
        "pv_navigator_export",
        os.path.join(_HERE, os.pardir, "tools", "pv_navigator_export.py"))
    nav = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nav)

    band = os.path.join(_HERE, "fixtures", "phonopy", "SnSe2_band.hdf5")
    mesh = os.path.join(_HERE, "fixtures", "navigator", "SnSe2_mesh_coarse.hdf5")
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    nav.main([band, mesh, "-o", str(tmp_path / "navigator_data.json"),
              "--grid", "8", "--also-html"])

    gen = tmp_path / "index.html"
    assert gen.is_file()
    _check_html_wellformed(gen.read_text(encoding="utf-8"))
    # self-contained: every local asset the page loads is copied (no network)
    for name in ("plotly.min.js", "three.min.js", "TrackballControls.js",
                 "kinematics.js", "navigator_data.js"):
        assert (tmp_path / name).is_file(), name


def _norm(s):
    return re.sub(r"\s+", " ", s)


def test_surface_scene_has_isotropic_inplane_ranges():
    """Regression guard for the anisotropic-BZ stretch (0.6.4): the 3D scene must
    give kx and ky the SAME symmetric range derived from the BZ circumradius, and
    aspectratio.x must equal aspectratio.y, so the manual 1:1 is a true 1:1."""
    text = _norm(open(_TPL, encoding="utf-8").read())
    # circumradius + symmetric extent (margin only) are computed
    assert "Math.max(...bzPoly.map(" in text, "BZ circumradius not computed"
    assert "const kExt = 1.06 * kRad" in text, "symmetric in-plane extent missing"
    # BOTH in-plane axes use the identical symmetric range. Since 0.7.2 the
    # extent is DYNAMIC (kExtNow() = kExt, or the widened extKExt when the
    # extended-zone toggle is on) — the isotropy contract is unchanged: one
    # shared symmetric extent for both axes, whatever its current value.
    assert "const kExtNow = () => (extZones ? extKExt : kExt)" in text
    assert 'xaxis: { title: "kx", range: [-kExtNow(), kExtNow()] }' in text
    assert 'yaxis: { title: "ky", range: [-kExtNow(), kExtNow()] }' in text
    # 1:1:.. aspectratio (x === y)
    assert re.search(r"aspectratio: \{ x: 1, y: 1, z: [0-9.]+ \}", text), \
        "aspectratio.x must equal aspectratio.y"


def test_atoms_panel_single_source_kinematics_and_local_assets():
    """1b-3: the atom panel's per-frame positions come from
    PhononKinematics.displaceMode; the displacement formula is NOT reimplemented
    in index.html (single source). three.js + controls + kinematics are vendored
    as LOCAL classic scripts (no CDN, file://-openable)."""
    text = open(_TPL, encoding="utf-8").read()
    inline = text[text.index("(function ()"):text.rindex("})();") + len("})();")]

    # the proven engine is called, and NO displacement physics is re-rolled here:
    # none of sqrt/cos/sin/eigRe/eigIm/exp( may appear in the inline script (the
    # peak readout uses only the engine + Math.hypot/Math.max). Single source.
    assert "PhononKinematics.displaceMode" in inline
    for tok in ("Math.sqrt", "Math.cos", "Math.sin", "eigRe", "eigIm", "exp("):
        assert tok not in inline, f"forbidden token in inline viewer script: {tok}"

    # vendored local classic scripts, no external loads
    for tag in ('<script src="three.min.js"></script>',
                '<script src="TrackballControls.js"></script>',
                '<script src="kinematics.js"></script>'):
        assert tag in text, tag
    assert "cdn." not in text and 'src="http' not in text

    # the third panel exists and the existing click handler drives it (additive)
    assert 'id="atoms3d"' in text
    assert "selectModeForAnimation(qi, branch)" in inline
