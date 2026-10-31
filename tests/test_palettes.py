"""Tests for the element-color palettes (Jmol, VESTA) and the registry.

The motivating bug: phonon-core historically exposed only the Jmol CPK
table but under a misnamed ``vesta_color()`` accessor, so "VESTA" actually
rendered Jmol colors. 0.5.3 split these into ``jmol_color()`` (Jmol CPK)
and a genuinely-different ``vesta_color()`` (real VESTA elements.ini).

These tests assert the palettes are genuinely distinct (the canonical
example being carbon: Jmol gray vs VESTA brown), that every accessor
returns a valid RGB triple, and that out-of-table elements never raise.
"""
import pytest

from phonon_core.radii import (
    JMOL_COLOR_RGB,
    VESTA_COLOR_RGB,
    PALETTES,
    jmol_color,
    vesta_color,
    _DEFAULT_COLOR,
)


def _is_valid_rgb(c) -> bool:
    return (
        isinstance(c, tuple)
        and len(c) == 3
        and all(isinstance(x, float) for x in c)
        and all(0.0 <= x <= 1.0 for x in c)
    )


# ---------------------------------------------------------------------------
# Distinctness: Jmol and VESTA must be genuinely different palettes
# ---------------------------------------------------------------------------
def test_carbon_jmol_gray_vesta_brown():
    """The canonical distinction the whole 0.5.3 change is about."""
    jmol_c = jmol_color("C")
    vesta_c = vesta_color("C")
    # Jmol carbon is gray: all three channels roughly equal and mid-valued
    assert jmol_c == pytest.approx((0.565, 0.565, 0.565), abs=1e-3)
    assert max(jmol_c) - min(jmol_c) < 0.02, "Jmol C should be gray"
    # VESTA carbon is brown: red > green > blue, warm tone
    assert vesta_c == pytest.approx((0.504, 0.287, 0.162), abs=1e-3)
    assert vesta_c[0] > vesta_c[1] > vesta_c[2], "VESTA C should be brown"
    # And they must differ
    assert jmol_c != vesta_c


def test_palettes_not_identical():
    """At least several common elements differ between Jmol and VESTA.

    If these were identical we'd be back to the original mislabel bug.
    """
    common = set(JMOL_COLOR_RGB) & set(VESTA_COLOR_RGB)
    differing = [el for el in common if JMOL_COLOR_RGB[el] != VESTA_COLOR_RGB[el]]
    # Empirically dozens differ; require a healthy margin so an accidental
    # alias (vesta -> jmol) can't sneak back in.
    assert len(differing) >= 20, (
        f"Only {len(differing)} elements differ between Jmol and VESTA; "
        "expected many. Did vesta_color get aliased back to Jmol?"
    )


def test_sn_se_differ_between_palettes():
    """SnSe2 is the user's visual-distinctness test material."""
    for el in ("Sn", "Se"):
        assert jmol_color(el) != vesta_color(el), (
            f"{el} must look different in Jmol vs VESTA for the SnSe2 test"
        )


# ---------------------------------------------------------------------------
# No-raise on unknown elements + documented mid-grey fallback
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("accessor", [jmol_color, vesta_color])
@pytest.mark.parametrize("unknown", ["Xx", "", "Zz", "NotAnElement", "uuo"])
def test_accessors_never_raise_on_unknown(accessor, unknown):
    c = accessor(unknown)
    assert c == _DEFAULT_COLOR
    assert _is_valid_rgb(c)


@pytest.mark.parametrize("accessor", [jmol_color, vesta_color])
def test_accessor_fallback_is_mid_grey(accessor):
    assert accessor("Xx") == (0.5, 0.5, 0.5)


# ---------------------------------------------------------------------------
# Valid RGB triples throughout both tables
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table", [JMOL_COLOR_RGB, VESTA_COLOR_RGB])
def test_all_entries_are_valid_rgb(table):
    for el, c in table.items():
        assert _is_valid_rgb(c), f"{el}: {c} is not a valid [0,1] RGB triple"


def test_known_elements_return_real_values_not_fallback():
    """Common elements must be in-table (not silently hitting the fallback)."""
    for el in ("H", "C", "N", "O", "Fe", "Sn", "Se", "W"):
        assert jmol_color(el) != _DEFAULT_COLOR or el == "?"
        assert vesta_color(el) != _DEFAULT_COLOR


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_palettes_registry_maps_to_accessors():
    assert PALETTES["jmol"] is jmol_color
    assert PALETTES["vesta"] is vesta_color


def test_registry_accessors_callable_and_valid():
    for name, accessor in PALETTES.items():
        c = accessor("C")
        assert _is_valid_rgb(c), f"{name} returned invalid RGB for C"


def test_vmd_deferred_not_in_registry():
    """VMD was deferred to 0.5.4 (no authoritative source). Guard against a
    half-wired entry that would KeyError or return wrong colors."""
    assert "vmd" not in PALETTES


# ---------------------------------------------------------------------------
# Real-VESTA spot checks against VESTA's own elements.ini (head H..F)
# ---------------------------------------------------------------------------
def test_vesta_matches_elements_ini_head():
    """First nine elements, transcribed from VESTA's own elements.ini.

    The shipped table is sourced from pymatgen's VESTA scheme (8-bit ints
    /255), so values agree with elements.ini's higher-precision floats only
    to within 8-bit rounding on both sides (~2/255 ~= 0.008). Carbon is the exception: it
    is overridden to elements.ini's exact float and is checked tightly in
    test_carbon_jmol_gray_vesta_brown.
    """
    expected = {  # from VESTA/.../elements.ini (prtkm.github.io transcription)
        "H":  (1.00000, 0.80000, 0.80000),
        "He": (0.98907, 0.91312, 0.81091),
        "Li": (0.52731, 0.87953, 0.45670),
        "Be": (0.37147, 0.84590, 0.48292),
        "B":  (0.12490, 0.63612, 0.05948),
        "C":  (0.50430, 0.28659, 0.16236),
        "N":  (0.69139, 0.72934, 0.90280),
        "O":  (0.99997, 0.01328, 0.00000),
        "F":  (0.69139, 0.72934, 0.90280),
    }
    for el, ref in expected.items():
        got = vesta_color(el)
        assert got == pytest.approx(ref, abs=8e-3), f"{el}: {got} vs {ref}"
