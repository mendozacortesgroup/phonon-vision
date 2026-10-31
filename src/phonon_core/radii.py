"""Atomic radii — used for visual bonding heuristics and sphere sizing.

Source: Cordero et al., Dalton Trans., 2008, 2832-2838 (covalent radii from
crystallographic data). Values in Ångström. Missing elements fall back to
the median radius (~1.4 Å).

This is a pure-data module, no dependencies. The Blender addon and the
future web tool both consume the same values so structure visualizations
look consistent across front-ends.
"""

from __future__ import annotations

# Covalent radii in Ångström (Cordero 2008, single-bond values)
COVALENT_RADIUS_ANGSTROM: dict[str, float] = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
    "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15,
    "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98,
    "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89,
    "Tm": 1.90, "Yb": 1.87, "Lu": 1.87,
    "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41,
    "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48,
    "Po": 1.40, "At": 1.50, "Rn": 1.50,
    "Fr": 2.60, "Ra": 2.21,
    "Ac": 2.15, "Th": 2.06, "Pa": 2.00, "U": 1.96, "Np": 1.90, "Pu": 1.87,
    "Am": 1.80, "Cm": 1.69,
}

_DEFAULT_RADIUS = 1.40


def covalent_radius(symbol: str) -> float:
    """Return the covalent radius of ``symbol`` in Å.

    Unknown elements get a sensible median value (1.4 Å) rather than raising,
    so visualization keeps working on exotic systems.
    """
    return COVALENT_RADIUS_ANGSTROM.get(symbol, _DEFAULT_RADIUS)


def bond_cutoff(symbol_a: str, symbol_b: str, *, tolerance: float = 1.2) -> float:
    """Return the maximum distance (Å) below which two atoms of given species
    should be considered bonded.

    Uses ``(r_a + r_b) * tolerance``. The default tolerance of 1.2 is a typical
    value matching VESTA's default — it catches typical bonds while staying
    short enough to exclude second-neighbour distances in most crystals.
    """
    return (covalent_radius(symbol_a) + covalent_radius(symbol_b)) * tolerance


# ============================================================
# VESTA / Jmol element colors
# ============================================================
# RGB triples in [0,1]. Source: Jmol's default CPK palette (which VESTA
# follows for most elements). Hex values from
# https://jmol.sourceforge.net/jscolors/ converted to (r,g,b) ∈ [0,1].
# Unknown elements fall back to mid-grey.
JMOL_COLOR_RGB: dict[str, tuple[float, float, float]] = {
    "H":  (1.000, 1.000, 1.000),
    "He": (0.851, 1.000, 1.000),
    "Li": (0.800, 0.502, 1.000),
    "Be": (0.761, 1.000, 0.000),
    "B":  (1.000, 0.710, 0.710),
    "C":  (0.565, 0.565, 0.565),
    "N":  (0.188, 0.314, 0.973),
    "O":  (1.000, 0.051, 0.051),
    "F":  (0.565, 0.878, 0.314),
    "Ne": (0.702, 0.890, 0.961),
    "Na": (0.671, 0.361, 0.949),
    "Mg": (0.541, 1.000, 0.000),
    "Al": (0.749, 0.651, 0.651),
    "Si": (0.941, 0.784, 0.627),
    "P":  (1.000, 0.502, 0.000),
    "S":  (1.000, 1.000, 0.188),
    "Cl": (0.122, 0.941, 0.122),
    "Ar": (0.502, 0.820, 0.890),
    "K":  (0.561, 0.251, 0.831),
    "Ca": (0.239, 1.000, 0.000),
    "Sc": (0.902, 0.902, 0.902),
    "Ti": (0.749, 0.761, 0.780),
    "V":  (0.651, 0.651, 0.671),
    "Cr": (0.541, 0.600, 0.780),
    "Mn": (0.612, 0.478, 0.780),
    "Fe": (0.878, 0.400, 0.200),
    "Co": (0.941, 0.565, 0.627),
    "Ni": (0.314, 0.816, 0.314),
    "Cu": (0.784, 0.502, 0.200),
    "Zn": (0.490, 0.502, 0.690),
    "Ga": (0.761, 0.561, 0.561),
    "Ge": (0.400, 0.561, 0.561),
    "As": (0.741, 0.502, 0.890),
    "Se": (1.000, 0.631, 0.000),
    "Br": (0.651, 0.161, 0.161),
    "Kr": (0.361, 0.722, 0.820),
    "Rb": (0.439, 0.180, 0.690),
    "Sr": (0.000, 1.000, 0.000),
    "Y":  (0.580, 1.000, 1.000),
    "Zr": (0.580, 0.878, 0.878),
    "Nb": (0.451, 0.761, 0.788),
    "Mo": (0.329, 0.710, 0.710),
    "Tc": (0.231, 0.620, 0.620),
    "Ru": (0.141, 0.561, 0.561),
    "Rh": (0.039, 0.490, 0.549),
    "Pd": (0.000, 0.412, 0.522),
    "Ag": (0.753, 0.753, 0.753),
    "Cd": (1.000, 0.851, 0.561),
    "In": (0.651, 0.459, 0.451),
    "Sn": (0.400, 0.502, 0.502),
    "Sb": (0.620, 0.388, 0.710),
    "Te": (0.831, 0.478, 0.000),
    "I":  (0.580, 0.000, 0.580),
    "Xe": (0.259, 0.620, 0.690),
    "Cs": (0.341, 0.090, 0.561),
    "Ba": (0.000, 0.788, 0.000),
    "La": (0.439, 0.831, 1.000),
    "Ce": (1.000, 1.000, 0.780),
    "Pr": (0.851, 1.000, 0.780),
    "Nd": (0.780, 1.000, 0.780),
    "Pm": (0.639, 1.000, 0.780),
    "Sm": (0.561, 1.000, 0.780),
    "Eu": (0.380, 1.000, 0.780),
    "Gd": (0.271, 1.000, 0.780),
    "Tb": (0.188, 1.000, 0.780),
    "Dy": (0.122, 1.000, 0.780),
    "Ho": (0.000, 1.000, 0.612),
    "Er": (0.000, 0.902, 0.459),
    "Tm": (0.000, 0.831, 0.322),
    "Yb": (0.000, 0.749, 0.220),
    "Lu": (0.000, 0.671, 0.141),
    "Hf": (0.302, 0.761, 1.000),
    "Ta": (0.302, 0.651, 1.000),
    "W":  (0.129, 0.580, 0.839),
    "Re": (0.149, 0.490, 0.671),
    "Os": (0.149, 0.400, 0.588),
    "Ir": (0.090, 0.329, 0.529),
    "Pt": (0.816, 0.816, 0.878),
    "Au": (1.000, 0.820, 0.137),
    "Hg": (0.722, 0.722, 0.816),
    "Tl": (0.651, 0.329, 0.302),
    "Pb": (0.341, 0.349, 0.380),
    "Bi": (0.620, 0.310, 0.710),
    "Po": (0.671, 0.361, 0.000),
    "At": (0.459, 0.310, 0.271),
    "Rn": (0.259, 0.510, 0.588),
    "Fr": (0.259, 0.000, 0.400),
    "Ra": (0.000, 0.490, 0.000),
    "Ac": (0.439, 0.671, 0.980),
    "Th": (0.000, 0.729, 1.000),
    "Pa": (0.000, 0.631, 1.000),
    "U":  (0.000, 0.561, 1.000),
    "Np": (0.000, 0.502, 1.000),
    "Pu": (0.000, 0.420, 1.000),
}

_DEFAULT_COLOR = (0.500, 0.500, 0.500)


# ============================================================
# Real VESTA element colors (distinct from Jmol)
# ============================================================
# VESTA's default element colors are NOT the same as Jmol's CPK palette.
# The clearest example is carbon: Jmol C is gray (0.565, 0.565, 0.565),
# VESTA C is brown (0.504, 0.287, 0.162). VESTA stores its defaults in an
# "elements.ini" file (columns: Z, symbol, atomic/vdW/ionic radii, R, G, B
# as floats in [0,1]).
#
# Source / provenance:
#   - Base table: pymatgen's ElementColorSchemes.yaml, "VESTA" scheme
#     (RGB 0-255), pymatgen 2026.5.4, file pymatgen/vis/ElementColorSchemes.yaml.
#     Converted to [0,1] by /255 and rounded to 3 decimals. pymatgen's
#     "VESTA" scheme transcribes VESTA's elements.ini.
#   - Carbon correction: pymatgen's VESTA scheme diverges from real VESTA
#     ONLY at carbon (it lists C as gray [76,76,76]). Verified against
#     VESTA's own elements.ini (head: H..F), where 8/9 sampled elements
#     matched pymatgen within +/-2/255 and only C differed. C is therefore
#     overridden to VESTA elements.ini's true brown (0.504, 0.287, 0.162),
#     transcribed from VESTA/VESTA.app/Contents/Resources/elements.ini as
#     reproduced at https://prtkm.github.io/posts/changing-default-atom-colors-in-vesta-to-asejmol-colors/
#
# Coverage: 109 elements (H through the actinides/early transactinides).
# Unknown elements fall back to mid-grey via vesta_color().
VESTA_COLOR_RGB: dict[str, tuple[float, float, float]] = {
    "H": (1.000, 0.800, 0.800),
    "He": (0.988, 0.910, 0.808),
    "Li": (0.525, 0.875, 0.451),
    "Be": (0.369, 0.843, 0.482),
    "B": (0.122, 0.635, 0.059),
    "C": (0.504, 0.287, 0.162),
    "N": (0.690, 0.725, 0.902),
    "O": (0.996, 0.012, 0.000),
    "F": (0.690, 0.725, 0.902),
    "Ne": (0.996, 0.216, 0.710),
    "Na": (0.976, 0.863, 0.235),
    "Mg": (0.984, 0.482, 0.082),
    "Al": (0.506, 0.698, 0.839),
    "Si": (0.106, 0.231, 0.980),
    "P": (0.753, 0.612, 0.761),
    "S": (1.000, 0.980, 0.000),
    "Cl": (0.192, 0.988, 0.008),
    "Ar": (0.812, 0.996, 0.769),
    "K": (0.631, 0.129, 0.965),
    "Ca": (0.353, 0.588, 0.741),
    "Sc": (0.710, 0.388, 0.671),
    "Ti": (0.471, 0.792, 1.000),
    "V": (0.898, 0.098, 0.000),
    "Cr": (0.000, 0.000, 0.620),
    "Mn": (0.655, 0.031, 0.616),
    "Fe": (0.710, 0.443, 0.000),
    "Co": (0.000, 0.000, 0.686),
    "Ni": (0.718, 0.733, 0.741),
    "Cu": (0.133, 0.278, 0.863),
    "Zn": (0.561, 0.561, 0.506),
    "Ga": (0.620, 0.890, 0.451),
    "Ge": (0.494, 0.431, 0.651),
    "As": (0.455, 0.816, 0.341),
    "Se": (0.604, 0.937, 0.059),
    "Br": (0.494, 0.192, 0.008),
    "Kr": (0.980, 0.757, 0.953),
    "Rb": (0.439, 0.180, 0.690),
    "Sr": (0.000, 1.000, 0.000),
    "Y": (0.580, 1.000, 1.000),
    "Zr": (0.000, 1.000, 0.000),
    "Nb": (0.451, 0.761, 0.788),
    "Mo": (0.329, 0.710, 0.710),
    "Tc": (0.231, 0.620, 0.620),
    "Ru": (0.141, 0.561, 0.561),
    "Rh": (0.039, 0.490, 0.549),
    "Pd": (0.000, 0.412, 0.522),
    "Ag": (0.753, 0.753, 0.753),
    "Cd": (1.000, 0.851, 0.561),
    "In": (0.651, 0.459, 0.451),
    "Sn": (0.604, 0.557, 0.725),
    "Sb": (0.620, 0.388, 0.710),
    "Te": (0.831, 0.478, 0.000),
    "I": (0.580, 0.000, 0.580),
    "Xe": (0.259, 0.620, 0.690),
    "Cs": (0.341, 0.090, 0.561),
    "Ba": (0.000, 0.788, 0.000),
    "La": (0.353, 0.769, 0.286),
    "Ce": (1.000, 1.000, 0.780),
    "Pr": (0.851, 1.000, 0.780),
    "Nd": (0.780, 1.000, 0.780),
    "Pm": (0.639, 1.000, 0.780),
    "Sm": (0.561, 1.000, 0.780),
    "Eu": (0.380, 1.000, 0.780),
    "Gd": (0.271, 1.000, 0.780),
    "Tb": (0.188, 1.000, 0.780),
    "Dy": (0.122, 1.000, 0.780),
    "Ho": (0.000, 1.000, 0.612),
    "Er": (0.000, 0.902, 0.459),
    "Tm": (0.000, 0.831, 0.322),
    "Yb": (0.000, 0.749, 0.220),
    "Lu": (0.000, 0.671, 0.141),
    "Hf": (0.302, 0.761, 1.000),
    "Ta": (0.302, 0.651, 1.000),
    "W": (0.129, 0.580, 0.839),
    "Re": (0.149, 0.490, 0.671),
    "Os": (0.149, 0.400, 0.588),
    "Ir": (0.090, 0.329, 0.529),
    "Pt": (0.816, 0.816, 0.878),
    "Au": (1.000, 0.820, 0.137),
    "Hg": (0.722, 0.722, 0.816),
    "Tl": (0.651, 0.329, 0.302),
    "Pb": (0.341, 0.349, 0.380),
    "Bi": (0.620, 0.310, 0.710),
    "Po": (0.671, 0.361, 0.000),
    "At": (0.459, 0.310, 0.271),
    "Rn": (0.259, 0.510, 0.588),
    "Fr": (0.259, 0.000, 0.400),
    "Ra": (0.000, 0.490, 0.000),
    "Ac": (0.439, 0.671, 0.980),
    "Th": (0.000, 0.729, 1.000),
    "Pa": (0.000, 0.631, 1.000),
    "U": (0.000, 0.561, 1.000),
    "Np": (0.000, 0.502, 1.000),
    "Pu": (0.000, 0.420, 1.000),
    "Am": (0.329, 0.361, 0.949),
    "Cm": (0.471, 0.361, 0.890),
    "Bk": (0.541, 0.310, 0.890),
    "Cf": (0.631, 0.212, 0.831),
    "Es": (0.702, 0.122, 0.831),
    "Fm": (0.702, 0.122, 0.729),
    "Md": (0.702, 0.051, 0.651),
    "No": (0.741, 0.051, 0.529),
    "Lr": (0.780, 0.000, 0.400),
    "Rf": (0.800, 0.000, 0.349),
    "Db": (0.820, 0.000, 0.310),
    "Sg": (0.851, 0.000, 0.271),
    "Bh": (0.878, 0.000, 0.220),
    "Hs": (0.902, 0.000, 0.180),
    "Mt": (0.922, 0.000, 0.149),
}


def jmol_color(symbol: str) -> tuple[float, float, float]:
    """Return the Jmol CPK color for ``symbol`` as an (r, g, b) tuple in [0,1].

    Jmol's default CPK palette (carbon gray). Source:
    https://jmol.sourceforge.net/jscolors/ . Coverage: 94 elements.
    Unknown elements fall back to documented mid-grey (0.5, 0.5, 0.5).
    Never raises.
    """
    return JMOL_COLOR_RGB.get(symbol, _DEFAULT_COLOR)


def vesta_color(symbol: str) -> tuple[float, float, float]:
    """Return the real VESTA color for ``symbol`` as an (r, g, b) tuple in [0,1].

    Real VESTA element colors (carbon brown), NOT Jmol's CPK palette. See
    VESTA_COLOR_RGB for provenance. Coverage: 109 elements. Unknown elements
    fall back to documented mid-grey (0.5, 0.5, 0.5). Never raises.
    """
    return VESTA_COLOR_RGB.get(symbol, _DEFAULT_COLOR)


# Registry of element-table palettes (name -> accessor). The add-on uses
# this to stay palette-agnostic: a new element-table palette becomes a
# one-line addition here. ("auto" is NOT here — it is categorical /
# mass-based, not an element-table lookup, and lives in species.py.)
PALETTES: dict[str, "callable"] = {
    "jmol": jmol_color,
    "vesta": vesta_color,
    # "vmd": vmd_color,   # deferred to 0.5.4 pending an authoritative source
}
