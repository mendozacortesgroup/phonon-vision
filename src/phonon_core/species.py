"""Color strategies for displaying per-mode atomic character.

The number of distinct species in the material determines which strategy is
sensible:

- **1 species**: no projection information to display — use a neutral color.
- **2 species**: classic diverging colormap (e.g. seismic, blue↔red). The
  scalar input is the fractional weight of species A; 0.5 → white (mixed).
- **3 species**: barycentric / ternary coloring. Each species gets a corner
  color (default RGB primaries); the displayed color is the weighted mix.
- **N ≥ 4 species**: categorical / dominant-species coloring. Pick the
  winning species' color and modulate its saturation by how dominant it is.
  Equal mixes become grey; pure modes are fully saturated.

All functions here return RGB triples in ``[0, 1]^3``. RGBA is left to the
plotting layer.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

# ============================================================
# Palettes
# ============================================================
# Okabe-Ito: 7-color, colorblind-safe, perceptually distinct.
OKABE_ITO: list[tuple[float, float, float]] = [
    (0.902, 0.624, 0.000),   # orange
    (0.337, 0.706, 0.914),   # sky blue
    (0.000, 0.620, 0.451),   # bluish green
    (0.941, 0.894, 0.259),   # yellow
    (0.000, 0.447, 0.698),   # blue
    (0.835, 0.369, 0.000),   # vermillion
    (0.800, 0.475, 0.655),   # reddish purple
]

# Paul Tol's Vibrant — pairs well with Okabe-Ito as an extended set.
TOL_VIBRANT: list[tuple[float, float, float]] = [
    (0.918, 0.435, 0.118),   # orange
    (0.165, 0.467, 0.706),   # blue
    (0.769, 0.106, 0.490),   # magenta
    (0.235, 0.706, 0.294),   # green
    (0.118, 0.118, 0.118),   # grey
    (0.561, 0.380, 0.122),   # brown
    (0.969, 0.612, 0.671),   # pink
]

EXTENDED_PALETTE: list[tuple[float, float, float]] = OKABE_ITO + TOL_VIBRANT


def assign_species_palette(species: Sequence[str]) -> dict[str, tuple[float, float, float]]:
    """Assign a deterministic palette color to each unique species symbol."""
    seen: dict[str, tuple[float, float, float]] = {}
    palette = EXTENDED_PALETTE
    for s in species:
        if s in seen:
            continue
        seen[s] = palette[len(seen) % len(palette)]
    return seen


# ============================================================
# Strategy: 2 species (diverging)
# ============================================================
def color_binary(
    weight_a: float,
    color_a: tuple[float, float, float] = (1.0, 0.0, 0.0),  # red
    color_b: tuple[float, float, float] = (0.0, 0.0, 1.0),  # blue
    color_mid: tuple[float, float, float] = (1.0, 1.0, 1.0),  # white
) -> tuple[float, float, float]:
    """Diverging color: ``weight_a = 0 → color_b``, ``= 0.5 → color_mid``,
    ``= 1 → color_a``. Replicates the seismic colormap default.
    """
    w = float(np.clip(weight_a, 0.0, 1.0))
    if w < 0.5:
        t = w * 2.0
        a = np.array(color_b)
        b = np.array(color_mid)
    else:
        t = (w - 0.5) * 2.0
        a = np.array(color_mid)
        b = np.array(color_a)
    out = (1.0 - t) * a + t * b
    return tuple(float(x) for x in out)  # type: ignore[return-value]


# ============================================================
# Strategy: 3 species (barycentric / ternary)
# ============================================================
def color_ternary(
    w_a: float,
    w_b: float,
    w_c: float,
    color_a: tuple[float, float, float] = (0.835, 0.157, 0.157),  # red
    color_b: tuple[float, float, float] = (0.157, 0.627, 0.157),  # green
    color_c: tuple[float, float, float] = (0.094, 0.467, 0.706),  # blue
) -> tuple[float, float, float]:
    """Barycentric mix of three corner colors weighted by species fractions.

    Normalizes the weights so the function is robust to inputs that don't
    sum to 1.
    """
    total = w_a + w_b + w_c
    if total < 1e-15:
        return (0.5, 0.5, 0.5)
    wa, wb, wc = w_a / total, w_b / total, w_c / total
    out = wa * np.array(color_a) + wb * np.array(color_b) + wc * np.array(color_c)
    return tuple(float(x) for x in out)  # type: ignore[return-value]


# ============================================================
# Strategy: N ≥ 4 species (categorical with dominance saturation)
# ============================================================
def color_categorical(
    weights: dict[str, float],
    species_palette: dict[str, tuple[float, float, float]],
    *,
    background: tuple[float, float, float] = (0.85, 0.85, 0.85),
) -> tuple[float, float, float]:
    """Pick the dominant species' color, modulate saturation by dominance.

    The saturation s ∈ [0, 1] is linearly mapped from the dominance excess::

        s = max(0, (max_weight - 1/N_species) / (1 - 1/N_species))

    where ``N_species`` is the number of species with nonzero weight. So:

    - An equal N-way mix has ``max_weight = 1/N`` → s = 0 → background color.
    - A 100% pure mode has ``max_weight = 1`` → s = 1 → full palette color.

    Parameters
    ----------
    weights : dict[str, float]
        Species → normalized weight (should sum to 1, but normalized internally).
    species_palette : dict[str, (r, g, b)]
        Palette color per species.
    background : (r, g, b)
        Used for fully mixed modes (s = 0); default neutral grey.
    """
    if not weights:
        return background

    total = sum(weights.values())
    if total < 1e-15:
        return background
    norm = {k: v / total for k, v in weights.items()}

    winner = max(norm, key=norm.get)  # type: ignore[arg-type]
    w_max = norm[winner]
    n = len(norm)
    even = 1.0 / n
    if w_max <= even + 1e-9:
        return background

    # Saturation from dominance excess
    s = (w_max - even) / (1.0 - even)
    s = float(np.clip(s, 0.0, 1.0))

    palette_color = species_palette.get(winner, (0.5, 0.5, 0.5))
    fg = np.array(palette_color)
    bg = np.array(background)
    out = (1.0 - s) * bg + s * fg
    return tuple(float(x) for x in out)  # type: ignore[return-value]


# ============================================================
# Strategy dispatch
# ============================================================
ColorStrategy = Callable[..., tuple[float, float, float]]


def get_strategy(num_species: int):
    """Return the appropriate coloring strategy for the species count."""
    if num_species <= 1:
        return lambda *args, **kwargs: (0.5, 0.5, 0.5)
    if num_species == 2:
        return color_binary
    if num_species == 3:
        return color_ternary
    return color_categorical
