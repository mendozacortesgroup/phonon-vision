"""Frequency unit conversions.

Phonopy outputs frequencies in THz by default. Spectroscopists prefer cm⁻¹,
infrared people prefer meV. All three are linear conversions.

Conversion factors (CODATA 2018):
  1 THz = 33.35640951981520 cm⁻¹
  1 THz = 4.135667696923859 meV
  1 cm⁻¹ = 0.12398419843320026 meV
"""

THZ_TO_CM1 = 33.35640951981520
THZ_TO_MEV = 4.135667696923859

CM1_TO_THZ = 1.0 / THZ_TO_CM1
MEV_TO_THZ = 1.0 / THZ_TO_MEV

CM1_TO_MEV = THZ_TO_MEV / THZ_TO_CM1
MEV_TO_CM1 = 1.0 / CM1_TO_MEV


def thz_to_cm1(thz: float) -> float:
    return thz * THZ_TO_CM1


def thz_to_mev(thz: float) -> float:
    return thz * THZ_TO_MEV


def cm1_to_thz(cm1: float) -> float:
    return cm1 * CM1_TO_THZ


def mev_to_thz(mev: float) -> float:
    return mev * MEV_TO_THZ


def format_frequency(thz: float, unit: str = "THz", precision: int = 3) -> str:
    """Format a frequency as 'value unit' with the right precision.

    Imaginary frequencies (negative input) are shown with an 'i' suffix:
        format_frequency(-0.05) -> '0.050i THz'

    Parameters
    ----------
    thz : float
        Frequency in THz (may be negative for imaginary modes).
    unit : {'THz', 'cm-1', 'meV'}
    precision : int
        Decimal places.
    """
    is_imag = thz < 0
    val_thz = abs(thz)
    if unit == "THz":
        val = val_thz
    elif unit in ("cm-1", "cm^-1", "cm⁻¹"):
        val = thz_to_cm1(val_thz)
    elif unit == "meV":
        val = thz_to_mev(val_thz)
    else:
        raise ValueError(f"Unknown frequency unit: {unit}")
    suffix = "i" if is_imag else ""
    return f"{val:.{precision}f}{suffix} {unit}"


# ============================================================
# LaTeX-style label prettification
# ============================================================
# Phonopy emits high-symmetry-point labels in LaTeX math mode (e.g.
# '$\Gamma$', '$\Sigma$', '$M_1$'). For UI display we want compact
# Unicode strings. The web tool will use the same mapping.

_LATEX_NAMED = {
    # Greek (capital + lowercase)
    "\\Gamma": "Γ", "\\Delta": "Δ", "\\Theta": "Θ", "\\Lambda": "Λ",
    "\\Xi":    "Ξ", "\\Pi":    "Π", "\\Sigma":  "Σ", "\\Upsilon": "Υ",
    "\\Phi":   "Φ", "\\Psi":   "Ψ", "\\Omega":  "Ω",
    "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\delta": "δ",
    "\\epsilon": "ε", "\\zeta": "ζ", "\\eta": "η", "\\theta": "θ",
    "\\iota": "ι", "\\kappa": "κ", "\\lambda": "λ", "\\mu": "μ",
    "\\nu": "ν", "\\xi": "ξ", "\\pi": "π", "\\rho": "ρ",
    "\\sigma": "σ", "\\tau": "τ", "\\phi": "φ", "\\chi": "χ",
    "\\psi": "ψ", "\\omega": "ω",
    # Common math typeset commands
    "\\prime": "′",     # X' for prime-marked variant high-symmetry points
    "\\ast":   "∗",
}

# Keep the old name for back-compat with any external code
_LATEX_GREEK = _LATEX_NAMED


_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


import re as _re

_LATEX_NAMED_SORTED = sorted(_LATEX_NAMED.items(), key=lambda kv: -len(kv[0]))
_SUB_RE = _re.compile(r"_\{([^}]*)\}|_([A-Za-z0-9])")
_SUP_RE = _re.compile(r"\^\{([^}]*)\}|\^([A-Za-z0-9])")
_REMNANT_BACKSLASH_RE = _re.compile(r"\\([A-Za-z]+)")
# Pre-compiled comma-trailing pattern for labels with trailing punctuation
_TRAILING_PUNCT_RE = _re.compile(r"[,;.]+$")


def _prettify_atom(s: str) -> str:
    """Prettify a single atomic label (no pipe characters)."""
    s = s.strip()
    if not s:
        return s

    # Strip outer $...$ tolerantly — allow trailing punctuation after the
    # second $ (phonopy emits things like '$\Gamma$,').
    if s.startswith("$"):
        # find the matching closing $
        m = _re.match(r"^\$([^$]*)\$(.*)$", s)
        if m is not None:
            inner, tail = m.group(1), m.group(2)
            s = inner + tail

    # Named LaTeX commands (longest-first) — done FIRST so '\prime' inside
    # '^{\prime}' becomes '′' BEFORE the superscript regex eats the body
    for tex, uni in _LATEX_NAMED_SORTED:
        s = s.replace(tex, uni)

    # Subscripts and superscripts. If the body is a plain digit, render with
    # Unicode digit subscripts/superscripts. Otherwise, if the body is just
    # a punctuation/symbol character (like '′' from \prime), output the
    # symbol directly without the '_' or '^' marker — looks cleaner.
    def _sub_repl(m):
        body = m.group(1) or m.group(2) or ""
        if body.isdigit():
            return body.translate(_SUBSCRIPT_DIGITS)
        if len(body) == 1 and not body.isalnum():
            return body
        return "_" + body
    s = _SUB_RE.sub(_sub_repl, s)

    def _sup_repl(m):
        body = m.group(1) or m.group(2) or ""
        if body.isdigit():
            return body.translate(_SUPERSCRIPT_DIGITS)
        if len(body) == 1 and not body.isalnum():
            return body
        return "^" + body
    s = _SUP_RE.sub(_sup_repl, s)

    # Tidy braces and remaining backslash-commands
    s = s.replace("{", "").replace("}", "")
    s = _REMNANT_BACKSLASH_RE.sub(r"\1", s)
    s = s.replace("$", "")
    s = _TRAILING_PUNCT_RE.sub("", s)
    return s


def prettify_label(label: str) -> str:
    """Convert a LaTeX-style high-symmetry-point label to compact Unicode.

    Handles the common phonopy output forms:

    - ``$\\Gamma$`` → ``Γ``, ``$X^{\\prime}$`` → ``X′``, ``$T_2$`` → ``T₂``
    - Trailing punctuation (``,`` ``;`` ``.``) from compound segment ends
      is stripped: ``$\\Gamma$,`` → ``Γ``
    - Pipe-separated compound labels (``M|L``, ``X,|Y``) are prettified
      per side and rejoined with ``|``
    - Unknown LaTeX commands fall back gracefully (e.g. ``$\\Foo$`` → ``Foo``)
    - Non-string or empty input returns ``""``
    - Any unexpected error returns the original string rather than raising
      (so a bad label can't break panel drawing)
    """
    if not isinstance(label, str):
        return ""
    if not label:
        return label
    try:
        if "|" in label:
            return "|".join(_prettify_atom(part) for part in label.split("|"))
        return _prettify_atom(label)
    except Exception:
        # Never let a buggy label break the UI
        return label
