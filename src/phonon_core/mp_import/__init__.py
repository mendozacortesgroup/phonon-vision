"""phonon_core.mp_import — Materials Project ingestion (Track A: pheasy FC).

Turns a harvested MP *pheasy* document set (force constants + metadata) into
the phonopy file triple the rest of the toolchain already eats —
``phonopy.yaml`` + ``FORCE_CONSTANTS`` + ``band.conf`` — so an MP material is
format-indistinguishable from a local phonopy calculation downstream.

PURITY CONTRACT: this package imports ``json``/``os`` (stdlib), ``numpy`` and
(lazily, inside functions) ``phonopy`` — NOTHING ELSE. No pymatgen, no mp-api:
the harvest script owns those, in the user's MP environment. The pymatgen
``Structure.as_dict()`` payload is parsed BY HAND.

NAC IS WIRED OFF (Track A): Born charges and the dielectric tensor are carried
into a JSON sidecar (``mp_nac.json`` — deliberately NOT named ``BORN``, which
phonopy/A1 auto-discovers) with ``nac_applied: false``. LO-TO is the 0.6.91
arc.
"""
from phonon_core.mp_import.pheasy import (  # noqa: F401
    MPImportError,
    convert_pheasy,
    parse_mp_structure,
)
from phonon_core.mp_import.dfpt import convert_dfpt  # noqa: F401
from phonon_core.mp_import.nac import (  # noqa: F401
    PHEASY_NAC_FACTOR,
    gamma_nac_directions,
    write_phonopy_born,
)

__all__ = ["MPImportError", "PHEASY_NAC_FACTOR", "convert_dfpt",
           "convert_pheasy", "gamma_nac_directions", "parse_mp_structure",
           "write_phonopy_born"]
